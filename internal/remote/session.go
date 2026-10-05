package remote

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm"
)

// Session is the conversation the bot drives: one agent working on one
// directory, its history, and where that history is saved.
//
// It runs one turn at a time, whoever asks. Two turns at once would be two
// agents editing the same files, so a message that arrives mid-turn is
// answered with "busy" instead of being run.
type Session struct {
	Agent   *agent.Agent
	WorkDir string
	History []llm.Message

	// Save persists a finished turn and returns the history to continue
	// from, which may be compacted.
	Save func(ctx context.Context, turn []llm.Message) ([]llm.Message, error)
	// Reset starts a new, empty conversation (the /new command).
	Reset func(ctx context.Context) error

	// Log reports progress to the operator's terminal. May be nil.
	Log func(string)
	// Approver, when set, is the permission.Decider of Agent's broker: it
	// is pointed at the chat of each turn so questions reach whoever asked.
	Approver *Approver
	// LiveInterval is how often the live progress message may be edited.
	// Zero means DefaultLiveInterval.
	LiveInterval time.Duration

	// Cron answers /cron. Nil means scheduled tasks are off.
	Cron *Scheduler
	// NewRun starts a separate saved session for a scheduled run and returns
	// how to save into it. Nil means scheduled runs are not saved.
	NewRun func(ctx context.Context) (save func(ctx context.Context, turn []llm.Message) error, err error)

	// turnSem holds one token while a turn runs: one turn at a time, which a
	// scheduled run can wait for and a message cannot.
	once    sync.Once
	turnSem chan struct{}
}

func (s *Session) sem() chan struct{} {
	s.once.Do(func() { s.turnSem = make(chan struct{}, 1) })
	return s.turnSem
}

// Busy is the reply to a message that arrives while a turn is running.
const Busy = "Sigo con la tarea anterior. Te escribo cuando acabe; mándame esto después."

// Handle is a Handler: it answers commands and runs everything else as a turn.
// While a turn runs, conv shows its progress; conv may be nil.
func (s *Session) Handle(ctx context.Context, msg Message, conv Conversation) string {
	text := strings.TrimSpace(msg.Text)

	// /cron only touches the job list, so it works mid-turn too.
	if command(text) == "/cron" {
		if s.Cron == nil {
			return "Las tareas programadas no están activadas en este kiwi serve."
		}
		_, args, _ := strings.Cut(text, " ")
		return s.Cron.Command(ctx, msg.Chat.ID, args)
	}

	select {
	case s.sem() <- struct{}{}:
		defer func() { <-s.sem() }()
	default:
		return Busy
	}

	switch command(text) {
	case "/start", "/help":
		return fmt.Sprintf("Kiwi trabajando en %s.\n\nEscríbeme una tarea: mientras trabajo, un mensaje va mostrando lo que hago, y al terminar te contesto.\n/new empieza una conversación nueva.\n/cron programa tareas que se ejecutan solas.", s.WorkDir)
	case "/new":
		if err := s.Reset(ctx); err != nil {
			return "No he podido empezar una conversación nueva: " + err.Error()
		}
		s.History = nil
		return "Conversación nueva. ¿Qué hacemos?"
	}

	reply, turn := s.turn(ctx, text, conv, s.History)
	if turn != nil {
		history, err := s.Save(ctx, turn)
		if err != nil {
			// The answer is still worth sending; only the memory of it is lost.
			s.logf("could not save the turn: %v", err)
			history = append(append([]llm.Message(nil), s.History...), turn...)
		}
		s.History = history
	}
	return reply
}

// RunScheduled runs a scheduled task as a turn of its own: it waits for any
// turn in progress, starts from an empty history so the context does not grow
// from one run to the next, and is saved as a separate session. It does not
// touch the chat's conversation.
func (s *Session) RunScheduled(ctx context.Context, prompt string, conv Conversation) string {
	select {
	case s.sem() <- struct{}{}:
		defer func() { <-s.sem() }()
	case <-ctx.Done():
		return ""
	}

	reply, turn := s.turn(ctx, prompt, conv, nil)
	if turn != nil && s.NewRun != nil {
		save, err := s.NewRun(ctx)
		if err == nil {
			err = save(ctx, turn)
		}
		if err != nil {
			s.logf("could not save the scheduled run: %v", err)
		}
	}
	return reply
}

// turn runs the agent once and returns the reply, and the turn's messages if
// it completed (nil if it failed). The caller holds the turn semaphore.
func (s *Session) turn(ctx context.Context, input string, conv Conversation, history []llm.Message) (string, []llm.Message) {
	s.logf("turn: %s", truncateRunes(input, 80))

	if s.Approver != nil && conv != nil {
		s.Approver.attach(conv)
		defer s.Approver.detach()
	}

	obs := observers{&logObserver{s: s}}
	var live *liveObserver
	if conv != nil {
		live = startLive(ctx, conv, s.LiveInterval, s.Log)
		obs = append(obs, live)
	}

	res, err := s.Agent.Run(ctx, input, history, obs)
	if live != nil {
		live.finish(ctx, err == nil)
	}
	if err != nil {
		if ctx.Err() != nil {
			// Shutting down: nothing to tell the user that would reach them.
			return "", nil
		}
		s.logf("turn failed: %v", err)
		if errors.Is(err, agent.ErrMaxSteps) {
			return "He llegado al límite de pasos sin terminar. Dime si sigo.", nil
		}
		return "Error: " + err.Error(), nil
	}
	s.logf("turn done in %d step(s)", res.Steps)

	if strings.TrimSpace(res.Text) == "" {
		return "Hecho.", res.Messages
	}
	return res.Text, res.Messages
}

func (s *Session) logf(format string, args ...any) {
	if s.Log != nil {
		s.Log(fmt.Sprintf(format, args...))
	}
}

// command returns the bot command a message starts with, without any
// @botname suffix Telegram adds in some clients, or "" if it is not one.
func command(text string) string {
	if !strings.HasPrefix(text, "/") {
		return ""
	}
	cmd, _, _ := strings.Cut(strings.Fields(text)[0], "@")
	return strings.ToLower(cmd)
}

// observers fans every event out to several observers.
type observers []agent.Observer

func (os observers) OnText(d string) {
	for _, o := range os {
		o.OnText(d)
	}
}

func (os observers) OnToolCall(c llm.ToolCall) {
	for _, o := range os {
		o.OnToolCall(c)
	}
}

func (os observers) OnToolResult(c llm.ToolCall, out string, isErr bool) {
	for _, o := range os {
		o.OnToolResult(c, out, isErr)
	}
}

func (os observers) OnUsage(u llm.Usage) {
	for _, o := range os {
		o.OnUsage(u)
	}
}

// logObserver reports tool calls to the operator's terminal. The Telegram
// user only gets the final answer for now. It may be called from several
// goroutines at once (task subagents), which Session.logf tolerates as long
// as Log does.
type logObserver struct{ s *Session }

func (o *logObserver) OnText(string) {}
func (o *logObserver) OnToolCall(call llm.ToolCall) {
	o.s.logf("● %s %s", call.Name, truncateRunes(string(call.Input), 100))
}
func (o *logObserver) OnToolResult(call llm.ToolCall, _ string, isErr bool) {
	if isErr {
		o.s.logf("  %s failed", call.Name)
	}
}
func (o *logObserver) OnUsage(llm.Usage) {}

func truncateRunes(s string, n int) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n-1]) + "…"
}
