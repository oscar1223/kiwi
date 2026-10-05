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
	// LiveInterval is how often the live progress message may be edited.
	// Zero means DefaultLiveInterval.
	LiveInterval time.Duration

	mu sync.Mutex
}

// Busy is the reply to a message that arrives while a turn is running.
const Busy = "Sigo con la tarea anterior. Te escribo cuando acabe; mándame esto después."

// Handle is a Handler: it answers commands and runs everything else as a turn.
// While a turn runs, conv shows its progress; conv may be nil.
func (s *Session) Handle(ctx context.Context, msg Message, conv Conversation) string {
	if !s.mu.TryLock() {
		return Busy
	}
	defer s.mu.Unlock()

	text := strings.TrimSpace(msg.Text)
	switch command(text) {
	case "/start", "/help":
		return fmt.Sprintf("Kiwi trabajando en %s.\n\nEscríbeme una tarea: mientras trabajo, un mensaje va mostrando lo que hago, y al terminar te contesto.\n/new empieza una conversación nueva.", s.WorkDir)
	case "/new":
		if err := s.Reset(ctx); err != nil {
			return "No he podido empezar una conversación nueva: " + err.Error()
		}
		s.History = nil
		return "Conversación nueva. ¿Qué hacemos?"
	}

	return s.turn(ctx, text, conv)
}

func (s *Session) turn(ctx context.Context, input string, conv Conversation) string {
	s.logf("turn: %s", truncateRunes(input, 80))

	obs := observers{&logObserver{s: s}}
	var live *liveObserver
	if conv != nil {
		live = startLive(ctx, conv, s.LiveInterval, s.Log)
		obs = append(obs, live)
	}

	res, err := s.Agent.Run(ctx, input, s.History, obs)
	if live != nil {
		live.finish(ctx, err == nil)
	}
	if err != nil {
		if ctx.Err() != nil {
			// Shutting down: nothing to tell the user that would reach them.
			return ""
		}
		s.logf("turn failed: %v", err)
		if errors.Is(err, agent.ErrMaxSteps) {
			return "He llegado al límite de pasos sin terminar. Dime si sigo."
		}
		return "Error: " + err.Error()
	}

	history, err := s.Save(ctx, res.Messages)
	if err != nil {
		// The answer is still worth sending; only the memory of it is lost.
		s.logf("could not save the turn: %v", err)
		history = append(append([]llm.Message(nil), s.History...), res.Messages...)
	}
	s.History = history
	s.logf("turn done in %d step(s)", res.Steps)

	if strings.TrimSpace(res.Text) == "" {
		return "Hecho."
	}
	return res.Text
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
