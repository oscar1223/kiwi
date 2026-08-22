package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/textarea"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/prompt"
	"github.com/oscar1223/kiwi/internal/session"
)

// Options configures a TUI session.
type Options struct {
	Agent      *agent.Agent
	Broker     *permission.Broker
	WorkDir    string
	ModelLabel string
	// PromptOptions is kept so the system prompt can be rebuilt when the mode
	// changes: the model is told what it is allowed to do.
	PromptOptions prompt.Options
	// Events is the shared stream, already handed to the broker as its
	// permission.Decider.
	Events *Events
	// History seeds the conversation, e.g. from --continue or --resume.
	History []llm.Message
	// Store and SessionID persist every turn. Both nil/empty is valid — the
	// TUI then keeps history in memory only, which is what the tests below
	// use.
	Store     *session.Store
	SessionID string
}

// Model is the Bubble Tea model for Kiwi's terminal interface.
//
// Completed output is printed into the terminal's own scrollback rather than
// held in a viewport, so scrolling, selecting and copying all work the way
// they do for any other command. Only the live tail — the sentence being
// streamed, the spinner, the input box — is re-rendered.
type Model struct {
	opts   Options
	events *Events

	input   textarea.Model
	spinner spinner.Model

	width  int
	height int

	history []llm.Message

	// Turn state. gen increments on every turn so events from a cancelled
	// turn can be recognised and dropped.
	busy   bool
	gen    int
	cancel context.CancelFunc
	began  time.Time

	// tail is the incomplete line currently being streamed. Complete lines
	// are flushed to scrollback as soon as they arrive.
	tail    string
	inFence bool
	spoke   bool

	pending *permission.Request

	// saveTokens serializes background persistence across turns: each
	// persistTurn call takes the single token before writing and returns it
	// after, so two turns' writes can never interleave even though neither
	// blocks the UI. See persistTurn.
	saveTokens chan struct{}

	quitting bool
}

func New(opts Options) *Model {
	ta := textarea.New()
	ta.Placeholder = "Ask anything, or /help"
	ta.ShowLineNumbers = false
	ta.CharLimit = 0
	ta.Prompt = ""
	ta.SetHeight(1)
	ta.MaxHeight = 12
	ta.Focus()

	sp := spinner.New()
	sp.Spinner = spinner.Points
	sp.Style = lipgloss.NewStyle().Foreground(colKiwi)

	ev := opts.Events
	if ev == nil {
		ev = NewEvents()
	}
	tokens := make(chan struct{}, 1)
	tokens <- struct{}{}
	return &Model{
		opts:       opts,
		events:     ev,
		input:      ta,
		spinner:    sp,
		history:    append([]llm.Message(nil), opts.History...),
		saveTokens: tokens,
	}
}

// LogAutoDecision forwards a policy decision to the event stream.
func (m *Model) LogAutoDecision(req *permission.Request, allowed bool) {
	m.events.LogAutoDecision(req, allowed)
}

func (m *Model) Init() tea.Cmd {
	return tea.Batch(
		m.events.next(),
		m.input.Focus(),
		tea.Println(banner(m.opts.ModelLabel, m.opts.WorkDir)),
	)
}

func (m *Model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		m.input.SetWidth(max(20, msg.Width-4))
		return m, nil

	case tea.KeyPressMsg:
		return m.onKey(msg)

	case spinner.TickMsg:
		if !m.busy {
			return m, nil
		}
		var cmd tea.Cmd
		m.spinner, cmd = m.spinner.Update(msg)
		return m, cmd

	case textDeltaMsg:
		if msg.gen != m.gen {
			return m, m.events.next()
		}
		return m, tea.Batch(m.stream(msg.delta), m.events.next())

	case toolCallMsg:
		if msg.gen != m.gen {
			return m, m.events.next()
		}
		return m, tea.Batch(
			tea.Sequence(m.flushTail(), tea.Println(renderToolCall(msg.call))),
			m.events.next(),
		)

	case toolResultMsg:
		if msg.gen != m.gen {
			return m, m.events.next()
		}
		return m, tea.Batch(
			tea.Println(renderToolResult(msg.output, msg.isErr)),
			m.events.next(),
		)

	case turnDoneMsg:
		if msg.gen != m.gen {
			return m, m.events.next()
		}
		// Applied immediately and in memory: the next turn must see what just
		// happened regardless of how long persisting it to disk takes.
		m.history = append(m.history, msg.messages...)
		m.busy = false
		if m.cancel != nil {
			m.cancel()
			m.cancel = nil
		}
		return m, tea.Batch(
			tea.Sequence(m.flushTail(), tea.Println("")),
			m.persistTurn(msg.gen, msg.messages),
			m.events.next(),
		)

	case turnErrMsg:
		if msg.gen != m.gen {
			return m, m.events.next()
		}
		return m, tea.Batch(m.endTurn(msg.err), m.events.next())

	case historyPersistedMsg:
		if msg.err != nil {
			return m, tea.Batch(
				tea.Println(styleWarn.Render("  (session not saved: "+msg.err.Error()+")")),
				m.events.next(),
			)
		}
		// A newer turn may already be running by the time this returns: only
		// adopt the (possibly compacted) result while it is still current,
		// never regress history that has since moved on. Skipping a stale
		// compaction is harmless — it durably landed on disk regardless, and
		// the next compaction cycle will catch up.
		if msg.gen == m.gen && msg.history != nil {
			m.history = msg.history
		}
		return m, m.events.next()

	case permissionMsg:
		// A pending request is answered before another can arrive, because
		// Update is the only consumer and it parks here until resolved.
		m.pending = msg.req
		return m, tea.Batch(tea.Println(renderPermissionPrompt(msg.req)), m.events.next())

	case autoDecisionMsg:
		return m, tea.Batch(tea.Println(renderAutoDecision(msg.req, msg.allowed)), m.events.next())
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m *Model) onKey(msg tea.KeyPressMsg) (tea.Model, tea.Cmd) {
	key := msg.String()

	// A pending permission prompt captures the keyboard: nothing else should
	// happen while a tool is blocked waiting for an answer.
	if m.pending != nil {
		switch key {
		case "y", "Y", "enter":
			req := m.pending
			m.pending = nil
			req.Allow()
			return m, tea.Println(styleDim.Render("  approved"))
		case "n", "N", "esc", "ctrl+c":
			req := m.pending
			m.pending = nil
			req.Deny()
			return m, tea.Println(styleDim.Render("  denied"))
		}
		return m, nil
	}

	switch key {
	case "ctrl+c":
		if m.busy {
			return m, m.cancelTurn()
		}
		m.quitting = true
		return m, tea.Quit

	case "ctrl+d":
		if m.input.Value() == "" {
			m.quitting = true
			return m, tea.Quit
		}

	case "esc":
		if m.busy {
			return m, m.cancelTurn()
		}
		return m, nil

	case "shift+tab":
		next := m.opts.Broker.Mode().Next()
		m.opts.Broker.SetMode(next)
		m.applyMode(next)
		return m, tea.Println(renderModeChange(next))

	case "enter":
		if m.busy {
			return m, nil
		}
		text := strings.TrimSpace(m.input.Value())
		if text == "" {
			return m, nil
		}
		m.input.Reset()
		return m, m.submit(text)
	}

	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

// submit starts a new turn.
func (m *Model) submit(text string) tea.Cmd {
	if cmd, handled := m.command(text); handled {
		return cmd
	}

	m.gen++
	gen := m.gen
	m.busy = true
	m.began = time.Now()
	m.spoke = false
	m.tail = ""
	m.inFence = false

	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel

	history := append([]llm.Message(nil), m.history...)
	go runTurn(ctx, m.opts.Agent, gen, text, history, m.events)

	return tea.Batch(
		tea.Println(bullet(styleUser.Render(">"), styleUser.Render(text))),
		m.spinner.Tick,
	)
}

// persistTurn saves one turn's messages in the background and reports the
// resulting history (compacted, if that fired) via historyPersistedMsg.
//
// It never blocks Update: the token channel serializes writes across
// goroutines by itself, so turns started back-to-back by a fast typist still
// land on disk in order without the UI ever waiting on I/O or on a
// summarization call to the model.
func (m *Model) persistTurn(gen int, turnMessages []llm.Message) tea.Cmd {
	store := m.opts.Store
	if store == nil {
		return nil
	}
	sessionID := m.opts.SessionID
	provider := m.opts.Agent.Provider
	tokens := m.saveTokens

	return func() tea.Msg {
		<-tokens
		defer func() { tokens <- struct{}{} }()

		history, err := session.Persist(context.Background(), store, sessionID, provider, turnMessages)
		if err != nil {
			return historyPersistedMsg{gen: gen, err: err}
		}
		return historyPersistedMsg{gen: gen, history: history}
	}
}

// cancelTurn tears down the running turn. Bumping the generation makes every
// event still in flight irrelevant, and cancelling the context stops the model
// stream and kills any child process a tool started.
func (m *Model) cancelTurn() tea.Cmd {
	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}
	m.gen++
	m.busy = false
	tail := m.tail
	m.tail = ""

	var cmds []tea.Cmd
	if tail != "" {
		cmds = append(cmds, tea.Println(styleKiwi.Render("  "+tail)))
	}
	cmds = append(cmds, tea.Println(styleWarn.Render("  cancelled")))
	return tea.Sequence(cmds...)
}

func (m *Model) endTurn(err error) tea.Cmd {
	m.busy = false
	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}

	cmds := []tea.Cmd{m.flushTail()}
	if err != nil {
		if errors.Is(err, context.Canceled) {
			cmds = append(cmds, tea.Println(styleWarn.Render("  cancelled")))
		} else {
			cmds = append(cmds, tea.Println(bullet(styleErr.Render("✗"), styleErr.Render(err.Error()))))
		}
	}
	cmds = append(cmds, tea.Println("")) // breathing room before the next turn
	return tea.Sequence(cmds...)
}

// stream appends a delta and flushes any lines it completed.
//
// Printing line by line is what keeps the answer in the terminal's scrollback
// instead of a repainted region: each finished line is emitted once and never
// touched again.
func (m *Model) stream(delta string) tea.Cmd {
	m.tail += delta
	if !strings.Contains(m.tail, "\n") {
		return nil
	}

	parts := strings.Split(m.tail, "\n")
	complete, remainder := parts[:len(parts)-1], parts[len(parts)-1]
	m.tail = remainder

	var cmds []tea.Cmd
	for _, line := range complete {
		cmds = append(cmds, tea.Println(m.renderLine(line)))
	}
	// Sequence, not Batch: Batch runs commands concurrently with no ordering
	// guarantee, which would shuffle the lines of an answer.
	return tea.Sequence(cmds...)
}

// flushTail emits the last partial line at the end of a turn.
func (m *Model) flushTail() tea.Cmd {
	if m.tail == "" {
		return nil
	}
	line := m.tail
	m.tail = ""
	return tea.Println(m.renderLine(line))
}

// renderLine styles one line of assistant output. Full markdown rendering
// needs whole blocks, which streaming does not have; tracking fences is what
// can be done a line at a time, and it covers the case that matters most in a
// coding agent.
func (m *Model) renderLine(line string) string {
	prefix := "  "
	if !m.spoke {
		prefix = styleKiwi.Render("● ")
		m.spoke = true
	}

	if strings.HasPrefix(strings.TrimSpace(line), "```") {
		m.inFence = !m.inFence
		return prefix + styleDim.Render(line)
	}
	if m.inFence {
		return prefix + styleCode.Render(line)
	}
	return prefix + styleKiwi.Render(line)
}

// applyMode rebuilds the system prompt so the model knows what it may do.
func (m *Model) applyMode(mode permission.Mode) {
	opts := m.opts.PromptOptions
	opts.ModeInstructions = mode.Instructions()
	m.opts.Agent.System = prompt.Build(opts)
}

func (m *Model) View() tea.View {
	var b strings.Builder

	if m.pending != nil {
		b.WriteString(styleWarn.Render("  allow? [y/N] "))
		b.WriteString("\n")
	} else {
		if m.tail != "" {
			b.WriteString("  " + styleKiwi.Render(m.tail) + "\n")
		}
		if m.busy {
			b.WriteString(sprintf("%s %s\n",
				m.spinner.View(),
				styleDim.Render(sprintf("working… %s · esc to cancel", elapsed(m.began)))))
		}
	}

	b.WriteString(m.statusLine())
	b.WriteString("\n")
	b.WriteString(stylePrompt.Render("› "))
	b.WriteString(m.input.View())

	v := tea.NewView(b.String())
	if m.pending == nil {
		v.Cursor = m.input.Cursor()
	}
	return v
}

func (m *Model) statusLine() string {
	mode := m.opts.Broker.Mode()
	parts := []string{
		modeStyle(mode).Render(mode.Label()),
		styleDim.Render(m.opts.ModelLabel),
	}
	if len(m.history) > 0 {
		parts = append(parts, styleDim.Render(sprintf("%d msgs", len(m.history))))
	}
	line := strings.Join(parts, styleDim.Render(" · "))
	hint := styleDim.Render("shift+tab: mode")
	gap := m.width - lipgloss.Width(line) - lipgloss.Width(hint)
	if gap < 1 {
		return line
	}
	return line + strings.Repeat(" ", gap) + hint
}

func elapsed(since time.Time) string {
	d := time.Since(since).Round(time.Second)
	if d < time.Minute {
		return sprintf("%ds", int(d.Seconds()))
	}
	return d.String()
}

func banner(model, workDir string) string {
	title := lipgloss.NewStyle().Foreground(colKiwi).Bold(true).Render("🥝 kiwi")
	return sprintf("\n%s  %s\n%s\n",
		title,
		styleDim.Render(model),
		styleDim.Render("  "+workDir))
}

func renderToolCall(call llm.ToolCall) string {
	summary := toolSummary(call.Name, call.Input)
	line := styleTool.Render(call.Name)
	if summary != "" {
		line += " " + styleToolArgs.Render(summary)
	}
	return bullet(styleTool.Render("⏺"), line)
}

func renderToolResult(output string, isErr bool) string {
	style, marker := styleDim, "  ↳"
	if isErr {
		style, marker = styleErr, "  ✗"
	}
	return sprintf("%s %s", styleDim.Render(marker), style.Render(oneLine(output, 160)))
}

func renderModeChange(mode permission.Mode) string {
	return bullet(styleDim.Render("·"),
		modeStyle(mode).Render(mode.Label()+" mode")+" "+styleDim.Render(modeHint(mode)))
}

func modeHint(mode permission.Mode) string {
	switch mode {
	case permission.ModePlan:
		return "read-only; edits blocked"
	case permission.ModeWork:
		return "edits apply without asking"
	default:
		return "every action is confirmed"
	}
}

func renderAutoDecision(req *permission.Request, allowed bool) string {
	verb, style := "blocked", styleWarn
	if allowed {
		verb, style = "auto-approved", styleDim
	}
	return sprintf("  %s %s",
		style.Render(verb+" ("+req.Mode.Label()+")"),
		styleDim.Render(oneLine(req.Detail, 100)))
}

func renderPermissionPrompt(req *permission.Request) string {
	var b strings.Builder

	marker := styleWarn.Render("?")
	title := req.Detail
	if req.Dangerous {
		title = styleErr.Render("⚠ ") + title
	}
	b.WriteString(bullet(marker, styleWarn.Render(title)))

	if req.Diff != "" {
		b.WriteString("\n")
		b.WriteString(renderDiff(req.Diff, 40))
	}
	return b.String()
}

// command handles slash commands typed into the prompt.
func (m *Model) command(text string) (tea.Cmd, bool) {
	if !strings.HasPrefix(text, "/") {
		return nil, false
	}
	switch strings.Fields(text)[0] {
	case "/help":
		return tea.Println(helpText()), true
	case "/clear":
		m.history = nil
		return tea.Batch(tea.ClearScreen, tea.Println(banner(m.opts.ModelLabel, m.opts.WorkDir))), true
	case "/ask", "/plan", "/work":
		mode := permission.Mode(strings.TrimPrefix(strings.Fields(text)[0], "/"))
		m.opts.Broker.SetMode(mode)
		m.applyMode(mode)
		return tea.Println(renderModeChange(mode)), true
	case "/quit", "/exit":
		m.quitting = true
		return tea.Quit, true
	}
	return tea.Println(styleErr.Render("  unknown command: " + text)), true
}

func helpText() string {
	rows := [][2]string{
		{"/ask /plan /work", "switch permission mode"},
		{"/clear", "forget the conversation"},
		{"/help", "this list"},
		{"/quit", "exit"},
		{"shift+tab", "cycle mode"},
		{"esc", "cancel the running turn"},
		{"ctrl+c", "cancel, or exit when idle"},
	}
	var b strings.Builder
	for _, r := range rows {
		fmt.Fprintf(&b, "  %s  %s\n",
			styleTool.Render(fmt.Sprintf("%-18s", r[0])),
			styleDim.Render(r[1]))
	}
	return b.String()
}
