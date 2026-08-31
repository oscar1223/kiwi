package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"charm.land/bubbles/v2/spinner"
	"charm.land/bubbles/v2/textarea"
	"charm.land/bubbles/v2/textinput"
	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/prompt"
	"github.com/oscar1223/kiwi/internal/session"
	"github.com/oscar1223/kiwi/internal/telemetry"
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
	// BaseContext seeds every turn's cancellable context. It is where
	// process-wide values live — telemetry's OS-user id, in particular —
	// that a turn's own context.WithCancel must inherit to keep traces
	// attributed correctly. Nil falls back to context.Background(), which is
	// what the tests below use.
	BaseContext context.Context
	// Rebuild reconstructs the agent from current on-disk configuration —
	// model profile, MCP servers, skills — after a /model, /mcp, or /skill
	// change persists something new. Nil is valid: those commands then
	// report success without the agent actually changing underneath them,
	// which is what the tests below rely on.
	Rebuild func() (*agent.Agent, string, error)
	// NeedsOnboarding is true when Agent is nil because no model provider is
	// configured yet — the shape of a brand-new install. Init runs the
	// setup wizard instead of the normal banner in that case.
	NeedsOnboarding bool
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
	// sessionUsage accumulates token usage across every turn of this TUI
	// session (not the provider's own lifetime total) — "how much have I
	// spent since I opened kiwi", shown in the status line.
	sessionUsage llm.Usage

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

	// activePick/activeText hold an open command-flow prompt (from /model,
	// /config, /mcp, /skill, /memory). Mutually exclusive with each other and
	// with pending: only one modal is ever on screen.
	activePick *pickState
	activeText *textState
	// activeQuestion holds an open ask_questions prompt from the model
	// itself (see internal/tools.AskQuestionsTool). Mutually exclusive with
	// the above for the same reason.
	activeQuestion *questionState
	// flowBusy is true while a /model, /config, /mcp, /skill or /memory flow
	// is running in its own goroutine. Gates chat submission and starting a
	// second flow the same way busy gates them for a running turn.
	flowBusy bool

	// cmdSuggestIndex is the highlighted row in the "/" autocomplete list;
	// lastSlashInput is what the input held the last time it was updated, so
	// the index can be reset to 0 exactly when the filtered list changes
	// rather than on every keystroke regardless.
	cmdSuggestIndex int
	lastSlashInput  string

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
	if m.opts.NeedsOnboarding {
		// No banner: onboardingFlow opens with its own welcome and takes
		// over immediately, the same way any other flow does — just started
		// automatically instead of waiting for the user to type a command.
		m.runFlow(m.onboardingFlow)
		return tea.Batch(m.events.next(), m.input.Focus())
	}
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

	case tea.PasteMsg:
		return m, m.onPaste(msg)

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
		m.sessionUsage.InputTokens += msg.usage.InputTokens
		m.sessionUsage.OutputTokens += msg.usage.OutputTokens
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

	case *pickRequest:
		m.activePick = &pickState{req: msg}
		return m, m.events.next()

	case *textRequest:
		ti := textinput.New()
		ti.Placeholder = msg.placeholder
		ti.SetValue(msg.defaultVal)
		if msg.secret {
			ti.EchoMode = textinput.EchoPassword
		}
		ti.Focus()
		ti.SetWidth(max(20, m.width-4))
		m.activeText = &textState{req: msg, input: ti}
		return m, m.events.next()

	case *questionRequest:
		m.activeQuestion = &questionState{req: msg}
		return m, m.events.next()

	case systemMsg:
		return m, tea.Batch(tea.Println(bullet(styleDim.Render("·"), styleDim.Render(msg.text))), m.events.next())

	case errMsg:
		return m, tea.Batch(tea.Println(bullet(styleErr.Render("✗"), styleErr.Render(msg.err.Error()))), m.events.next())

	case printLinesMsg:
		prints := make([]tea.Cmd, 0, len(msg.lines))
		for _, l := range msg.lines {
			prints = append(prints, tea.Println(l))
		}
		return m, tea.Batch(tea.Sequence(prints...), m.events.next())

	case clearHistoryMsg:
		m.history = nil
		return m, tea.Batch(tea.Println(styleDim.Render("  memory cleared")), m.events.next())

	case historyCompactedMsg:
		m.history = msg.history
		line := sprintf("  compacted: %d messages → %d", msg.before, len(msg.history))
		return m, tea.Batch(tea.Println(styleDim.Render(line)), m.events.next())

	case sessionSwitchedMsg:
		m.opts.SessionID = msg.sessionID
		m.history = msg.history
		line := styleDim.Render("  switched to session " + msg.sessionID + " (" + msg.title + ")")
		return m, tea.Batch(tea.Println(line), m.events.next())

	case requestRebuildMsg:
		return m, tea.Batch(m.rebuildAgent(), m.events.next())

	case agentRebuiltMsg:
		m.opts.Agent = msg.agent
		m.opts.ModelLabel = msg.modelLabel
		return m, tea.Batch(tea.Println(styleDim.Render("  reloaded: "+msg.modelLabel)), m.events.next())

	case flowDoneMsg:
		m.flowBusy = false
		return m, m.events.next()
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

	if m.activePick != nil {
		return m, m.onPickKey(key)
	}
	if m.activeText != nil {
		return m, m.onTextKey(msg, key)
	}
	if m.activeQuestion != nil {
		return m, m.onQuestionKey(msg, key)
	}

	// Slash-command autocomplete: while the input is "/" plus an unfinished
	// command name, arrows browse the filtered list and enter/tab complete
	// to the highlighted one instead of submitting. Once the text exactly
	// matches a known command, enter falls through to the normal submit path
	// below — typing the whole thing by hand and hitting enter must still
	// just run it.
	if suggestions := m.slashSuggestions(); len(suggestions) > 0 {
		if m.input.Value() != m.lastSlashInput {
			m.cmdSuggestIndex = 0
			m.lastSlashInput = m.input.Value()
		}
		if m.cmdSuggestIndex >= len(suggestions) {
			m.cmdSuggestIndex = len(suggestions) - 1
		}
		switch key {
		case "up":
			if m.cmdSuggestIndex > 0 {
				m.cmdSuggestIndex--
			}
			return m, nil
		case "down":
			if m.cmdSuggestIndex < len(suggestions)-1 {
				m.cmdSuggestIndex++
			}
			return m, nil
		case "tab":
			m.input.SetValue(suggestions[m.cmdSuggestIndex].Name + " ")
			m.input.CursorEnd()
			return m, nil
		case "enter":
			if !isKnownCommand(m.input.Value()) {
				m.input.SetValue(suggestions[m.cmdSuggestIndex].Name + " ")
				m.input.CursorEnd()
				return m, nil
			}
			// Falls through to the ordinary "enter" case below: the text
			// already names a real command, so run it.
		}
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
		if m.flowBusy || m.opts.Agent == nil {
			// No agent yet (onboarding still running) means applyMode below
			// would dereference a nil Agent.System — and switching modes
			// mid-flow makes no sense regardless of whether an agent exists.
			return m, nil
		}
		next := m.opts.Broker.Mode().Next()
		m.opts.Broker.SetMode(next)
		m.applyMode(next)
		return m, tea.Println(renderModeChange(next))

	case "enter":
		if m.busy || m.flowBusy || m.opts.Agent == nil {
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

	base := m.opts.BaseContext
	if base == nil {
		base = context.Background()
	}
	base = telemetry.WithSessionID(base, m.opts.SessionID)
	ctx, cancel := context.WithCancel(base)
	m.cancel = cancel

	// The model gets the file contents; the transcript keeps what the user
	// actually typed, so scrollback stays readable no matter how much was
	// attached.
	sent, attached, missing := expandFileMentions(m.opts.WorkDir, text)

	history := append([]llm.Message(nil), m.history...)
	go runTurn(ctx, m.opts.Agent, gen, sent, history, m.events)

	lines := []string{bullet(styleUser.Render(">"), styleUser.Render(text))}
	if len(attached) > 0 {
		lines = append(lines, styleDim.Render("  attached "+strings.Join(attached, ", ")))
	}
	if len(missing) > 0 {
		// Worth saying out loud: a question about a file that was never
		// attached looks identical to one that was, until the answer is wrong.
		lines = append(lines, styleWarn.Render("  could not read "+strings.Join(missing, ", ")))
	}

	return tea.Batch(
		tea.Println(strings.Join(lines, "\n")),
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
// onPickKey drives an open picker: arrows move the highlight, enter resolves
// it, esc cancels. Resolving prints a one-line record into scrollback — the
// same reason permission answers are printed — so what was chosen has a
// permanent trace, not just a state that vanished with the modal.
func (m *Model) onPickKey(key string) tea.Cmd {
	p := m.activePick
	switch key {
	case "up", "k":
		p.up()
		if p.req.onHighlight != nil {
			p.req.onHighlight(p.selected().Value)
		}
		return nil
	case "down", "j":
		p.down()
		if p.req.onHighlight != nil {
			p.req.onHighlight(p.selected().Value)
		}
		return nil
	case "enter":
		m.activePick = nil
		choice := p.selected()
		p.req.resp <- pickResult{value: choice.Value, ok: true}
		return tea.Println(bullet(styleDim.Render("·"), styleDim.Render(p.req.title+": "+choice.Label)))
	case "esc", "ctrl+c":
		m.activePick = nil
		if p.req.onCancel != nil {
			p.req.onCancel()
		}
		p.req.resp <- pickResult{ok: false}
		return tea.Println(styleDim.Render("  cancelled"))
	}
	return nil
}

// onPaste routes a bracketed-paste event to whichever text field is actually
// on screen. Bubbletea delivers paste as its own tea.PasteMsg rather than a
// tea.KeyPressMsg, so without this it always fell through to Update's default
// case and landed in m.input — the bottom chat box — even while a text prompt
// like the MCP "Environment variables" field was open and visibly focused.
func (m *Model) onPaste(msg tea.PasteMsg) tea.Cmd {
	var cmd tea.Cmd
	switch {
	case m.activeText != nil:
		m.activeText.input, cmd = m.activeText.input.Update(msg)
	case m.activeQuestion != nil && m.activeQuestion.otherActive:
		m.activeQuestion.otherInput, cmd = m.activeQuestion.otherInput.Update(msg)
	case m.activePick != nil || m.pending != nil:
		// Neither has anywhere to put pasted text.
	default:
		m.input, cmd = m.input.Update(msg)
	}
	return cmd
}

// onTextKey drives an open text prompt: everything but enter/esc is handed to
// the embedded textinput.
func (m *Model) onTextKey(msg tea.KeyPressMsg, key string) tea.Cmd {
	t := m.activeText
	switch key {
	case "enter":
		m.activeText = nil
		value := t.input.Value()
		t.req.resp <- textResult{value: value, ok: true}
		shown := value
		switch {
		case t.req.secret && value != "":
			shown = styleDim.Render("••••••••")
		case shown == "":
			shown = styleDim.Render("(empty)")
		}
		return tea.Println(bullet(styleDim.Render("·"), styleDim.Render(t.req.title+": ")+shown))
	case "esc", "ctrl+c":
		m.activeText = nil
		t.req.resp <- textResult{ok: false}
		return tea.Println(styleDim.Render("  cancelled"))
	}
	var cmd tea.Cmd
	t.input, cmd = t.input.Update(msg)
	return cmd
}

// onQuestionKey drives an open ask_questions prompt. Single-select: arrows
// move the highlight, enter resolves with that option (or, on the trailing
// "Other" row, opens a text field first). Multi-select: space toggles the
// highlighted option — including "Other", which opens the text field the
// same way — and enter confirms whatever is toggled; enter with nothing
// toggled falls back to the highlighted row, so a multi-select question
// still answers in one keystroke when the user only wants one thing.
func (m *Model) onQuestionKey(msg tea.KeyPressMsg, key string) tea.Cmd {
	qs := m.activeQuestion

	if qs.otherActive {
		switch key {
		case "enter":
			text := strings.TrimSpace(qs.otherInput.Value())
			qs.otherActive = false
			if text == "" {
				return nil
			}
			qs.otherText = text
			if qs.req.q.MultiSelect {
				qs.setSelected(qs.otherIndex(), true)
				return nil
			}
			return m.resolveQuestion([]string{text})
		case "esc", "ctrl+c":
			qs.otherActive = false
			return nil
		}
		var cmd tea.Cmd
		qs.otherInput, cmd = qs.otherInput.Update(msg)
		return cmd
	}

	switch key {
	case "up", "k":
		qs.up()
		return nil
	case "down", "j":
		qs.down()
		return nil
	case "esc", "ctrl+c":
		m.activeQuestion = nil
		qs.req.resp <- questionResult{ok: false}
		return tea.Println(styleDim.Render("  cancelled"))
	case "space":
		if !qs.req.q.MultiSelect {
			return nil
		}
		if qs.index == qs.otherIndex() {
			qs.openOtherInput(m.width)
			return nil
		}
		qs.setSelected(qs.index, !qs.selected[qs.index])
		return nil
	case "enter":
		if qs.index == qs.otherIndex() && (!qs.req.q.MultiSelect || !qs.selected[qs.otherIndex()]) {
			qs.openOtherInput(m.width)
			return nil
		}
		if qs.req.q.MultiSelect {
			values := qs.selectedLabels()
			if qs.selected[qs.otherIndex()] {
				values = append(values, qs.otherText)
			}
			if len(values) == 0 {
				values = []string{qs.req.q.Options[qs.index].Label}
			}
			return m.resolveQuestion(values)
		}
		return m.resolveQuestion([]string{qs.req.q.Options[qs.index].Label})
	}
	return nil
}

// resolveQuestion closes the current ask_questions prompt with the given
// values, printing the same one-line record a pick or text answer leaves.
func (m *Model) resolveQuestion(values []string) tea.Cmd {
	qs := m.activeQuestion
	m.activeQuestion = nil
	qs.req.resp <- questionResult{values: values, ok: true}
	line := qs.req.q.Question + ": " + strings.Join(values, ", ")
	return tea.Println(bullet(styleDim.Render("·"), styleDim.Render(line)))
}

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

	switch {
	case m.pending != nil:
		b.WriteString(styleWarn.Render("  allow? [y/N] "))
		b.WriteString("\n")
	case m.activePick != nil:
		b.WriteString(renderPick(m.activePick))
		b.WriteString("\n")
	case m.activeText != nil:
		b.WriteString(renderTextPrompt(m.activeText))
		b.WriteString("\n")
	case m.activeQuestion != nil:
		b.WriteString(renderQuestion(m.activeQuestion))
		b.WriteString("\n")
	default:
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

	blocked := m.pending != nil || m.activePick != nil || m.activeText != nil || m.activeQuestion != nil
	if !blocked {
		b.WriteString(stylePrompt.Render("› "))
		b.WriteString(m.input.View())

		if suggestions := m.slashSuggestions(); len(suggestions) > 0 {
			b.WriteString("\n")
			b.WriteString(renderSlashSuggestions(suggestions, m.cmdSuggestIndex))
		}
	}

	v := tea.NewView(b.String())
	switch {
	case m.activeText != nil:
		v.Cursor = m.activeText.input.Cursor()
	case m.activeQuestion != nil && m.activeQuestion.otherActive:
		v.Cursor = m.activeQuestion.otherInput.Cursor()
	case !blocked:
		v.Cursor = m.input.Cursor()
	}
	return v
}

// renderPick draws an arrow-navigable list: the highlighted option gets the
// kiwi-coloured marker, everything else stays dim.
func renderPick(p *pickState) string {
	var b strings.Builder
	b.WriteString(styleWarn.Render("  " + p.req.title))
	b.WriteString(styleDim.Render("  (↑↓ enter · esc cancels)"))
	b.WriteString("\n")
	for i, opt := range p.req.options {
		if i == p.index {
			b.WriteString(styleKiwi.Render("  ▸ " + opt.Label))
		} else {
			b.WriteString(styleDim.Render("    " + opt.Label))
		}
		b.WriteString("\n")
	}
	return b.String()
}

func renderTextPrompt(t *textState) string {
	var b strings.Builder
	b.WriteString(styleWarn.Render("  " + t.req.title))
	b.WriteString(styleDim.Render("  (enter confirms · esc cancels)"))
	b.WriteString("\n  ")
	b.WriteString(t.input.View())
	return b.String()
}

// renderQuestion draws one ask_questions prompt: an arrow-navigable list
// like renderPick, plus checkboxes when the question is multi-select and a
// trailing "Other" row that switches to free-text entry.
func renderQuestion(qs *questionState) string {
	q := qs.req.q
	var b strings.Builder
	b.WriteString(styleWarn.Render("  " + q.Question))
	if q.MultiSelect {
		b.WriteString(styleDim.Render("  (↑↓ space toggles · enter confirms · esc cancels)"))
	} else {
		b.WriteString(styleDim.Render("  (↑↓ enter · esc cancels)"))
	}
	b.WriteString("\n")

	if qs.otherActive {
		b.WriteString(styleDim.Render("    type your own answer"))
		b.WriteString("\n  ")
		b.WriteString(qs.otherInput.View())
		return b.String()
	}

	for i, opt := range q.Options {
		line := opt.Label
		if opt.Description != "" {
			line += " — " + opt.Description
		}
		b.WriteString(renderQuestionRow(checkbox(q.MultiSelect, qs.selected[i])+line, i == qs.index))
	}

	other := "Other (type your own)"
	if qs.otherText != "" {
		other += ": " + qs.otherText
	}
	b.WriteString(renderQuestionRow(checkbox(q.MultiSelect, qs.selected[qs.otherIndex()])+other, qs.index == qs.otherIndex()))
	return b.String()
}

func checkbox(multiSelect, checked bool) string {
	switch {
	case !multiSelect:
		return ""
	case checked:
		return "[x] "
	default:
		return "[ ] "
	}
}

func renderQuestionRow(line string, highlighted bool) string {
	if highlighted {
		return styleKiwi.Render("  ▸ "+line) + "\n"
	}
	return styleDim.Render("    "+line) + "\n"
}

// maxSlashSuggestions caps how many rows the "/" autocomplete shows at once,
// so a broad query (bare "/") does not push the input off screen.
const maxSlashSuggestions = 6

func renderSlashSuggestions(suggestions []commandSpec, index int) string {
	shown := suggestions
	if len(shown) > maxSlashSuggestions {
		shown = shown[:maxSlashSuggestions]
	}
	if index < 0 || index >= len(shown) {
		index = 0
	}
	var b strings.Builder
	for i, c := range shown {
		line := sprintf("%-12s %s", c.Name, c.Desc)
		if i == index {
			b.WriteString(styleKiwi.Render("  ▸ " + line))
		} else {
			b.WriteString(styleDim.Render("    " + line))
		}
		if i < len(shown)-1 {
			b.WriteString("\n")
		}
	}
	if len(suggestions) > maxSlashSuggestions {
		fmt.Fprintf(&b, "\n%s", styleDim.Render(sprintf("    … %d more", len(suggestions)-maxSlashSuggestions)))
	}
	return b.String()
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
	if total := m.sessionUsage.InputTokens + m.sessionUsage.OutputTokens; total > 0 {
		parts = append(parts, styleDim.Render(formatTokenCount(total)+" tok"))
	}
	if used, window := m.contextUsage(); window > 0 && used > 0 {
		parts = append(parts, m.contextStyle(used, window).Render(
			sprintf("ctx %d%%", percentOf(used, window))))
	}
	line := strings.Join(parts, styleDim.Render(" · "))
	hint := styleDim.Render("shift+tab: mode")
	gap := m.width - lipgloss.Width(line) - lipgloss.Width(hint)
	if gap < 1 {
		return line
	}
	return line + strings.Repeat(" ", gap) + hint
}

// formatTokenCount renders a token total compactly for the status line:
// exact below 1000, one decimal of "k" above it — precise enough to be
// useful, short enough to leave room for everything else on the line.
func formatTokenCount(n int) string {
	if n < 1000 {
		return sprintf("%d", n)
	}
	return sprintf("%.1fk", float64(n)/1000)
}

// contextUsage estimates how much of the model's context window the next
// request will occupy: the stored conversation plus the system prompt, which
// carries the project instructions, memory, skills and every tool schema and
// is often the larger half early in a session.
//
// It is an estimate (see llm.EstimateMessageTokens) rather than the provider's
// own count, because the number is wanted *before* the next request is sent —
// the point is to warn while there is still room to act on the warning.
func (m *Model) contextUsage() (used, window int) {
	window = llm.ContextWindow(m.opts.ModelLabel)
	used = llm.EstimateMessageTokens(m.history)
	if m.opts.Agent != nil {
		used += llm.EstimateTokens(m.opts.Agent.System)
	}
	return used, window
}

// contextStyle escalates as the window fills. The thresholds sit either side
// of the automatic compaction point: by the time history alone is half the
// window, the next persisted turn will summarize it, and the user is better
// off knowing that is about to happen than discovering it afterwards.
func (m *Model) contextStyle(used, window int) lipgloss.Style {
	switch pct := percentOf(used, window); {
	case pct >= 80:
		return styleWarn
	case pct >= 50:
		return styleKiwi
	default:
		return styleDim
	}
}

func percentOf(part, whole int) int {
	if whole <= 0 {
		return 0
	}
	return part * 100 / whole
}

func elapsed(since time.Time) string {
	d := time.Since(since).Round(time.Second)
	if d < time.Minute {
		return sprintf("%ds", int(d.Seconds()))
	}
	return d.String()
}

var welcomePrompts = []string{
	"explain this repository",
	"fix the failing test",
	"add a README",
}

func banner(model, workDir string) string {
	title := lipgloss.NewStyle().Foreground(colKiwi).Bold(true).Render("🥝 kiwi")

	modes := sprintf("  %s confirms everything · %s is read-only · %s applies edits — cycle with %s",
		modeStyle(permission.ModeAsk).Render("ask"),
		modeStyle(permission.ModePlan).Render("plan"),
		modeStyle(permission.ModeWork).Render("work"),
		styleDim.Render("shift+tab"))

	var examples strings.Builder
	examples.WriteString(styleDim.Render("  try:"))
	for _, p := range welcomePrompts {
		examples.WriteString(sprintf(" \"%s\"", p))
	}

	hint := styleDim.Render("  type ") + styleUser.Render("/") + styleDim.Render(" to see all commands")

	return sprintf("\n%s  %s\n%s\n%s\n%s\n%s\n",
		title,
		styleDim.Render(model),
		styleDim.Render("  "+workDir),
		modes,
		examples.String(),
		hint)
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
// commandSpec is one entry in the "/" autocomplete list and in /help's
// command table — the single source of truth for both, so they cannot drift
// apart.
type commandSpec struct {
	Name string
	Desc string
}

var commandRegistry = []commandSpec{
	{"/ask", "switch to ask mode — confirm everything"},
	{"/plan", "switch to plan mode — read-only"},
	{"/work", "switch to work mode — edits apply automatically"},
	{"/settings", "open the settings menu"},
	{"/model", "switch or manage model profiles"},
	{"/config", "manage .env variables"},
	{"/mcp", "manage MCP servers"},
	{"/skill", "manage skills"},
	{"/theme", "switch the colour theme"},
	{"/sessions", "switch between saved conversations"},
	{"/memory", "view or edit what kiwi remembers"},
	{"/compact", "summarize the conversation to free up context"},
	{"/clear", "forget the conversation"},
	{"/help", "show this list"},
	{"/quit", "exit kiwi"},
}

// filterCommands returns registry entries whose name contains q as a
// substring (case-insensitive), or every entry when q is empty — so typing
// bare "/" shows the full list, narrowing as more is typed.
func filterCommands(q string) []commandSpec {
	q = strings.ToLower(strings.TrimPrefix(q, "/"))
	if q == "" {
		return commandRegistry
	}
	var out []commandSpec
	for _, c := range commandRegistry {
		if strings.Contains(strings.ToLower(strings.TrimPrefix(c.Name, "/")), q) {
			out = append(out, c)
		}
	}
	return out
}

// isKnownCommand reports whether text is exactly one registered command name
// (arguments after a space are fine — /ask and /ask now both count).
func isKnownCommand(text string) bool {
	name := strings.Fields(strings.TrimSpace(text))
	if len(name) == 0 {
		return false
	}
	for _, c := range commandRegistry {
		if c.Name == name[0] {
			return true
		}
	}
	return false
}

// slashSuggestions returns the autocomplete list for the current input, or
// nil when it does not apply — not focused on a "/"-prefixed command, or
// another modal already owns the keyboard.
func (m *Model) slashSuggestions() []commandSpec {
	if m.pending != nil || m.activePick != nil || m.activeText != nil || m.activeQuestion != nil {
		return nil
	}
	v := m.input.Value()
	if !strings.HasPrefix(v, "/") || strings.ContainsAny(v, " \n") {
		return nil
	}
	return filterCommands(v)
}

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
	case "/model":
		return m.runFlow(m.modelFlow), true
	case "/config":
		return m.runFlow(m.configFlow), true
	case "/mcp":
		return m.runFlow(m.mcpFlow), true
	case "/skill":
		return m.runFlow(m.skillFlow), true
	case "/theme":
		return m.runFlow(m.themeFlow), true
	case "/sessions":
		return m.runFlow(m.sessionsFlow), true
	case "/compact":
		snapshot := append([]llm.Message(nil), m.history...)
		return m.runFlow(func(ctx context.Context) { m.compactFlow(ctx, snapshot) }), true

	case "/memory":
		snapshot := append([]llm.Message(nil), m.history...)
		return m.runFlow(func(ctx context.Context) { m.memoryFlow(ctx, snapshot) }), true
	case "/settings":
		return m.runFlow(m.settingsFlow), true
	}
	return tea.Println(styleErr.Render("  unknown command: " + text)), true
}

// runFlow launches a /model, /config, /mcp, /skill or /memory flow in its own
// goroutine. Flows read like a straight-line script — ask this, act on that —
// by blocking on Events.Pick/Text/Confirm the same way a tool call blocks on
// permission.Broker.Ask; Update stays the only goroutine that ever touches
// Model state, everything a flow decides comes back as a message.
func (m *Model) runFlow(fn func(ctx context.Context)) tea.Cmd {
	m.flowBusy = true
	base := m.opts.BaseContext
	if base == nil {
		base = context.Background()
	}
	ctx, cancel := context.WithCancel(base)
	go func() {
		defer cancel()
		fn(ctx)
		m.events.send(context.Background(), flowDoneMsg{})
	}()
	return nil
}

// rebuildAgent asks Options.Rebuild (if any) to reconstruct the agent from
// current on-disk config, off the Update goroutine — MCP servers reconnect
// over real subprocesses, which must never block the UI.
func (m *Model) rebuildAgent() tea.Cmd {
	rebuild := m.opts.Rebuild
	if rebuild == nil {
		return nil
	}
	return func() tea.Msg {
		a, label, err := rebuild()
		if err != nil {
			return errMsg{err}
		}
		return agentRebuiltMsg{agent: a, modelLabel: label}
	}
}

// applyRebuild is rebuildAgent's synchronous twin: it runs Options.Rebuild
// directly instead of wrapping it in a tea.Cmd, and sends the resulting
// message itself. Only the onboarding wizard uses it, and only because it
// must: the wizard's flow goroutine sends agentRebuiltMsg before it returns,
// and runFlow only sends flowDoneMsg after that — since both land on the
// same FIFO channel from the same goroutine, agentRebuiltMsg is guaranteed
// to reach Update before flowBusy clears, so there is no window where the
// user could submit a chat turn against a still-nil Agent. Routing through
// the async rebuildAgent Cmd instead would reopen exactly that window.
func (m *Model) applyRebuild(ctx context.Context) {
	rebuild := m.opts.Rebuild
	if rebuild == nil {
		return
	}
	a, label, err := rebuild()
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	m.events.send(ctx, agentRebuiltMsg{agent: a, modelLabel: label})
}

// keybindRows are the non-slash keybindings /help lists after the command
// table — commandRegistry is the source of truth for the commands
// themselves, so that list and the "/" autocomplete can never drift apart.
var keybindRows = [][2]string{
	{"/ (typing)", "autocomplete commands as you type"},
	{"shift+tab", "cycle mode"},
	{"↑↓ enter esc", "navigate a menu"},
	{"esc", "cancel the running turn"},
	{"ctrl+c", "cancel, or exit when idle"},
}

func helpText() string {
	var b strings.Builder
	for _, c := range commandRegistry {
		fmt.Fprintf(&b, "  %s  %s\n",
			styleTool.Render(fmt.Sprintf("%-18s", c.Name)),
			styleDim.Render(c.Desc))
	}
	for _, r := range keybindRows {
		fmt.Fprintf(&b, "  %s  %s\n",
			styleTool.Render(fmt.Sprintf("%-18s", r[0])),
			styleDim.Render(r[1]))
	}
	return b.String()
}
