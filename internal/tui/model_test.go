package tui

import (
	"context"
	"strings"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/prompt"
	"github.com/oscar1223/kiwi/internal/tools"
)

func newTestModel(t *testing.T, mode permission.Mode) (*Model, *permission.Broker) {
	t.Helper()
	ev := NewEvents()
	broker := permission.NewBroker(mode, ev)
	m := New(Options{
		Agent: &agent.Agent{
			Tools:  tools.NewRegistry(),
			System: "test",
		},
		Broker:        broker,
		WorkDir:       t.TempDir(),
		ModelLabel:    "test/model",
		PromptOptions: prompt.Options{WorkingDir: "/tmp"},
		Events:        ev,
	})
	m.width = 80
	return m, broker
}

// key builds the message a terminal actually sends, so the tests exercise the
// same encoding the running program sees rather than a convenient fiction.
func key(name string) tea.KeyPressMsg {
	switch name {
	case "esc":
		return tea.KeyPressMsg{Code: tea.KeyEscape}
	case "enter":
		return tea.KeyPressMsg{Code: tea.KeyEnter}
	case "shift+tab":
		return tea.KeyPressMsg{Code: tea.KeyTab, Mod: tea.ModShift}
	case "ctrl+c":
		return tea.KeyPressMsg{Code: 'c', Mod: tea.ModCtrl}
	case "ctrl+d":
		return tea.KeyPressMsg{Code: 'd', Mod: tea.ModCtrl}
	case "up":
		return tea.KeyPressMsg{Code: tea.KeyUp}
	case "down":
		return tea.KeyPressMsg{Code: tea.KeyDown}
	case "tab":
		return tea.KeyPressMsg{Code: tea.KeyTab}
	case "space":
		return tea.KeyPressMsg{Code: tea.KeySpace, Text: " "}
	default:
		r := []rune(name)
		if len(r) != 1 {
			panic("key: unsupported key name " + name)
		}
		return tea.KeyPressMsg{Code: r[0], Text: name}
	}
}

// TestKeyHelperMatchesRealEncoding guards the helper itself: if these stop
// matching, every key test below would be asserting against a fiction.
func TestKeyHelperMatchesRealEncoding(t *testing.T) {
	for _, name := range []string{"esc", "enter", "shift+tab", "ctrl+c", "ctrl+d", "up", "down", "tab", "space", "y", "n", "a"} {
		if got := key(name).String(); got != name {
			t.Errorf("key(%q).String() = %q", name, got)
		}
	}
}

// update applies a message and returns the model, discarding the command.
func update(t *testing.T, m *Model, msg tea.Msg) *Model {
	t.Helper()
	next, _ := m.Update(msg)
	got, ok := next.(*Model)
	if !ok {
		t.Fatalf("Update returned %T, want *Model", next)
	}
	return got
}

// Streaming must flush completed lines and hold only the partial tail, which
// is what keeps finished output in the terminal's scrollback.
func TestStreamFlushesCompleteLinesOnly(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.busy = true

	m.stream("hello")
	if got := len(m.transcript.entries); got != 0 {
		t.Errorf("a partial line was filed already: %d entries", got)
	}
	if m.tail != "hello" {
		t.Errorf("tail = %q, want %q", m.tail, "hello")
	}

	m.stream(" world\nsecond")
	if got := len(m.transcript.entries); got != 1 {
		t.Errorf("completing a line filed %d entries, want 1", got)
	}
	if m.tail != "second" {
		t.Errorf("tail = %q, want the remainder %q", m.tail, "second")
	}

	// Two newlines in one delta must flush both lines and leave nothing.
	m.stream(" line\nthird\n")
	if m.tail != "" {
		t.Errorf("tail = %q, want empty", m.tail)
	}
	if got := len(m.transcript.entries); got != 3 {
		t.Errorf("filed %d lines in total, want 3", got)
	}
}

func TestFlushTailEmitsThePartialLine(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.tail = "trailing"

	m.flushTail()
	if got := len(m.transcript.entries); got != 1 {
		t.Fatalf("a non-empty tail must be flushed at the end of a turn: %d entries", got)
	}
	if m.tail != "" {
		t.Errorf("tail = %q after flush", m.tail)
	}
	m.flushTail()
	if got := len(m.transcript.entries); got != 1 {
		t.Errorf("flushing an empty tail was not a no-op: %d entries", got)
	}
}

// Events from a cancelled turn must not reach the screen.
func TestStaleEventsAreDropped(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.busy = true
	m.gen = 5

	m = update(t, m, textDeltaMsg{gen: 4, delta: "from an old turn"})
	if m.tail != "" {
		t.Errorf("a stale delta was rendered: tail = %q", m.tail)
	}

	m = update(t, m, turnDoneMsg{gen: 4, messages: []llm.Message{{Content: "old"}}})
	if len(m.history) != 0 {
		t.Errorf("a stale turn wrote %d messages into history", len(m.history))
	}
	if !m.busy {
		t.Error("a stale turn ended the current one")
	}

	// The current generation still works.
	m = update(t, m, textDeltaMsg{gen: 5, delta: "current"})
	if m.tail != "current" {
		t.Errorf("tail = %q, want the live delta", m.tail)
	}
}

// Cancelling must bump the generation so in-flight events are ignored, and
// release the context so tools stop.
func TestCancelTurnInvalidatesInFlightWork(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.busy = true
	m.gen = 1
	m.tail = "half a sentence"

	ctx, cancel := context.WithCancel(context.Background())
	m.cancel = cancel

	m.cancelTurn()

	if m.busy {
		t.Error("still busy after cancelling")
	}
	if m.gen != 2 {
		t.Errorf("gen = %d, want it bumped to 2", m.gen)
	}
	if m.tail != "" {
		t.Errorf("tail = %q, want it flushed", m.tail)
	}
	select {
	case <-ctx.Done():
	default:
		t.Error("the turn context was not cancelled; tools would keep running")
	}
}

func TestEscCancelsOnlyWhenBusy(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	m = update(t, m, key("esc"))
	if m.gen != 0 {
		t.Error("esc while idle should do nothing")
	}

	m.busy = true
	_, cancel := context.WithCancel(context.Background())
	m.cancel = cancel
	m = update(t, m, key("esc"))
	if m.busy {
		t.Error("esc while busy should cancel the turn")
	}
}

// A pending permission prompt owns the keyboard: the answer must reach the
// blocked tool, and nothing else may be typed meanwhile.
func TestPermissionPromptAnswersTheBlockedTool(t *testing.T) {
	for _, tc := range []struct {
		key  string
		want bool
	}{
		{"y", true}, {"Y", true}, {"enter", true},
		{"n", false}, {"N", false}, {"esc", false},
	} {
		m, broker := newTestModel(t, permission.ModeAsk)

		result := make(chan error, 1)
		go func() {
			result <- broker.Ask(context.Background(), permission.Action{
				Name:   permission.ActionBash,
				Detail: "echo hi",
			})
		}()

		// Drain the queued question the way the event loop would.
		var msg tea.Msg
		select {
		case msg = <-m.events.ch:
		case <-time.After(2 * time.Second):
			t.Fatalf("%s: no permission question arrived", tc.key)
		}
		pm, ok := msg.(permissionMsg)
		if !ok {
			t.Fatalf("%s: got %T, want permissionMsg", tc.key, msg)
		}
		m = update(t, m, pm)
		if m.pending == nil {
			t.Fatalf("%s: the model is not showing a prompt", tc.key)
		}

		m = update(t, m, key(tc.key))
		if m.pending != nil {
			t.Errorf("%s: the prompt was not dismissed", tc.key)
		}

		select {
		case err := <-result:
			allowed := err == nil
			if allowed != tc.want {
				t.Errorf("%s: allowed = %v, want %v (err %v)", tc.key, allowed, tc.want, err)
			}
		case <-time.After(2 * time.Second):
			t.Errorf("%s: the blocked tool was never released", tc.key)
		}
	}
}

func TestTypingIsIgnoredWhileAPromptIsPending(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.pending = &permission.Request{}

	m = update(t, m, key("a"))
	if m.input.Value() != "" {
		t.Errorf("input = %q; keystrokes must not reach the textarea while a prompt is up", m.input.Value())
	}
}

// Changing mode must move the broker *and* the system prompt: a model that is
// not told edits are blocked wastes turns on refused calls.
func TestModeChangeUpdatesBrokerAndSystemPrompt(t *testing.T) {
	m, broker := newTestModel(t, permission.ModeAsk)
	before := m.opts.Agent.System

	m = update(t, m, key("shift+tab"))

	if broker.Mode() != permission.ModePlan {
		t.Errorf("broker mode = %s, want plan", broker.Mode())
	}
	if m.opts.Agent.System == before {
		t.Error("the system prompt was not rebuilt for the new mode")
	}
	if !strings.Contains(m.opts.Agent.System, "Plan") {
		t.Error("the rebuilt prompt does not mention Plan mode")
	}
}

func TestSlashCommandsSwitchMode(t *testing.T) {
	m, broker := newTestModel(t, permission.ModeAsk)

	if _, handled := m.command("/work"); !handled {
		t.Fatal("/work was not handled")
	}
	if broker.Mode() != permission.ModeWork {
		t.Errorf("mode = %s, want work", broker.Mode())
	}

	if _, handled := m.command("/plan"); !handled {
		t.Fatal("/plan was not handled")
	}
	if broker.Mode() != permission.ModePlan {
		t.Errorf("mode = %s, want plan", broker.Mode())
	}
}

func TestSlashClearForgetsHistory(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.history = []llm.Message{{Role: llm.RoleUser, Content: "old"}}

	if _, handled := m.command("/clear"); !handled {
		t.Fatal("/clear was not handled")
	}
	if len(m.history) != 0 {
		t.Errorf("history still has %d messages", len(m.history))
	}
}

func TestNonCommandTextIsNotIntercepted(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	if _, handled := m.command("just a question"); handled {
		t.Error("ordinary input must not be treated as a command")
	}
}

func TestUnknownCommandIsReported(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	_, handled := m.command("/nope")
	if !handled {
		t.Fatal("an unknown slash command was sent to the model")
	}
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "/nope") {
		t.Errorf("the unknown command was not reported: %q", got)
	}
}

func TestRecordTracksCodeFences(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	m.record("prose")
	if m.inFence {
		t.Error("plain prose opened a fence")
	}
	m.record("```go")
	if !m.inFence {
		t.Error("a fence was not opened")
	}
	m.record("func main() {}")
	if !m.inFence {
		t.Error("the fence closed early")
	}
	m.record("```")
	if m.inFence {
		t.Error("the fence was not closed")
	}
}

// The fence is resolved once, when the line is recorded. Rendering must not
// walk it again, however many times the window is resized.
func TestRecordedFencesSurviveRerendering(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	for _, line := range []string{"prose", "```go", "func main() {}", "```", "more prose"} {
		m.record(line)
	}

	want := append([]string(nil), m.transcript.render(80)...)
	m.transcript.render(40)
	got := m.transcript.render(80)

	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d changed when the transcript was rerendered:\n %q\n %q",
				i, plain(want[i]), plain(got[i]))
		}
	}
}

// The first line of an answer gets the bullet; the rest are indented under it.
func TestOnlyTheFirstLineIsMarked(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	m.record("one")
	m.record("two")

	rows := m.transcript.render(80)
	if len(rows) != 2 {
		t.Fatalf("two recorded lines rendered as %d rows", len(rows))
	}
	if !strings.Contains(rows[0], "●") {
		t.Errorf("the first line has no marker: %q", plain(rows[0]))
	}
	if strings.Contains(rows[1], "●") {
		t.Errorf("a later line repeated the marker: %q", plain(rows[1]))
	}
}

func TestViewShowsModeAndModel(t *testing.T) {
	m, _ := newTestModel(t, permission.ModePlan)
	view := m.View().Content

	if !strings.Contains(view, "Plan") {
		t.Errorf("the status line does not show the mode:\n%s", view)
	}
	if !strings.Contains(view, "test/model") {
		t.Errorf("the status line does not show the model:\n%s", view)
	}
}

func TestViewShowsTheStreamingTail(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.busy = true
	m.began = time.Now()
	m.tail = "still writing this"

	view := m.View().Content
	if !strings.Contains(view, "still writing this") {
		t.Errorf("the partial line is not visible:\n%s", view)
	}
	if !strings.Contains(view, "esc to cancel") {
		t.Errorf("the cancel hint is missing while busy:\n%s", view)
	}
}

func TestEnterDoesNotSubmitWhileBusy(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.busy = true
	m.gen = 3
	m.input.SetValue("another question")

	m = update(t, m, key("enter"))
	if m.gen != 3 {
		t.Error("a second turn was started while one was running")
	}
}

func TestFormatTokenCount(t *testing.T) {
	cases := map[int]string{
		0:     "0",
		42:    "42",
		999:   "999",
		1000:  "1.0k",
		1234:  "1.2k",
		12400: "12.4k",
	}
	for in, want := range cases {
		if got := formatTokenCount(in); got != want {
			t.Errorf("formatTokenCount(%d) = %q, want %q", in, got, want)
		}
	}
}

func TestTurnDoneAccumulatesSessionUsage(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.gen = 1
	m.busy = true

	m = update(t, m, turnDoneMsg{
		gen:      1,
		messages: []llm.Message{{Role: llm.RoleAssistant, Content: "ok"}},
		usage:    llm.Usage{InputTokens: 100, OutputTokens: 20},
	})
	if m.sessionUsage.InputTokens != 100 || m.sessionUsage.OutputTokens != 20 {
		t.Errorf("sessionUsage = %+v", m.sessionUsage)
	}

	// A second turn accumulates, it does not replace.
	m.gen = 2
	m.busy = true
	m = update(t, m, turnDoneMsg{
		gen:      2,
		messages: []llm.Message{{Role: llm.RoleAssistant, Content: "ok again"}},
		usage:    llm.Usage{InputTokens: 50, OutputTokens: 10},
	})
	if m.sessionUsage.InputTokens != 150 || m.sessionUsage.OutputTokens != 30 {
		t.Errorf("sessionUsage after second turn = %+v", m.sessionUsage)
	}
}

func TestStatusLineOmitsTokensBeforeAnyUsage(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 80
	if strings.Contains(m.statusLine(), "tok") {
		t.Error("status line shows a token count before any turn has run")
	}
}

func TestStatusLineShowsAccumulatedTokens(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 80
	m.sessionUsage = llm.Usage{InputTokens: 900, OutputTokens: 100}
	if !strings.Contains(m.statusLine(), "1.0k tok") {
		t.Errorf("status line = %q, want it to show 1.0k tok", m.statusLine())
	}
}

func TestStatusLineShowsContextUsage(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 120

	// Roughly a tenth of the 128k-token fallback window, in characters.
	m.history = []llm.Message{{Role: llm.RoleUser, Content: strings.Repeat("x", 4*12_800)}}

	got := ansiRE.ReplaceAllString(m.statusLine(), "")
	if !strings.Contains(got, "ctx 10%") {
		t.Errorf("status line = %q, want it to report ctx 10%%", got)
	}
}

// The indicator is about the window filling up, so an empty conversation with
// a trivial system prompt must not clutter the line with "ctx 0%".
func TestStatusLineOmitsContextUsageWhenEmpty(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 120
	m.history = nil
	m.opts.Agent = nil

	if strings.Contains(m.statusLine(), "ctx") {
		t.Errorf("status line = %q, want no context indicator on an empty session", m.statusLine())
	}
}

// The system prompt is often the larger half of the window early on — project
// instructions, memory, skills and every tool schema live in it — so leaving
// it out would under-report exactly when the number first starts to matter.
func TestContextUsageCountsTheSystemPrompt(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.history = nil
	m.opts.Agent.System = strings.Repeat("s", 4000)

	used, window := m.contextUsage()
	if used < 900 {
		t.Errorf("contextUsage() = %d tokens, want the system prompt counted", used)
	}
	if window != llm.ContextWindow(m.opts.ModelLabel) {
		t.Errorf("window = %d, want the model's own %d", window, llm.ContextWindow(m.opts.ModelLabel))
	}
}

// The welcome banner is the whole onboarding for a user who already has a
// provider configured — it must explain modes, show example prompts, and
// point at "/" now that autocomplete exists to back it up.
func TestBannerShowsModesExamplesAndSlashHint(t *testing.T) {
	got := ansiRE.ReplaceAllString(banner("fake/fake-1", "/tmp/proj"), "")

	for _, want := range []string{
		"fake/fake-1", "/tmp/proj",
		"ask", "plan", "work", "shift+tab",
		"explain this repository",
		"type / to see all commands",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("banner() missing %q:\n%s", want, got)
		}
	}
}
