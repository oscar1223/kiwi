package tui

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/tools"
)

// --- ! shell ---

func TestShellRunsWithoutSpendingATurn(t *testing.T) {
	m := historyModel(t)
	m.setInput("!echo hello-from-the-shell")

	m.onKey(key("enter"))
	if m.busy {
		t.Error("a ! command started a model turn")
	}

	// The command runs off the UI goroutine and reports back.
	m.Update(shellResultMsg{command: "echo hello", output: "hello-from-the-shell\n"})
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "hello-from-the-shell") {
		t.Errorf("the output did not reach the transcript: %q", got)
	}
}

// Plan mode is a promise that nothing will change, and a promise the user can
// step around by typing "!" is not one worth making.
func TestShellRespectsPlanMode(t *testing.T) {
	m := historyModel(t)
	m.opts.Broker.SetMode(permission.ModePlan)

	m.setInput("!rm -rf /tmp/whatever")
	m.onKey(key("enter"))

	got := plain(strings.Join(m.transcript.render(120), "\n"))
	if !strings.Contains(got, "read-only") {
		t.Errorf("plan mode did not refuse a writing command: %q", got)
	}

	// A read-only command is still fine in plan mode.
	m.setInput("!ls")
	m.onKey(key("enter"))
	if strings.Count(plain(strings.Join(m.transcript.render(120), "\n")), "read-only") != 1 {
		t.Error("plan mode refused a read-only command")
	}
}

func TestBareBangExplainsItself(t *testing.T) {
	m := historyModel(t)
	m.setInput("!")
	m.onKey(key("enter"))
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "shell command") {
		t.Errorf("a bare ! said %q", got)
	}
}

func TestShellOutputIsCapped(t *testing.T) {
	var lines []string
	for i := 0; i < maxShellOutput+30; i++ {
		lines = append(lines, "OUTPUT")
	}
	got := renderShellResult(shellResultMsg{output: strings.Join(lines, "\n")})
	if n := strings.Count(got, "OUTPUT"); n != maxShellOutput {
		t.Errorf("printed %d lines, want the cap of %d", n, maxShellOutput)
	}
	if !strings.Contains(plain(got), "30 more lines") {
		t.Errorf("the report does not say how much was cut:\n%s", plain(got))
	}
}

func TestShellReportsFailureAndSilence(t *testing.T) {
	if got := plain(renderShellResult(shellResultMsg{output: ""})); !strings.Contains(got, "no output") {
		t.Errorf("a silent command said %q", got)
	}
	if got := plain(renderShellResult(shellResultMsg{err: os.ErrPermission})); !strings.Contains(got, "permission") {
		t.Errorf("a failing command did not report its error: %q", got)
	}
}

// --- queued messages ---

func TestTypingDuringATurnQueuesInsteadOfVanishing(t *testing.T) {
	m := historyModel(t)
	m.busy = true

	m.setInput("also check the tests")
	m.onKey(key("enter"))

	if len(m.queued) != 1 {
		t.Fatalf("queued %d messages, want 1", len(m.queued))
	}
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "queued") {
		t.Errorf("the queued message was not shown: %q", got)
	}
	if got := plain(strings.Join(m.renderQueued(80), "")); !strings.Contains(got, "1 message queued") {
		t.Errorf("the queue indicator = %q", got)
	}
}

func TestTheQueueGoesAsOneMessage(t *testing.T) {
	m := historyModel(t)
	m.busy = true
	for _, text := range []string{"first thought", "second thought"} {
		m.setInput(text)
		m.onKey(key("enter"))
	}

	// Three thoughts had while watching one piece of work are one piece of
	// feedback, not three turns.
	m.busy = false
	m.flushQueue()
	if len(m.queued) != 0 {
		t.Error("the queue was not drained")
	}
	if !m.busy {
		t.Fatal("flushing the queue did not start a turn")
	}
	if got := plain(strings.Join(m.transcript.render(120), "\n")); !strings.Contains(got, "second thought") {
		t.Errorf("the queued text was not sent: %q", got)
	}
}

func TestEscDiscardsTheQueue(t *testing.T) {
	m := historyModel(t)
	m.busy = true
	m.setInput("never mind")
	m.onKey(key("enter"))

	m.onKey(key("esc"))
	if len(m.queued) != 0 {
		t.Error("esc did not discard the queue")
	}
	// The turn is still running: esc emptied the queue rather than cancelling.
	if !m.busy {
		t.Error("esc cancelled the turn instead of clearing the queue")
	}
}

func TestFlushIsANoOpWithNothingQueued(t *testing.T) {
	m := historyModel(t)
	if cmd := m.flushQueue(); cmd != nil {
		t.Error("flushing an empty queue started something")
	}
	if got := m.renderQueued(80); got != nil {
		t.Errorf("an empty queue drew %v", got)
	}
}

// --- ctrl+t task list ---

func TestCtrlTShowsTheModelsOwnList(t *testing.T) {
	m := historyModel(t)
	todos := &tools.TodoList{}
	m.opts.Todos = todos

	if _, handled := m.command("/help"); !handled {
		t.Fatal("sanity check failed")
	}
	// Empty: nothing to toggle, and it says so rather than opening a blank
	// panel.
	m.onKey(key("ctrl+t"))
	if m.showTodos {
		t.Error("an empty list was opened")
	}
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "empty") {
		t.Errorf("nothing was said about the empty list: %q", got)
	}
}

func TestRenderTodosDrawsEachStatus(t *testing.T) {
	m := historyModel(t)
	m.opts.Todos = todosWith(t,
		tools.TodoItem{Content: "done thing", Status: "done"},
		tools.TodoItem{Content: "current thing", Status: "in_progress"},
		tools.TodoItem{Content: "later thing", Status: "pending"},
	)
	m.showTodos = true

	got := plain(strings.Join(m.renderTodos(80), "\n"))
	for _, want := range []string{"✓ done thing", "▸ current thing", "○ later thing"} {
		if !strings.Contains(got, want) {
			t.Errorf("the list is missing %q:\n%s", want, got)
		}
	}

	m.showTodos = false
	if got := m.renderTodos(80); got != nil {
		t.Error("the list drew itself while hidden")
	}
}

func TestTodosAreAbsentWithoutAList(t *testing.T) {
	m := historyModel(t)
	m.opts.Todos = nil
	m.onKey(key("ctrl+t"))
	if m.renderTodos(80) != nil {
		t.Error("a session with no task list drew one")
	}
}

// --- ? panel and $EDITOR ---

func TestQuestionMarkTogglesTheShortcutPanel(t *testing.T) {
	m := historyModel(t)

	m.onKey(key("?"))
	if !m.showShortcuts {
		t.Fatal("? did not open the panel")
	}
	if got := plain(strings.Join(m.renderShortcuts(80), "\n")); !strings.Contains(got, "ctrl+r") {
		t.Errorf("the panel does not list the shortcuts:\n%s", got)
	}
	m.onKey(key("?"))
	if m.showShortcuts {
		t.Error("? did not close the panel")
	}
}

// "?" is an ordinary character the rest of the time; stealing it would make
// the input unusable for questions.
func TestQuestionMarkStillTypesIntoANonEmptyPrompt(t *testing.T) {
	m := historyModel(t)
	m.setInput("what about this")
	m.onKey(key("?"))

	if m.showShortcuts {
		t.Error("? opened the panel instead of typing")
	}
	if !strings.HasSuffix(m.input.Value(), "?") {
		t.Errorf("? was not typed: %q", m.input.Value())
	}
}

func TestComposeInEditorNeedsAnEditor(t *testing.T) {
	m := historyModel(t)
	t.Setenv("VISUAL", "")
	t.Setenv("EDITOR", "")

	m.onKey(key("ctrl+e"))
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "$EDITOR") {
		t.Errorf("nothing explained the missing editor: %q", got)
	}
}

func TestEditorCommandCarriesFlagsAndTheDraft(t *testing.T) {
	editorTempPath = filepath.Join(t.TempDir(), "draft.md")
	cmd := editorCommand("code --wait", "my draft")

	if cmd.Args[0] != "code" || cmd.Args[1] != "--wait" {
		t.Errorf("args = %v, want the editor's own flags kept", cmd.Args)
	}
	body, err := os.ReadFile(editorTempPath)
	if err != nil || string(body) != "my draft" {
		t.Errorf("the draft was not handed to the editor: %q, %v", body, err)
	}
	editorTempPath = ""
}

func TestEditorResultReplacesTheInput(t *testing.T) {
	m := historyModel(t)
	m.Update(editorRequestMsg{text: "written elsewhere\n"})
	if got := m.input.Value(); got != "written elsewhere" {
		t.Errorf("input = %q, want the editor's text with the trailing newline trimmed", got)
	}

	m.Update(editorRequestMsg{err: os.ErrNotExist})
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "editor:") {
		t.Errorf("an editor failure was not reported: %q", got)
	}
}

func TestFirstNonEmpty(t *testing.T) {
	if got := firstNonEmpty("", "  ", "vim", "nano"); got != "vim" {
		t.Errorf("firstNonEmpty = %q", got)
	}
	if got := firstNonEmpty("", "   "); got != "" {
		t.Errorf("firstNonEmpty with nothing set = %q", got)
	}
}

func TestQuoteAppleScriptEscapes(t *testing.T) {
	// The directory name reaches this from the filesystem, so it cannot be
	// pasted into a script raw.
	if got := quoteAppleScript(`say "hi" \ there`); got != `"say \"hi\" \\ there"` {
		t.Errorf("quoteAppleScript = %s", got)
	}
}

func TestNotifyIsSilentForAShortTurnOrWhenTurnedOff(t *testing.T) {
	m := historyModel(t)

	// Still watching: a terminal that beeps at every quick answer gets muted.
	m.began = timeNow()
	if cmd := m.notifyDone(); cmd != nil {
		t.Error("a short turn rang the bell")
	}

	m.began = timeNow().Add(-notifyAfter * 2)
	t.Setenv(EnvNotify, "off")
	if cmd := m.notifyDone(); cmd != nil {
		t.Error("KIWI_NOTIFY=off did not silence the bell")
	}

	t.Setenv(EnvNotify, "")
	if cmd := m.notifyDone(); cmd == nil {
		t.Error("a long turn did not ring the bell")
	}
}

func TestNewKeybindsAreDocumented(t *testing.T) {
	help := plain(helpText())
	for _, want := range []string{"ctrl+t", "!command", "ctrl+e", "@"} {
		if !strings.Contains(help, want) {
			t.Errorf("/help does not mention %q", want)
		}
	}
}

// todosWith builds a checklist in a given state, going through the tool so the
// test uses the same path the model does.
func todosWith(t *testing.T, items ...tools.TodoItem) *tools.TodoList {
	t.Helper()
	list := &tools.TodoList{}
	payload, err := json.Marshal(map[string]any{"todos": items})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := (tools.TodoWrite{List: list}).Run(context.Background(), payload); err != nil {
		t.Fatal(err)
	}
	return list
}

func timeNow() time.Time { return time.Now() }
