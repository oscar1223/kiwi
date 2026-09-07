package tui

import (
	"os"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/permission"
)

func historyModel(t *testing.T, entries ...string) *Model {
	t.Helper()
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	m := runnableModel(t)
	m.promptHistory = loadHistory(t.TempDir())
	for _, e := range entries {
		m.promptHistory.record(e)
	}
	return m
}

func TestArrowsWalkBackAndForwardThroughHistory(t *testing.T) {
	m := historyModel(t, "first", "second", "third")

	m.onKey(key("up"))
	if got := m.input.Value(); got != "third" {
		t.Fatalf("first up = %q, want the most recent prompt", got)
	}
	m.onKey(key("up"))
	m.onKey(key("up"))
	if got := m.input.Value(); got != "first" {
		t.Errorf("three ups = %q, want the oldest prompt", got)
	}
	// Nowhere further back: the input must not clear itself.
	m.onKey(key("up"))
	if got := m.input.Value(); got != "first" {
		t.Errorf("up past the start = %q", got)
	}

	m.onKey(key("down"))
	if got := m.input.Value(); got != "second" {
		t.Errorf("down = %q, want to walk forward", got)
	}
}

// Stepping into the history must not eat what was being typed.
func TestGoingBackStashesTheDraftAndReturnsIt(t *testing.T) {
	m := historyModel(t, "an old prompt")
	m.setInput("half-written thought")

	m.onKey(key("up"))
	if got := m.input.Value(); got != "an old prompt" {
		t.Fatalf("up = %q", got)
	}
	m.onKey(key("down"))
	if got := m.input.Value(); got != "half-written thought" {
		t.Errorf("coming back = %q, want the draft returned", got)
	}
}

func TestHistorySkipsAnImmediateRepeat(t *testing.T) {
	m := historyModel(t, "same", "same", "different")
	if len(m.promptHistory.entries) != 2 {
		t.Errorf("entries = %v, want the repeat collapsed", m.promptHistory.entries)
	}
}

func TestHistoryIgnoresBlanksAndHugePastes(t *testing.T) {
	m := historyModel(t, "", "   ", strings.Repeat("x", maxHistoryEntryBytes+1))
	if len(m.promptHistory.entries) != 0 {
		t.Errorf("entries = %d, want nothing recorded", len(m.promptHistory.entries))
	}
}

func TestHistoryIsCappedAndPersistedPerDirectory(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	dir := t.TempDir()

	h := loadHistory(dir)
	for i := 0; i < maxHistory+20; i++ {
		h.record(strings.Repeat("p", 1) + string(rune('a'+i%26)) + sprintf("%d", i))
	}
	if len(h.entries) != maxHistory {
		t.Errorf("kept %d entries, want the cap of %d", len(h.entries), maxHistory)
	}

	// Reopened in the same directory, the list comes back.
	again := loadHistory(dir)
	if len(again.entries) != maxHistory {
		t.Fatalf("reloaded %d entries", len(again.entries))
	}
	if again.entries[len(again.entries)-1] != h.entries[len(h.entries)-1] {
		t.Error("the most recent prompt did not survive a reload")
	}

	// A different project does not see it. The prompts for one project are
	// noise in another.
	other := loadHistory(t.TempDir())
	if len(other.entries) != 0 {
		t.Errorf("another directory saw %d entries", len(other.entries))
	}
}

// A multi-line prompt has to survive a line-based file.
func TestMultiLinePromptsRoundTrip(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", t.TempDir())
	dir := t.TempDir()

	h := loadHistory(dir)
	original := "first line\nsecond line\\with a backslash"
	h.record(original)

	if got := loadHistory(dir).entries[0]; got != original {
		t.Errorf("round trip = %q, want %q", got, original)
	}
}

func TestReverseSearchFindsAndCanBeCancelled(t *testing.T) {
	m := historyModel(t, "run the tests", "fix the parser", "write the docs")
	m.setInput("draft")

	m.onKey(key("ctrl+r"))
	if m.promptHistory.search == nil {
		t.Fatal("ctrl+r did not open the search")
	}
	for _, r := range "pars" {
		m.onKey(key(string(r)))
	}
	if got := m.input.Value(); got != "fix the parser" {
		t.Errorf("search for 'pars' = %q", got)
	}
	if !strings.Contains(plain(m.searchLine()), "pars") {
		t.Errorf("the search prompt does not show the query: %q", plain(m.searchLine()))
	}

	m.onKey(key("esc"))
	if m.promptHistory.search != nil {
		t.Error("esc did not close the search")
	}
	if got := m.input.Value(); got != "draft" {
		t.Errorf("cancelling = %q, want the draft back", got)
	}
}

func TestReverseSearchKeepsTheMatchOnEnter(t *testing.T) {
	m := historyModel(t, "run the tests")
	m.onKey(key("ctrl+r"))
	for _, r := range "tests" {
		m.onKey(key(string(r)))
	}
	m.onKey(key("enter"))

	if m.promptHistory.search != nil {
		t.Error("enter did not close the search")
	}
	// Kept for editing rather than sent: a recalled prompt is usually
	// recalled to be changed.
	if got := m.input.Value(); got != "run the tests" {
		t.Errorf("input after accepting = %q", got)
	}
	if m.busy {
		t.Error("accepting a search result sent it")
	}
}

func TestReverseSearchSaysWhenNothingMatches(t *testing.T) {
	m := historyModel(t, "run the tests")
	m.onKey(key("ctrl+r"))
	for _, r := range "zzz" {
		m.onKey(key(string(r)))
	}
	if !strings.Contains(plain(m.searchLine()), "no match") {
		t.Errorf("search line = %q, want it to say there is no match", plain(m.searchLine()))
	}
}

func TestReverseSearchOnAnEmptyHistorySaysSo(t *testing.T) {
	m := historyModel(t)
	m.onKey(key("ctrl+r"))
	if m.promptHistory.search != nil {
		t.Error("search opened with nothing to search")
	}
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "no history") {
		t.Errorf("nothing was said about the empty history: %q", got)
	}
}

// A multi-line draft navigates its own text; the history only gets the arrow
// when the input cannot use it.
func TestAMultiLineDraftKeepsItsArrows(t *testing.T) {
	m := historyModel(t, "an old prompt")
	m.setInput("line one\nline two")
	m.input.MoveToBegin()

	if m.wantsHistory("down") {
		t.Error("the history claimed an arrow a multi-line draft was using")
	}
}

func TestHistoryIsNotRecalledWhileBusy(t *testing.T) {
	m := historyModel(t, "an old prompt")
	m.busy = true
	if m.wantsHistory("up") {
		t.Error("the history was reachable during a turn")
	}
}

func TestEditingARecalledPromptKeepsTheEdit(t *testing.T) {
	m := historyModel(t, "original")
	m.onKey(key("up"))
	m.onKey(key("!"))

	// Having edited it, the down arrow must not restore the stash over the
	// top of what was just typed.
	if got := m.input.Value(); !strings.HasSuffix(got, "!") {
		t.Fatalf("the edit was lost: %q", got)
	}
	if m.promptHistory.pos != len(m.promptHistory.entries) {
		t.Error("editing did not leave history navigation")
	}
}

func TestHistorySurvivesAnUnwritableDataDir(t *testing.T) {
	t.Setenv("XDG_DATA_HOME", "/proc/definitely-not-writable")
	h := loadHistory(t.TempDir())
	h.record("still works in memory")
	if len(h.entries) != 1 {
		t.Error("an unwritable directory broke the in-memory history")
	}
}

func TestHistoryKeybindsAreDocumented(t *testing.T) {
	help := plain(helpText())
	for _, want := range []string{"ctrl+r", "recall an earlier prompt"} {
		if !strings.Contains(help, want) {
			t.Errorf("/help does not mention %q", want)
		}
	}
}

func TestLoadHistoryWithoutADataDir(t *testing.T) {
	// No XDG_DATA_HOME and no home: the history is in memory only, and
	// nothing panics.
	t.Setenv("XDG_DATA_HOME", "")
	t.Setenv("HOME", "")
	m, _ := newTestModel(t, permission.ModeAsk)
	_ = m
	h := loadHistory("")
	h.record("x")
	if len(h.entries) != 1 {
		t.Error("the in-memory history did not work without a data directory")
	}
	_ = os.Getenv("HOME")
}
