package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func pickerProject(t *testing.T, paths ...string) string {
	t.Helper()
	dir := t.TempDir()
	for _, p := range paths {
		full := filepath.Join(dir, p)
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestMentionQueryOnlyAppliesToARealMention(t *testing.T) {
	cases := map[string]struct {
		query string
		ok    bool
	}{
		"@":                    {"", true},
		"@int":                 {"int", true},
		"look at @internal/tu": {"internal/tu", true},
		// An email address in a sentence must not open a file picker.
		"write to someone@example.com": {"", false},
		// Once there is a space the mention is finished and this is prose.
		"@notes.txt and then": {"", false},
		"no mention here":     {"", false},
	}
	for input, want := range cases {
		query, ok := mentionQuery(input)
		if ok != want.ok || query != want.query {
			t.Errorf("mentionQuery(%q) = %q, %v; want %q, %v", input, query, ok, want.query, want.ok)
		}
	}
}

// People type "tumod" for internal/tui/model.go. A picker that only matches
// literal substrings sends them back to typing the whole path.
func TestSubsequenceMatching(t *testing.T) {
	if !subsequence("internal/tui/model.go", "tumod") {
		t.Error("a scattered match was rejected")
	}
	if !subsequence("anything", "") {
		t.Error("an empty query matched nothing")
	}
	if subsequence("internal/tui/model.go", "zzz") {
		t.Error("a non-match was accepted")
	}
	// Order matters: the characters have to appear in sequence.
	if subsequence("abc", "cba") {
		t.Error("an out-of-order match was accepted")
	}
}

func TestSuggestionsPreferTheObviousFile(t *testing.T) {
	dir := pickerProject(t,
		"model.go",
		"internal/tui/model.go",
		"internal/a/b/c/d/model.go",
		"unrelated.txt",
	)

	got := rankFiles(scanFiles(dir), "model.go")
	if len(got) < 3 {
		t.Fatalf("got %v, want the three model.go files", got)
	}
	// Shallowest first: the file someone means is almost never the one buried
	// six directories down.
	if got[0] != "model.go" {
		t.Errorf("first suggestion = %q, want the top-level file", got[0])
	}
	if got[len(got)-1] == "model.go" {
		t.Error("the deepest match sorted first")
	}
	for _, p := range got {
		if p == "unrelated.txt" {
			t.Error("a file that does not match was offered")
		}
	}
}

func TestScanSkipsTheHeavyDirectories(t *testing.T) {
	dir := pickerProject(t,
		"app.js",
		"node_modules/dep/index.js",
		".git/config",
		"vendor/thing/thing.go",
		".hidden/secret.txt",
	)

	for _, p := range scanFiles(dir) {
		for _, skipped := range []string{"node_modules", ".git", "vendor", ".hidden"} {
			if strings.HasPrefix(p, skipped) {
				t.Errorf("scan walked into %s: %q", skipped, p)
			}
		}
	}
}

func TestSuggestionsAreCapped(t *testing.T) {
	var paths []string
	for i := 0; i < maxFileSuggestions+10; i++ {
		paths = append(paths, sprintf("file%02d.go", i))
	}
	dir := pickerProject(t, paths...)

	if got := rankFiles(scanFiles(dir), "file"); len(got) != maxFileSuggestions {
		t.Errorf("offered %d suggestions, want the cap of %d", len(got), maxFileSuggestions)
	}
}

func TestTabCompletesAMention(t *testing.T) {
	dir := pickerProject(t, "internal/tui/model.go")
	m := historyModel(t)
	m.opts.WorkDir = dir

	m.setInput("look at @tumod")
	if got := m.fileSuggestionsFor(); len(got) != 1 {
		t.Fatalf("suggestions = %v, want the one match", got)
	}
	m.onKey(key("tab"))

	if got := m.input.Value(); got != "look at @internal/tui/model.go " {
		t.Errorf("input after tab = %q", got)
	}
	// The trailing space ends the mention, so the picker closes.
	if got := m.fileSuggestionsFor(); len(got) != 0 {
		t.Errorf("the picker stayed open after completing: %v", got)
	}
}

// Typing a path in full and pressing enter has to send the message, not swap
// in whichever suggestion happens to be highlighted.
func TestEnterSubmitsWhenTheMentionIsAlreadyComplete(t *testing.T) {
	dir := pickerProject(t, "notes.md", "notes-other.md")
	m := historyModel(t)
	m.opts.WorkDir = dir

	m.setInput("summarize @notes.md")
	if !m.mentionIsComplete() {
		t.Fatal("a path naming a real file was not seen as complete")
	}
	m.onKey(key("enter"))

	if !m.busy {
		t.Error("enter completed instead of submitting")
	}
	if got := plain(strings.Join(m.transcript.render(120), "\n")); !strings.Contains(got, "@notes.md") {
		t.Errorf("the mention was rewritten: %q", got)
	}
}

func TestEnterCompletesWhenTheMentionIsPartial(t *testing.T) {
	dir := pickerProject(t, "notes.md")
	m := historyModel(t)
	m.opts.WorkDir = dir

	m.setInput("summarize @not")
	m.onKey(key("enter"))

	if m.busy {
		t.Error("a partial mention was submitted instead of completed")
	}
	if got := m.input.Value(); !strings.Contains(got, "@notes.md") {
		t.Errorf("input = %q, want the mention completed", got)
	}
}

func TestArrowsMoveThroughTheSuggestions(t *testing.T) {
	dir := pickerProject(t, "a.go", "b.go")
	m := historyModel(t)
	m.opts.WorkDir = dir
	m.setInput("@.go")

	if len(m.fileSuggestionsFor()) != 2 {
		t.Fatalf("suggestions = %v", m.fileSuggestionsFor())
	}
	m.onKey(key("down"))
	if m.mentionIndex != 1 {
		t.Errorf("index = %d, want the second row", m.mentionIndex)
	}
	m.onKey(key("up"))
	if m.mentionIndex != 0 {
		t.Errorf("index = %d, want the first row", m.mentionIndex)
	}
	// Never past the ends.
	m.onKey(key("up"))
	if m.mentionIndex != 0 {
		t.Errorf("index went past the top: %d", m.mentionIndex)
	}
}

func TestMentionIsCompleteRejectsADirectory(t *testing.T) {
	dir := pickerProject(t, "pkg/file.go")
	m := historyModel(t)
	m.opts.WorkDir = dir

	m.setInput("@pkg")
	if m.mentionIsComplete() {
		t.Error("a directory was treated as a finished mention")
	}
}

func TestTheFileIndexIsCached(t *testing.T) {
	dir := pickerProject(t, "one.go")
	m := historyModel(t)
	m.opts.WorkDir = dir

	first := m.projectFiles()
	if len(first) != 1 {
		t.Fatalf("scanned %v", first)
	}
	// A file written behind the cache's back is not seen until it expires.
	if err := os.WriteFile(filepath.Join(dir, "two.go"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if len(m.projectFiles()) != 1 {
		t.Error("the cache was not used")
	}

	m.fileIndexAt = m.fileIndexAt.Add(-fileIndexTTL * 2)
	if len(m.projectFiles()) != 2 {
		t.Error("the cache did not expire")
	}
}

func TestRenderFileSuggestionsHighlightsTheSelection(t *testing.T) {
	got := plain(renderFileSuggestions([]string{"a.go", "b.go"}, 1, 80))
	if !strings.Contains(got, "▸ b.go") {
		t.Errorf("the selected row is not marked:\n%s", got)
	}
	// An out-of-range index falls back to the first row rather than panicking.
	if got := plain(renderFileSuggestions([]string{"a.go"}, 9, 80)); !strings.Contains(got, "▸ a.go") {
		t.Errorf("an out-of-range index broke the render:\n%s", got)
	}
}
