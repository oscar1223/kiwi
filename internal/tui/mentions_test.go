package tui

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExpandFileMentionsInlinesTheFile(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main"), 0o644)

	sent, attached, missing := expandFileMentions(dir, "what does @main.go do?")

	if len(attached) != 1 || attached[0] != "main.go" {
		t.Fatalf("attached = %v, want [main.go]", attached)
	}
	if len(missing) != 0 {
		t.Errorf("missing = %v, want none", missing)
	}
	if !strings.Contains(sent, "package main") {
		t.Errorf("the file contents were not inlined:\n%s", sent)
	}
	// The user's own words survive verbatim, mention and all.
	if !strings.HasPrefix(sent, "what does @main.go do?") {
		t.Errorf("the user's text was rewritten:\n%s", sent)
	}
}

func TestExpandFileMentionsReportsWhatItCouldNotRead(t *testing.T) {
	dir := t.TempDir()

	sent, attached, missing := expandFileMentions(dir, "look at @nope.go")

	if len(attached) != 0 {
		t.Errorf("attached = %v, want none", attached)
	}
	if len(missing) != 1 || missing[0] != "nope.go" {
		t.Fatalf("missing = %v, want [nope.go]", missing)
	}
	// A mention that resolves to nothing degrades into prose, not an error.
	if sent != "look at @nope.go" {
		t.Errorf("sent = %q, want the original text unchanged", sent)
	}
}

// A directory would either fail confusingly or read forever.
func TestExpandFileMentionsRefusesDirectories(t *testing.T) {
	dir := t.TempDir()
	os.Mkdir(filepath.Join(dir, "pkg"), 0o755)

	_, attached, missing := expandFileMentions(dir, "check @pkg")
	if len(attached) != 0 {
		t.Errorf("a directory was attached: %v", attached)
	}
	if len(missing) != 1 {
		t.Errorf("missing = %v, want the directory reported", missing)
	}
}

// The boundary before "@" is what keeps ordinary prose from being read as a
// file reference.
func TestExpandFileMentionsIgnoresEmailAddresses(t *testing.T) {
	dir := t.TempDir()

	sent, attached, missing := expandFileMentions(dir, "mail someone@example.com about it")
	if len(attached) != 0 || len(missing) != 0 {
		t.Errorf("an email address was treated as a mention: attached=%v missing=%v", attached, missing)
	}
	if sent != "mail someone@example.com about it" {
		t.Errorf("sent = %q, want unchanged", sent)
	}
}

func TestExpandFileMentionsAttachesEachFileOnce(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("body"), 0o644)

	sent, attached, _ := expandFileMentions(dir, "compare @a.txt with @a.txt")
	if len(attached) != 1 {
		t.Errorf("attached = %v, want one entry", attached)
	}
	if strings.Count(sent, "--- a.txt ---") != 1 {
		t.Errorf("the same file was inlined twice:\n%s", sent)
	}
}

func TestExpandFileMentionsHandlesSeveralFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.txt"), []byte("first"), 0o644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("second"), 0o644)

	sent, attached, _ := expandFileMentions(dir, "@a.txt and @b.txt")
	if len(attached) != 2 {
		t.Fatalf("attached = %v, want both files", attached)
	}
	for _, want := range []string{"first", "second"} {
		if !strings.Contains(sent, want) {
			t.Errorf("%q is missing from the expanded message:\n%s", want, sent)
		}
	}
}

func TestExpandFileMentionsTruncatesOversizedFiles(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "big.log"), []byte(strings.Repeat("x", maxMentionBytes+5000)), 0o644)

	sent, attached, _ := expandFileMentions(dir, "@big.log")
	if len(attached) != 1 {
		t.Fatalf("attached = %v, want the file", attached)
	}
	if len(sent) > maxMentionBytes+1000 {
		t.Errorf("the message is %d bytes: the cap did not apply", len(sent))
	}
	if !strings.Contains(sent, "truncated") {
		t.Error("the truncation was not announced to the model")
	}
}

func TestExpandFileMentionsAcceptsAbsolutePaths(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "abs.txt")
	os.WriteFile(path, []byte("absolute body"), 0o644)

	// A working directory deliberately unrelated to the file's own.
	sent, attached, _ := expandFileMentions(t.TempDir(), "read @"+path)
	if len(attached) != 1 {
		t.Fatalf("attached = %v, want the absolute path", attached)
	}
	if !strings.Contains(sent, "absolute body") {
		t.Errorf("the absolute path was not inlined:\n%s", sent)
	}
}

func TestExpandFileMentionsLeavesPlainTextAlone(t *testing.T) {
	sent, attached, missing := expandFileMentions(t.TempDir(), "no mentions here at all")
	if sent != "no mentions here at all" || attached != nil || missing != nil {
		t.Errorf("plain text was modified: %q %v %v", sent, attached, missing)
	}
}
