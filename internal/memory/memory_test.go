package memory

import (
	"strings"
	"testing"
)

// newStore points memory at a temp config dir so no test ever reads or writes
// the developer's own saved notes.
func newStore(t *testing.T, workDir string) *Store {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	return New(workDir)
}

func TestReadIsEmptyBeforeAnythingIsRemembered(t *testing.T) {
	s := newStore(t, t.TempDir())
	for _, scope := range []Scope{Global, Project} {
		got, err := s.Read(scope)
		if err != nil {
			t.Fatalf("Read(%s): %v", scope, err)
		}
		if got != "" {
			t.Errorf("Read(%s) = %q on a fresh install, want empty", scope, got)
		}
	}
}

func TestAppendBulletsAndAccumulates(t *testing.T) {
	s := newStore(t, t.TempDir())
	if _, err := s.Append(Project, "uses gofmt, not goimports"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Append(Project, "- ships a single binary"); err != nil {
		t.Fatal(err)
	}

	got, err := s.Read(Project)
	if err != nil {
		t.Fatal(err)
	}
	want := "- uses gofmt, not goimports\n- ships a single binary"
	if got != want {
		t.Errorf("Read(project) =\n%q\nwant\n%q", got, want)
	}
}

// A fact spanning several lines would be cut in half by capBody's
// oldest-line-first trimming, so it is flattened on the way in.
func TestAppendFlattensMultilineFacts(t *testing.T) {
	s := newStore(t, t.TempDir())
	if _, err := s.Append(Global, "prefers Spanish\nfor conversation"); err != nil {
		t.Fatal(err)
	}
	got, _ := s.Read(Global)
	if strings.Count(got, "\n") != 0 {
		t.Errorf("Read(global) = %q, want a single line", got)
	}
}

func TestScopesDoNotBleedIntoEachOther(t *testing.T) {
	s := newStore(t, t.TempDir())
	if _, err := s.Append(Global, "about the user"); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.Read(Project); got != "" {
		t.Errorf("a global note showed up in project memory: %q", got)
	}
}

// Two working directories must never share one file: a note about one
// codebase silently applying to another is the failure nobody would look for.
func TestProjectMemoryIsPerWorkingDirectory(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	a, b := New(t.TempDir()), New(t.TempDir())

	if _, err := a.Append(Project, "belongs to A"); err != nil {
		t.Fatal(err)
	}
	if got, _ := b.Read(Project); got != "" {
		t.Errorf("project B saw A's memory: %q", got)
	}

	pathA, _ := a.Path(Project)
	pathB, _ := b.Path(Project)
	if pathA == pathB {
		t.Errorf("both projects resolved to the same file: %s", pathA)
	}
}

// Sanitising a path alone is lossy — "/a/b" and "/a-b" collapse to the same
// slug — so the hash suffix is what actually keeps them apart.
func TestProjectKeySeparatesPathsThatSanitizeAlike(t *testing.T) {
	if projectKey("/a/b") == projectKey("/a-b") {
		t.Error("two different project paths produced the same memory file name")
	}
}

func TestProjectScopeNeedsAWorkingDirectory(t *testing.T) {
	s := newStore(t, "")
	if _, err := s.Path(Project); err == nil {
		t.Error("Path(project) with no working directory should report why, not guess")
	}
	if _, err := s.Append(Project, "anything"); err == nil {
		t.Error("Append(project) with no working directory should fail")
	}
}

func TestAppendDropsOldestNotesOverTheCap(t *testing.T) {
	s := newStore(t, t.TempDir())
	line := strings.Repeat("x", 200)
	for i := 0; i < 40; i++ { // 40 × ~200 chars is well past MaxChars
		if _, err := s.Append(Project, line); err != nil {
			t.Fatal(err)
		}
	}

	body, _ := s.Read(Project)
	if len(body) > MaxChars {
		t.Errorf("memory grew to %d chars, past the %d cap", len(body), MaxChars)
	}
	// A note big enough that it cannot fit in whatever slack the trimming
	// above left behind, so a drop is genuinely forced and reported.
	newest := strings.Repeat("y", 200)
	dropped, err := s.Append(Project, newest)
	if err != nil {
		t.Fatal(err)
	}
	if dropped == 0 {
		t.Error("appending over the cap dropped nothing and reported nothing")
	}
	body, _ = s.Read(Project)
	if !strings.HasSuffix(body, "- "+newest) {
		t.Error("the newest note was not kept; trimming must drop the oldest, not the newest")
	}
}

func TestClearForgetsOnlyItsOwnScope(t *testing.T) {
	s := newStore(t, t.TempDir())
	s.Append(Global, "keep me")
	s.Append(Project, "drop me")

	if err := s.Clear(Project); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.Read(Project); got != "" {
		t.Errorf("Read(project) = %q after Clear, want empty", got)
	}
	if got, _ := s.Read(Global); got != "- keep me" {
		t.Errorf("Clear(project) also wiped global memory: %q", got)
	}
}

// Clearing a scope that was never written must not error: a fresh install
// hitting "forget everything" is a no-op, not a failure.
func TestClearOnMissingFileIsFine(t *testing.T) {
	s := newStore(t, t.TempDir())
	if err := s.Clear(Global); err != nil {
		t.Errorf("Clear on a missing file: %v", err)
	}
}

func TestBlockIsEmptyUntilSomethingIsRemembered(t *testing.T) {
	s := newStore(t, t.TempDir())
	if got := s.Block(); got != "" {
		t.Errorf("Block() = %q on a fresh install, want empty so the prompt is untouched", got)
	}
}

func TestBlockLabelsBothScopes(t *testing.T) {
	s := newStore(t, t.TempDir())
	s.Append(Global, "writes in Spanish")
	s.Append(Project, "single Go binary")

	block := s.Block()
	for _, want := range []string{"## Memory", "writes in Spanish", "single Go binary", "all projects", "this project"} {
		if !strings.Contains(block, want) {
			t.Errorf("Block() is missing %q:\n%s", want, block)
		}
	}
}

func TestWriteEmptyBodyForgetsTheFile(t *testing.T) {
	s := newStore(t, t.TempDir())
	s.Append(Project, "temporary")
	if _, err := s.Write(Project, "   "); err != nil {
		t.Fatal(err)
	}
	if got, _ := s.Read(Project); got != "" {
		t.Errorf("Read(project) = %q, want empty after writing a blank body", got)
	}
}

func TestUnknownScopeIsRejected(t *testing.T) {
	s := newStore(t, t.TempDir())
	if Scope("session").Valid() {
		t.Error(`Scope("session") reported itself valid`)
	}
	if _, err := s.Path("session"); err == nil {
		t.Error("Path on an unknown scope should fail rather than invent a file")
	}
}
