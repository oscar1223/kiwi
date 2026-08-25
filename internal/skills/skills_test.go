package skills

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func withSkillsDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	skillsDir, err := Dir()
	if err != nil {
		t.Fatal(err)
	}
	return skillsDir
}

func TestLoadMissingDirIsEmptyNotError(t *testing.T) {
	withSkillsDir(t)
	got, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("got %d skills, want 0", len(got))
	}
}

func TestSaveThenLoadRoundTrip(t *testing.T) {
	withSkillsDir(t)
	if _, err := Save("commit-style", "Use when writing a git commit message", "Follow conventional commits."); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	sk, ok := got["commit-style"]
	if !ok {
		t.Fatal("skill not found after Save")
	}
	if sk.Description != "Use when writing a git commit message" {
		t.Errorf("Description = %q", sk.Description)
	}
	if sk.Body != "Follow conventional commits." {
		t.Errorf("Body = %q", sk.Body)
	}
}

func TestDeleteRemovesTheFile(t *testing.T) {
	dir := withSkillsDir(t)
	Save("temp", "d", "b")

	if err := Delete("temp"); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "temp.md")); !os.IsNotExist(err) {
		t.Error("file still exists after Delete")
	}
}

func TestDeleteUnknownSkill(t *testing.T) {
	withSkillsDir(t)
	if err := Delete("nope"); !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

// A file without valid frontmatter must not crash the whole load — one bad
// file should never take down every other skill.
func TestLoadSkipsMalformedFiles(t *testing.T) {
	dir := withSkillsDir(t)
	os.MkdirAll(dir, 0o755)
	os.WriteFile(filepath.Join(dir, "broken.md"), []byte("not frontmatter at all"), 0o644)
	os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("---\nname: ignored\n---\nx"), 0o644) // wrong extension
	Save("good", "a real one", "body")

	got, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d skills, want 1 (only the well-formed one): %+v", len(got), got)
	}
	if _, ok := got["good"]; !ok {
		t.Error("the well-formed skill was not loaded")
	}
}

func TestParseFallsBackToFilenameWhenNameOmitted(t *testing.T) {
	sk, ok := parse("---\ndescription: no name given\n---\nbody text", "my-skill")
	if !ok {
		t.Fatal("parse rejected valid frontmatter")
	}
	if sk.Name != "my-skill" {
		t.Errorf("Name = %q, want the filename fallback", sk.Name)
	}
	if sk.Body != "body text" {
		t.Errorf("Body = %q", sk.Body)
	}
}

func TestParseRejectsMissingClosingFence(t *testing.T) {
	if _, ok := parse("---\nname: x\nno closing fence", "f"); ok {
		t.Error("parse accepted frontmatter with no closing ---")
	}
}

func TestSummaryEmpty(t *testing.T) {
	if got := Summary(map[string]Skill{}); got != "" {
		t.Errorf("Summary(empty) = %q, want empty", got)
	}
}

func TestSummaryListsSortedByName(t *testing.T) {
	got := Summary(map[string]Skill{
		"zebra": {Description: "z"},
		"alpha": {Description: "a"},
	})
	posA := indexOf(got, "alpha")
	posZ := indexOf(got, "zebra")
	if posA < 0 || posZ < 0 || posA > posZ {
		t.Errorf("Summary did not list alphabetically:\n%s", got)
	}
}

func indexOf(s, substr string) int {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
