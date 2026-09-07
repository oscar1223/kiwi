package checkpoint

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// newProject creates a real git repository to snapshot, with the shadow repos
// redirected into the same temporary tree so nothing lands in the developer's
// own config directory.
func newProject(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not installed")
	}
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	dir := t.TempDir()
	run(t, dir, "git", "init", "--quiet", "--initial-branch=main")
	run(t, dir, "git", "config", "user.name", "test")
	run(t, dir, "git", "config", "user.email", "test@localhost")
	return dir
}

func run(t *testing.T, dir string, name string, args ...string) string {
	t.Helper()
	cmd := exec.Command(name, args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("%s %s: %v\n%s", name, strings.Join(args, " "), err, out)
	}
	return strings.TrimSpace(string(out))
}

func write(t *testing.T, dir, rel, body string) {
	t.Helper()
	path := filepath.Join(dir, rel)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func read(t *testing.T, dir, rel string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(dir, rel))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func newStore(t *testing.T, dir string) *Store {
	t.Helper()
	s, err := New(context.Background(), dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return s
}

func TestTakeThenRestoreBringsBackTheOldContents(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "main.go", "package main\n")
	s := newStore(t, dir)

	id, err := s.Take(context.Background(), "before the edit")
	if err != nil {
		t.Fatalf("Take: %v", err)
	}

	write(t, dir, "main.go", "package main // ruined\n")
	if err := s.Restore(context.Background(), id); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if got := read(t, dir, "main.go"); got != "package main\n" {
		t.Errorf("main.go = %q, want the pre-edit contents", got)
	}
}

func TestRestoreRemovesFilesCreatedAfterTheCheckpoint(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "main.go", "package main\n")
	s := newStore(t, dir)

	id, err := s.Take(context.Background(), "before")
	if err != nil {
		t.Fatal(err)
	}
	write(t, dir, "internal/new/new.go", "package new\n")

	if err := s.Restore(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "internal/new/new.go")); !os.IsNotExist(err) {
		t.Error("a file created after the checkpoint survived the restore")
	}
}

func TestRestoreBringsBackAFileDeletedAfterTheCheckpoint(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "keep.go", "package keep\n")
	s := newStore(t, dir)

	id, err := s.Take(context.Background(), "before")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "keep.go")); err != nil {
		t.Fatal(err)
	}

	if err := s.Restore(context.Background(), id); err != nil {
		t.Fatal(err)
	}
	if got := read(t, dir, "keep.go"); got != "package keep\n" {
		t.Errorf("keep.go = %q, want it restored", got)
	}
}

// The promise the whole design rests on: Kiwi snapshots through a repository
// of its own, so the user's index, stash and history come out untouched.
func TestSnapshottingLeavesTheProjectRepositoryAlone(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "committed.go", "package a\n")
	run(t, dir, "git", "add", "committed.go")
	run(t, dir, "git", "commit", "--quiet", "-m", "first")

	write(t, dir, "staged.go", "package b\n")
	run(t, dir, "git", "add", "staged.go")
	write(t, dir, "dirty.go", "package c\n")

	stashesBefore := run(t, dir, "git", "stash", "list")
	logBefore := run(t, dir, "git", "log", "--format=%H %s")
	statusBefore := run(t, dir, "git", "status", "--porcelain")

	s := newStore(t, dir)
	if _, err := s.Take(context.Background(), "snapshot"); err != nil {
		t.Fatalf("Take: %v", err)
	}

	if got := run(t, dir, "git", "status", "--porcelain"); got != statusBefore {
		t.Errorf("project status changed:\nbefore %q\nafter  %q", statusBefore, got)
	}
	if got := run(t, dir, "git", "log", "--format=%H %s"); got != logBefore {
		t.Errorf("project history changed:\nbefore %q\nafter  %q", logBefore, got)
	}
	if got := run(t, dir, "git", "stash", "list"); got != stashesBefore {
		t.Errorf("project stash changed: %q", got)
	}
}

func TestSnapshotsSkipIgnoredFiles(t *testing.T) {
	dir := newProject(t)
	write(t, dir, ".gitignore", "node_modules/\n")
	write(t, dir, "node_modules/dep/index.js", "module.exports = {}\n")
	write(t, dir, "app.js", "// app\n")
	s := newStore(t, dir)

	id, err := s.Take(context.Background(), "snapshot")
	if err != nil {
		t.Fatal(err)
	}
	files, err := s.Files(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range files {
		if strings.HasPrefix(f, "node_modules/") {
			t.Fatalf("ignored path %q made it into the snapshot", f)
		}
	}
}

func TestDiffReportsWhatChangedSinceTheCheckpoint(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "main.go", "package main\n")
	s := newStore(t, dir)

	id, err := s.Take(context.Background(), "before")
	if err != nil {
		t.Fatal(err)
	}
	if diff, err := s.Diff(context.Background(), id); err != nil || diff != "" {
		t.Fatalf("Diff on an unchanged tree = %q, %v; want empty", diff, err)
	}

	write(t, dir, "main.go", "package main\n\nfunc main() {}\n")
	diff, err := s.Diff(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(diff, "func main() {}") {
		t.Errorf("diff does not mention the change:\n%s", diff)
	}

	files, err := s.Files(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	if len(files) != 1 || files[0] != "main.go" {
		t.Errorf("Files = %v, want [main.go]", files)
	}
}

func TestListReturnsNewestFirst(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "a.txt", "1\n")
	s := newStore(t, dir)

	if _, err := s.Take(context.Background(), "first"); err != nil {
		t.Fatal(err)
	}
	write(t, dir, "a.txt", "2\n")
	if _, err := s.Take(context.Background(), "second"); err != nil {
		t.Fatal(err)
	}

	list, err := s.List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(list) != 2 {
		t.Fatalf("List returned %d checkpoints, want 2", len(list))
	}
	if list[0].Label != "second" || list[1].Label != "first" {
		t.Errorf("List order = %q, %q; want newest first", list[0].Label, list[1].Label)
	}
	if list[0].Files != 1 {
		t.Errorf("Files = %d, want 1 changed path", list[0].Files)
	}
	if list[0].Short() == list[0].ID {
		t.Error("Short() did not abbreviate the id")
	}
}

func TestPruneReRootsAndStillLeavesTheLastCheckpointRestorable(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "a.txt", "0\n")
	s := newStore(t, dir)

	// One more than the cap, so the prune inside Take has to fire.
	for i := 0; i < 4; i++ {
		write(t, dir, "a.txt", string(rune('a'+i))+"\n")
		if _, err := s.Take(context.Background(), "step"); err != nil {
			t.Fatal(err)
		}
		if err := s.Prune(context.Background(), 3); err != nil {
			t.Fatalf("Prune: %v", err)
		}
	}

	list, err := s.List(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(list) > 3 {
		t.Errorf("history kept %d checkpoints, want at most 3", len(list))
	}
	// The point of pruning before taking: what is left is still usable.
	last := list[0].ID
	write(t, dir, "a.txt", "ruined\n")
	if err := s.Restore(context.Background(), last); err != nil {
		t.Fatalf("Restore after prune: %v", err)
	}
	if read(t, dir, "a.txt") == "ruined\n" {
		t.Error("restoring the surviving checkpoint did nothing")
	}
}

func TestNewRefusesADirectoryThatIsNotARepository(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	if _, err := New(context.Background(), t.TempDir()); !errors.Is(err, ErrNoRepo) {
		t.Errorf("New on a plain directory = %v, want ErrNoRepo", err)
	}
}

func TestRestoreRejectsAnUnknownCheckpoint(t *testing.T) {
	dir := newProject(t)
	s := newStore(t, dir)
	if err := s.Restore(context.Background(), "0123456789abcdef0123456789abcdef01234567"); !errors.Is(err, ErrNotFound) {
		t.Errorf("Restore of a stale id = %v, want ErrNotFound", err)
	}
	if err := s.Restore(context.Background(), ""); !errors.Is(err, ErrNotFound) {
		t.Errorf("Restore of an empty id = %v, want ErrNotFound", err)
	}
}

func TestHeadIsEmptyBeforeTheFirstCheckpoint(t *testing.T) {
	dir := newProject(t)
	s := newStore(t, dir)
	head, err := s.Head(context.Background())
	if err != nil {
		t.Fatalf("Head on an empty history: %v", err)
	}
	if head != "" {
		t.Errorf("Head = %q, want empty", head)
	}
	if list, err := s.List(context.Background()); err != nil || list != nil {
		t.Errorf("List on an empty history = %v, %v", list, err)
	}
}

func TestNewIsIdempotent(t *testing.T) {
	dir := newProject(t)
	write(t, dir, "a.txt", "1\n")
	s := newStore(t, dir)
	id, err := s.Take(context.Background(), "first")
	if err != nil {
		t.Fatal(err)
	}

	// A second session on the same directory must find the same history.
	again := newStore(t, dir)
	head, err := again.Head(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if head != id {
		t.Errorf("reopened store Head = %q, want %q", head, id)
	}
}

func TestOneLine(t *testing.T) {
	cases := map[string]string{
		"":                     "checkpoint",
		"   ":                  "checkpoint",
		"\n\n":                 "checkpoint",
		"add the thing":        "add the thing",
		"first line\nsecond":   "first line",
		"\n\nleading newlines": "leading newlines",
	}
	for in, want := range cases {
		if got := oneLine(in); got != want {
			t.Errorf("oneLine(%q) = %q, want %q", in, got, want)
		}
	}
	long := strings.Repeat("x", 200)
	if got := oneLine(long); len([]rune(got)) != 73 {
		t.Errorf("oneLine truncated to %d runes, want 72 plus an ellipsis", len([]rune(got)))
	}
}
