package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/permission"
)

func newFS(t *testing.T, mode permission.Mode, d permission.Decider) *FS {
	t.Helper()
	return &FS{WorkDir: t.TempDir(), Perms: permission.NewBroker(mode, d)}
}

func call(t *testing.T, tool interface {
	Run(context.Context, json.RawMessage) (string, error)
}, in map[string]any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	return tool.Run(context.Background(), raw)
}

func TestReadFileNumbersLines(t *testing.T) {
	fs := newFS(t, permission.ModeAsk, permission.AllowAll{})
	path := filepath.Join(fs.WorkDir, "a.txt")
	os.WriteFile(path, []byte("one\ntwo\nthree\n"), 0o644)

	out, err := call(t, ReadFile{fs}, map[string]any{"path": "a.txt"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	for _, want := range []string{"1\tone", "2\ttwo", "3\tthree"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q:\n%s", want, out)
		}
	}
}

func TestReadFileOffsetAndLimit(t *testing.T) {
	fs := newFS(t, permission.ModeAsk, permission.AllowAll{})
	os.WriteFile(filepath.Join(fs.WorkDir, "a.txt"), []byte("l1\nl2\nl3\nl4\nl5\n"), 0o644)

	out, err := call(t, ReadFile{fs}, map[string]any{"path": "a.txt", "offset": 2, "limit": 2})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if strings.Contains(out, "l1") || strings.Contains(out, "l4") {
		t.Errorf("window not respected:\n%s", out)
	}
	if !strings.Contains(out, "l2") || !strings.Contains(out, "l3") {
		t.Errorf("window incomplete:\n%s", out)
	}
	if !strings.Contains(out, "more lines") {
		t.Error("the model must be told the file continues past the window")
	}
}

func TestReadFileErrors(t *testing.T) {
	fs := newFS(t, permission.ModeAsk, permission.AllowAll{})

	if _, err := call(t, ReadFile{fs}, map[string]any{"path": "nope.txt"}); err == nil {
		t.Error("expected an error for a missing file")
	}
	if _, err := call(t, ReadFile{fs}, map[string]any{"path": "."}); err == nil {
		t.Error("expected an error when reading a directory")
	}
	os.WriteFile(filepath.Join(fs.WorkDir, "short.txt"), []byte("one\n"), 0o644)
	if _, err := call(t, ReadFile{fs}, map[string]any{"path": "short.txt", "offset": 99}); err == nil {
		t.Error("expected an error for an out-of-range offset")
	}
}

func TestReadFileEmptyIsExplicit(t *testing.T) {
	fs := newFS(t, permission.ModeAsk, permission.AllowAll{})
	os.WriteFile(filepath.Join(fs.WorkDir, "empty.txt"), nil, 0o644)

	out, err := call(t, ReadFile{fs}, map[string]any{"path": "empty.txt"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "empty") {
		t.Errorf("out = %q; an empty result reads as a broken tool", out)
	}
}

func TestWriteFileCreatesParentDirs(t *testing.T) {
	fs := newFS(t, permission.ModeAsk, permission.AllowAll{})
	_, err := call(t, WriteFile{fs}, map[string]any{
		"path": "a/b/c.txt", "content": "hi\n",
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(fs.WorkDir, "a/b/c.txt"))
	if err != nil || string(got) != "hi\n" {
		t.Errorf("file = %q, err = %v", got, err)
	}
}

func TestWriteFileDeniedLeavesDiskUntouched(t *testing.T) {
	fs := newFS(t, permission.ModeAsk, permission.DenyAll{})
	path := filepath.Join(fs.WorkDir, "keep.txt")
	os.WriteFile(path, []byte("original\n"), 0o644)

	if _, err := call(t, WriteFile{fs}, map[string]any{"path": "keep.txt", "content": "clobbered"}); err != permission.ErrDenied {
		t.Fatalf("err = %v, want ErrDenied", err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != "original\n" {
		t.Errorf("file was modified despite denial: %q", got)
	}
}

// The user must see what they are approving.
func TestWriteFileAsksWithADiff(t *testing.T) {
	var seen *permission.Request
	d := deciderFunc(func(_ context.Context, r *permission.Request) (bool, error) {
		seen = r
		return true, nil
	})
	fs := newFS(t, permission.ModeAsk, d)
	os.WriteFile(filepath.Join(fs.WorkDir, "x.txt"), []byte("before\n"), 0o644)

	if _, err := call(t, WriteFile{fs}, map[string]any{"path": "x.txt", "content": "after\n"}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if seen == nil {
		t.Fatal("no permission request was raised")
	}
	if !strings.Contains(seen.Diff, "-before") || !strings.Contains(seen.Diff, "+after") {
		t.Errorf("diff did not show the change:\n%s", seen.Diff)
	}
	if !strings.Contains(seen.Detail, "overwrite") {
		t.Errorf("detail = %q, should say the file is being overwritten", seen.Detail)
	}
}

func TestWriteFilePlanModeIsBlocked(t *testing.T) {
	fs := newFS(t, permission.ModePlan, permission.AllowAll{})
	if _, err := call(t, WriteFile{fs}, map[string]any{"path": "x.txt", "content": "x"}); err != permission.ErrDenied {
		t.Errorf("err = %v, want ErrDenied even with a permissive decider", err)
	}
	if _, err := os.Stat(filepath.Join(fs.WorkDir, "x.txt")); err == nil {
		t.Error("Plan mode wrote a file")
	}
}

func TestEditFileReplacesUniqueMatch(t *testing.T) {
	fs := newFS(t, permission.ModeWork, permission.DenyAll{}) // Work auto-approves edits
	path := filepath.Join(fs.WorkDir, "v.txt")
	os.WriteFile(path, []byte("version = 1.0\nother = x\n"), 0o644)

	out, err := call(t, EditFile{fs}, map[string]any{
		"path": "v.txt", "old_string": "version = 1.0", "new_string": "version = 1.1",
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != "version = 1.1\nother = x\n" {
		t.Errorf("file = %q", got)
	}
	if !strings.Contains(out, "Edited") {
		t.Errorf("out = %q", out)
	}
}

func TestEditFileAmbiguousMatchIsRefused(t *testing.T) {
	fs := newFS(t, permission.ModeWork, permission.AllowAll{})
	path := filepath.Join(fs.WorkDir, "dup.txt")
	os.WriteFile(path, []byte("x\nx\n"), 0o644)

	_, err := call(t, EditFile{fs}, map[string]any{
		"path": "dup.txt", "old_string": "x", "new_string": "y",
	})
	if err == nil {
		t.Fatal("expected an error for an ambiguous match")
	}
	if !strings.Contains(err.Error(), "appears 2 times") {
		t.Errorf("err = %v; it should tell the model how to fix this", err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != "x\nx\n" {
		t.Errorf("file was modified despite the ambiguity: %q", got)
	}
}

func TestEditFileReplaceAll(t *testing.T) {
	fs := newFS(t, permission.ModeWork, permission.AllowAll{})
	path := filepath.Join(fs.WorkDir, "dup.txt")
	os.WriteFile(path, []byte("x\nx\n"), 0o644)

	out, err := call(t, EditFile{fs}, map[string]any{
		"path": "dup.txt", "old_string": "x", "new_string": "y", "replace_all": true,
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	got, _ := os.ReadFile(path)
	if string(got) != "y\ny\n" {
		t.Errorf("file = %q", got)
	}
	if !strings.Contains(out, "2 replacements") {
		t.Errorf("out = %q", out)
	}
}

func TestEditFileMissingMatch(t *testing.T) {
	fs := newFS(t, permission.ModeWork, permission.AllowAll{})
	os.WriteFile(filepath.Join(fs.WorkDir, "a.txt"), []byte("hello\n"), 0o644)

	if _, err := call(t, EditFile{fs}, map[string]any{
		"path": "a.txt", "old_string": "goodbye", "new_string": "x",
	}); err == nil {
		t.Error("expected an error when old_string is absent")
	}
}

func TestEditFileIdenticalStringsRejected(t *testing.T) {
	fs := newFS(t, permission.ModeWork, permission.AllowAll{})
	os.WriteFile(filepath.Join(fs.WorkDir, "a.txt"), []byte("hello\n"), 0o644)

	if _, err := call(t, EditFile{fs}, map[string]any{
		"path": "a.txt", "old_string": "hello", "new_string": "hello",
	}); err == nil {
		t.Error("a no-op edit should be reported rather than silently succeeding")
	}
}

func TestPathResolution(t *testing.T) {
	fs := newFS(t, permission.ModeAsk, permission.AllowAll{})

	rel, err := fs.resolve("sub/file.txt")
	if err != nil {
		t.Fatal(err)
	}
	if rel != filepath.Join(fs.WorkDir, "sub/file.txt") {
		t.Errorf("relative path resolved to %q", rel)
	}

	abs, err := fs.resolve("/etc/hosts")
	if err != nil {
		t.Fatal(err)
	}
	if abs != "/etc/hosts" {
		t.Errorf("absolute path was rewritten to %q", abs)
	}

	home, err := fs.resolve("~/x")
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(home, "~") {
		t.Errorf("~ was not expanded: %q", home)
	}

	if _, err := fs.resolve(""); err == nil {
		t.Error("expected an error for an empty path")
	}
}

type deciderFunc func(context.Context, *permission.Request) (bool, error)

func (f deciderFunc) Decide(ctx context.Context, r *permission.Request) (bool, error) {
	return f(ctx, r)
}
