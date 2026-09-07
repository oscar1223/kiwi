package tools

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// searchTree builds a small project to search through and returns its root.
func searchTree(t *testing.T) *FS {
	t.Helper()
	root := t.TempDir()
	write := func(rel, body string) {
		abs := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(abs, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("main.go", "package main\n\nfunc main() {}\n")
	write("internal/app/app.go", "package app\n\nfunc Start() error { return nil }\n")
	write("internal/app/app_test.go", "package app\n\nfunc TestStart(t *testing.T) {}\n")
	write("README.md", "# proyecto\n\nStart aquí.\n")
	// Both should be invisible to every search below.
	write("node_modules/dep/index.js", "function Start() {}\n")
	write(".git/config", "[core]\nStart = no\n")
	return &FS{WorkDir: root}
}

func runTool(t *testing.T, tool Tool, args map[string]any) string {
	t.Helper()
	input, err := json.Marshal(args)
	if err != nil {
		t.Fatal(err)
	}
	out, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatalf("%s: %v", tool.Name(), err)
	}
	return out
}

func TestGlobCrossesDirectoriesOnlyWithDoubleStar(t *testing.T) {
	fs := searchTree(t)

	deep := runTool(t, Glob{fs}, map[string]any{"pattern": "**/*.go"})
	for _, want := range []string{"main.go", "internal/app/app.go", "internal/app/app_test.go"} {
		if !strings.Contains(deep, want) {
			t.Errorf("**/*.go did not find %s:\n%s", want, deep)
		}
	}

	// A single star must not cross a separator, or the pattern language is
	// pointless.
	shallow := runTool(t, Glob{fs}, map[string]any{"pattern": "*.go"})
	if !strings.Contains(shallow, "main.go") {
		t.Errorf("*.go missed the file at the root:\n%s", shallow)
	}
	if strings.Contains(shallow, "internal/app/app.go") {
		t.Errorf("*.go crossed a directory separator:\n%s", shallow)
	}
}

func TestGlobSkipsHeavyDirectories(t *testing.T) {
	fs := searchTree(t)
	out := runTool(t, Glob{fs}, map[string]any{"pattern": "**/*"})

	for _, unwanted := range []string{"node_modules", ".git"} {
		if strings.Contains(out, unwanted) {
			t.Errorf("the walk descended into %s:\n%s", unwanted, out)
		}
	}
}

// Newest first is the ordering that makes the tool useful: what was touched
// recently is usually what the question is about.
func TestGlobReturnsNewestFirst(t *testing.T) {
	fs := searchTree(t)

	old := time.Now().Add(-48 * time.Hour)
	if err := os.Chtimes(filepath.Join(fs.WorkDir, "main.go"), old, old); err != nil {
		t.Fatal(err)
	}
	recent := time.Now()
	target := filepath.Join(fs.WorkDir, "internal/app/app.go")
	if err := os.Chtimes(target, recent, recent); err != nil {
		t.Fatal(err)
	}

	out := runTool(t, Glob{fs}, map[string]any{"pattern": "**/*.go"})
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 0 || !strings.Contains(lines[0], "app.go") {
		t.Errorf("the most recently modified file is not first:\n%s", out)
	}
	if i, j := indexOf(lines, "app.go"), indexOf(lines, "main.go"); i >= j {
		t.Errorf("ordering is not by modification time:\n%s", out)
	}
}

func indexOf(lines []string, needle string) int {
	for i, l := range lines {
		if strings.Contains(l, needle) {
			return i
		}
	}
	return -1
}

func TestGrepFindsMatchesWithLineNumbers(t *testing.T) {
	fs := searchTree(t)
	out := runTool(t, Grep{fs}, map[string]any{"pattern": `func Start`})

	if !strings.Contains(out, "internal/app/app.go") {
		t.Errorf("grep did not report the file:\n%s", out)
	}
	if !strings.Contains(out, "3:") {
		t.Errorf("grep did not report the line number:\n%s", out)
	}
	if strings.Contains(out, "node_modules") {
		t.Errorf("grep searched a skipped directory:\n%s", out)
	}
}

func TestGrepFiltersByGlob(t *testing.T) {
	fs := searchTree(t)

	all := runTool(t, Grep{fs}, map[string]any{"pattern": "Start"})
	if !strings.Contains(all, "README.md") {
		t.Fatalf("the unfiltered search missed the markdown file:\n%s", all)
	}

	onlyGo := runTool(t, Grep{fs}, map[string]any{"pattern": "Start", "glob": "**/*.go"})
	if strings.Contains(onlyGo, "README.md") {
		t.Errorf("the glob filter did not exclude the markdown file:\n%s", onlyGo)
	}
	if !strings.Contains(onlyGo, "app.go") {
		t.Errorf("the glob filter excluded a file it should have kept:\n%s", onlyGo)
	}
}

func TestGrepReportsNoMatchesPlainly(t *testing.T) {
	fs := searchTree(t)
	out := runTool(t, Grep{fs}, map[string]any{"pattern": "estoNoExisteEnNingunSitio"})

	if !strings.Contains(out, "no matches") {
		t.Errorf("a fruitless search should say so, not return nothing: %q", out)
	}
}

// A bad regexp is the model's mistake to fix, so it has to come back as an
// error it can read rather than an empty result it will misread as "nothing
// found".
func TestGrepRejectsABadPattern(t *testing.T) {
	fs := searchTree(t)
	input, _ := json.Marshal(map[string]any{"pattern": "func ("})

	if _, err := (Grep{fs}).Run(context.Background(), input); err == nil {
		t.Error("an unparseable regexp was accepted")
	}
}

func TestGrepSkipsBinaryFiles(t *testing.T) {
	fs := searchTree(t)
	body := append([]byte("Start\x00"), make([]byte, 100)...)
	if err := os.WriteFile(filepath.Join(fs.WorkDir, "blob.bin"), body, 0o644); err != nil {
		t.Fatal(err)
	}

	if out := runTool(t, Grep{fs}, map[string]any{"pattern": "Start"}); strings.Contains(out, "blob.bin") {
		t.Errorf("grep searched a binary file:\n%s", out)
	}
}

func TestListMarksDirectoriesAndSortsThemFirst(t *testing.T) {
	fs := searchTree(t)
	out := runTool(t, List{fs}, nil)

	if !strings.Contains(out, "internal/") {
		t.Errorf("directories are not marked with a trailing slash:\n%s", out)
	}
	lines := strings.Split(strings.TrimSpace(out), "\n")
	if len(lines) == 0 || !strings.HasSuffix(lines[0], "/") {
		t.Errorf("directories do not come first:\n%s", out)
	}
	if strings.Contains(out, "node_modules") {
		// ls shows one directory as it is, so node_modules *should* appear
		// here — it is only the recursive walks that skip it.
		t.Log("ls lists node_modules at the top level, which is correct")
	}
}

// ls with no arguments lists the working directory, so an absent or empty
// argument object must not be an error.
func TestListAcceptsNoArguments(t *testing.T) {
	fs := searchTree(t)
	for _, input := range []json.RawMessage{nil, json.RawMessage(`{}`)} {
		if _, err := (List{fs}).Run(context.Background(), input); err != nil {
			t.Errorf("ls(%s) failed: %v", string(input), err)
		}
	}
}

func TestListRejectsAFile(t *testing.T) {
	fs := searchTree(t)
	input, _ := json.Marshal(map[string]any{"path": "main.go"})

	if _, err := (List{fs}).Run(context.Background(), input); err == nil {
		t.Error("ls accepted a file where a directory was required")
	}
}

func TestGlobToRegexp(t *testing.T) {
	cases := []struct {
		pattern string
		path    string
		want    bool
	}{
		{"**/*.go", "main.go", true}, // ** also means "no directories"
		{"**/*.go", "a/b/c/main.go", true},
		{"**/*.go", "main.rs", false},
		{"*.go", "main.go", true},
		{"*.go", "a/main.go", false},
		{"internal/**/*.go", "internal/app/app.go", true},
		{"internal/**/*.go", "cmd/app/app.go", false},
		{"?.go", "a.go", true},
		{"?.go", "ab.go", false},
		{"a.go", "a.go", true},
		{"a.go", "aXgo", false}, // the dot is a literal, not "any character"
	}
	for _, c := range cases {
		re, err := globToRegexp(c.pattern)
		if err != nil {
			t.Fatalf("%q: %v", c.pattern, err)
		}
		if got := re.MatchString(c.path); got != c.want {
			t.Errorf("glob %q against %q = %v, want %v", c.pattern, c.path, got, c.want)
		}
	}
}
