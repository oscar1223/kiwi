package diagnostics

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

func project(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range files {
		path := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

// fakeChecker writes an executable that prints the given lines, and points
// EnvCommand at it. An inline command cannot be used: the override is split on
// spaces, so a message with spaces in it would not survive.
func fakeChecker(t *testing.T, dir string, lines ...string) string {
	t.Helper()
	path := filepath.Join(dir, "checker.sh")
	body := "#!/bin/sh\n"
	for _, l := range lines {
		body += "echo '" + l + "'\n"
	}
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseUnderstandsTheColonForm(t *testing.T) {
	out := `internal/app/app.go:12:3: undefined: Foo
internal/app/app.go:40: missing return
some progress line that is not a diagnostic
	indented continuation of the message above`

	got := parse(out)
	if len(got) != 2 {
		t.Fatalf("parse found %d problems, want 2: %+v", len(got), got)
	}
	if got[0].File != "internal/app/app.go" || got[0].Line != 12 || got[0].Msg != "undefined: Foo" {
		t.Errorf("first problem = %+v", got[0])
	}
	if got[1].Line != 40 || got[1].Msg != "missing return" {
		t.Errorf("second problem = %+v", got[1])
	}
}

func TestParseUnderstandsTypeScriptsParenForm(t *testing.T) {
	got := parse(`src/app.ts(31,7): error TS2339: Property 'nope' does not exist.`)
	if len(got) != 1 {
		t.Fatalf("parse found %d problems, want 1: %+v", len(got), got)
	}
	if got[0].File != "src/app.ts" || got[0].Line != 31 {
		t.Errorf("problem = %+v", got[0])
	}
	if !strings.Contains(got[0].Msg, "TS2339") {
		t.Errorf("the message was lost: %q", got[0].Msg)
	}
}

// go vet prefixes a build failure with its own name. Read as part of the path,
// the problem gets attributed to a file that does not exist — which looks
// exactly like no problem at all.
func TestParseStripsTheToolsOwnPrefix(t *testing.T) {
	got := parse("vet: ./main.go:4:2: undefined: undefinedFunction")
	if len(got) != 1 {
		t.Fatalf("parse found %d problems, want 1: %+v", len(got), got)
	}
	if got[0].File != "main.go" {
		t.Errorf("file = %q, want the tool prefix stripped", got[0].File)
	}
	if got[0].Line != 4 || got[0].Msg != "undefined: undefinedFunction" {
		t.Errorf("problem = %+v", got[0])
	}
}

func TestParseIgnoresLinesThatAreNotDiagnostics(t *testing.T) {
	for _, line := range []string{"", "   ", "Compiling...", "ok  \tpkg\t0.1s"} {
		if got := parse(line); len(got) != 0 {
			t.Errorf("parse(%q) = %+v, want nothing", line, got)
		}
	}
}

func TestCheckRunsTheOverrideAndReportsWhatItFinds(t *testing.T) {
	dir := project(t, map[string]string{"main.go": "package main\n"})
	target := filepath.Join(dir, "main.go")

	// A stand-in for the project's checker, so the test does not depend on
	// which toolchains happen to be installed.
	t.Setenv(EnvCommand, fakeChecker(t, dir, "main.go:3:1: something is wrong"))
	r := New(dir)

	got := r.Check(context.Background(), target)
	if !strings.Contains(got, "something is wrong") {
		t.Fatalf("Check = %q, want the problem reported", got)
	}
	if !strings.Contains(got, "main.go:3") {
		t.Errorf("Check does not locate the problem: %q", got)
	}
}

// A file with pre-existing problems would otherwise repeat them on every
// single edit, which is how a diagnostic channel teaches people to ignore it.
func TestCheckStaysQuietWhenNothingChanged(t *testing.T) {
	dir := project(t, map[string]string{"main.go": "package main\n"})
	target := filepath.Join(dir, "main.go")
	t.Setenv(EnvCommand, fakeChecker(t, dir, "main.go:3:1: same old problem"))
	r := New(dir)

	if first := r.Check(context.Background(), target); first == "" {
		t.Fatal("the first check said nothing")
	}
	if second := r.Check(context.Background(), target); second != "" {
		t.Errorf("the same findings were reported twice: %q", second)
	}
}

// Silence is not permanent: once a problem goes away and comes back, it is
// news again.
func TestCheckReportsAProblemThatComesBack(t *testing.T) {
	dir := project(t, map[string]string{"main.go": "package main\n"})
	target := filepath.Join(dir, "main.go")
	r := New(dir)

	broken := fakeChecker(t, dir, "main.go:3:1: broken")
	r.override = []string{broken}
	if r.Check(context.Background(), target) == "" {
		t.Fatal("the first check said nothing")
	}

	r.override = []string{"true"}
	if got := r.Check(context.Background(), target); got != "" {
		t.Errorf("a clean check reported something: %q", got)
	}

	r.override = []string{broken}
	if got := r.Check(context.Background(), target); got == "" {
		t.Error("a problem that came back was swallowed")
	}
}

// The edit is responsible for the file it touched, not for whatever else the
// package was already carrying.
func TestCheckIgnoresProblemsInOtherFiles(t *testing.T) {
	dir := project(t, map[string]string{"main.go": "package main\n", "other.go": "package main\n"})
	t.Setenv(EnvCommand, fakeChecker(t, dir, "other.go:9:1: not your problem"))
	r := New(dir)

	if got := r.Check(context.Background(), filepath.Join(dir, "main.go")); got != "" {
		t.Errorf("Check = %q, want silence about another file", got)
	}
}

func TestCheckCapsHowMuchItReports(t *testing.T) {
	dir := project(t, map[string]string{"main.go": "package main\n"})
	var lines []string
	for i := 1; i <= maxReported+5; i++ {
		lines = append(lines, "main.go:"+strconv.Itoa(i)+":1: a problem")
	}
	t.Setenv(EnvCommand, fakeChecker(t, dir, lines...))
	r := New(dir)

	got := r.Check(context.Background(), filepath.Join(dir, "main.go"))
	if strings.Count(got, "main.go:") != maxReported {
		t.Errorf("reported %d problems, want %d", strings.Count(got, "main.go:"), maxReported)
	}
	if !strings.Contains(got, "and 5 more") {
		t.Errorf("the report does not say how many were left out: %q", got)
	}
}

func TestCheckIsSilentWithNoCheckerAndNoFiles(t *testing.T) {
	dir := project(t, map[string]string{"notes.txt": "hello\n"})
	r := New(dir)

	if got := r.Check(context.Background()); got != "" {
		t.Errorf("Check with no paths = %q", got)
	}
	// A plain text file in a directory with no project markers: nothing to run.
	if got := r.Check(context.Background(), filepath.Join(dir, "notes.txt")); got != "" {
		t.Errorf("Check on an unrecognised file = %q", got)
	}
	// A nil Runner is the feature switch, and must not panic.
	var off *Runner
	if got := off.Check(context.Background(), filepath.Join(dir, "notes.txt")); got != "" {
		t.Errorf("a nil Runner reported %q", got)
	}
}

// A checker that fails to start, or takes too long, must not turn a successful
// edit into a failed tool call.
func TestABrokenCheckerIsSilentRatherThanFatal(t *testing.T) {
	dir := project(t, map[string]string{"main.go": "package main\n"})
	target := filepath.Join(dir, "main.go")

	t.Setenv(EnvCommand, "definitely-not-a-real-binary-9f3a")
	if got := New(dir).Check(context.Background(), target); got != "" {
		t.Errorf("a missing checker reported %q", got)
	}

	t.Setenv(EnvCommand, "sleep 30")
	if got := func() string {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		return New(dir).Check(ctx, target)
	}(); got != "" {
		t.Errorf("an abandoned check reported %q", got)
	}
}

func TestOverridePlacesThePathWhereItIsAsked(t *testing.T) {
	dir := project(t, map[string]string{"main.go": "package main\n"})
	target := filepath.Join(dir, "main.go")

	t.Setenv(EnvCommand, "check --file={} --strict")
	argv, ok := New(dir).command(target)
	if !ok {
		t.Fatal("the override was not used")
	}
	want := []string{"check", "--file=" + target, "--strict"}
	if strings.Join(argv, " ") != strings.Join(want, " ") {
		t.Errorf("argv = %v, want %v", argv, want)
	}

	t.Setenv(EnvCommand, "check --strict")
	argv, _ = New(dir).command(target)
	if argv[len(argv)-1] != target {
		t.Errorf("a command with no placeholder did not get the path appended: %v", argv)
	}
}

// The check is scoped to the edited file's package. On a large repository that
// is the difference between finishing inside the timeout and never finishing.
func TestGoProjectsAreCheckedPerPackage(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go is not installed")
	}
	dir := project(t, map[string]string{
		"go.mod":                  "module example.com/x\n\ngo 1.22\n",
		"internal/app/app.go":     "package app\n",
		"internal/other/other.go": "package other\n",
	})

	argv, ok := New(dir).command(filepath.Join(dir, "internal/app/app.go"))
	if !ok {
		t.Fatal("a Go project with go.mod was not recognised")
	}
	if strings.Join(argv, " ") != "go vet ./internal/app" {
		t.Errorf("argv = %v, want the edited file's package only", argv)
	}
}

func TestGoVetFindsARealBreak(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go is not installed")
	}
	dir := project(t, map[string]string{
		"go.mod":  "module example.com/x\n\ngo 1.22\n",
		"main.go": "package main\n\nfunc main() {\n\tundefinedFunction()\n}\n",
	})

	got := New(dir).Check(context.Background(), filepath.Join(dir, "main.go"))
	if !strings.Contains(got, "undefinedFunction") {
		t.Errorf("Check = %q, want the build failure reported", got)
	}
}
