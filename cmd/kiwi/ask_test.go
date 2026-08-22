package main

import (
	"strings"
	"testing"
	"testing/iotest"
)

// blockingReader stands in for an inherited pipe that never closes — what CI,
// cron, Docker and any non-interactive parent hand to a child process. Reading
// it must never happen unless the user explicitly asked for stdin.
type blockingReader struct{ t *testing.T }

func (b blockingReader) Read([]byte) (int, error) {
	b.t.Fatal("stdin was read when the user did not ask for it; kiwi would hang here")
	return 0, nil
}

func TestReadPromptArgsOnlyNeverTouchesStdin(t *testing.T) {
	got, err := readPrompt(blockingReader{t}, []string{"what", "does", "this", "do?"})
	if err != nil {
		t.Fatalf("readPrompt: %v", err)
	}
	if got != "what does this do?" {
		t.Errorf("got %q", got)
	}
}

func TestReadPromptStdinWhenNoArgs(t *testing.T) {
	got, err := readPrompt(strings.NewReader("review this patch\n"), nil)
	if err != nil {
		t.Fatalf("readPrompt: %v", err)
	}
	if got != "review this patch" {
		t.Errorf("got %q", got)
	}
}

func TestReadPromptDashCombinesBoth(t *testing.T) {
	got, err := readPrompt(strings.NewReader("diff --git a/x b/x\n"), []string{"review this", "-"})
	if err != nil {
		t.Fatalf("readPrompt: %v", err)
	}
	want := "review this\n\ndiff --git a/x b/x"
	if got != want {
		t.Errorf("got %q, want %q", got, want)
	}
}

func TestReadPromptDashAloneReadsStdin(t *testing.T) {
	got, err := readPrompt(strings.NewReader("piped\n"), []string{"-"})
	if err != nil {
		t.Fatalf("readPrompt: %v", err)
	}
	if got != "piped" {
		t.Errorf("got %q", got)
	}
}

func TestReadPromptEmptyStdinIsNotAnError(t *testing.T) {
	got, err := readPrompt(strings.NewReader(""), nil)
	if err != nil {
		t.Fatalf("readPrompt: %v", err)
	}
	if got != "" {
		t.Errorf("got %q, want empty", got)
	}
}

func TestReadPromptSurfacesStdinErrors(t *testing.T) {
	_, err := readPrompt(iotest.ErrReader(errRead), nil)
	if err == nil {
		t.Fatal("expected the read error to surface")
	}
	if !strings.Contains(err.Error(), "reading stdin") {
		t.Errorf("err = %v", err)
	}
}

type readErr struct{}

func (readErr) Error() string { return "disk on fire" }

var errRead = readErr{}
