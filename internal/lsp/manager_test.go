package lsp

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseEnvReadsTheOverride(t *testing.T) {
	got := parseEnv(".go=gopls; .ts,tsx = tsserver --stdio ;garbage;=nothing")
	if len(got) != 2 {
		t.Fatalf("parsed %d servers, want 2: %+v", len(got), got)
	}
	if got[0].Name != "gopls" || strings.Join(got[0].Extensions, ",") != ".go" {
		t.Errorf("first server = %+v", got[0])
	}
	// A bare extension is accepted as well as a dotted one: the leading dot is
	// punctuation, not information, and demanding it would be a papercut in a
	// setting people write by hand.
	if strings.Join(got[1].Extensions, ",") != ".ts,.tsx" {
		t.Errorf("extensions = %v, want both normalised with a leading dot", got[1].Extensions)
	}
	if strings.Join(got[1].Command, " ") != "tsserver --stdio" {
		t.Errorf("command = %v", got[1].Command)
	}
}

func TestParseEnvIgnoresNonsense(t *testing.T) {
	for _, spec := range []string{"", "   ", ";;;", "no-equals-sign", ".go=", "=gopls"} {
		if got := parseEnv(spec); len(got) != 0 {
			t.Errorf("parseEnv(%q) = %+v, want nothing", spec, got)
		}
	}
}

// A server being installed is not a reason to run it: gopls on the machine
// says nothing about whether this particular project is written in Go.
func TestDetectNeedsAProjectMarkerNotJustABinary(t *testing.T) {
	dir := t.TempDir()
	if got := detect(dir); len(got) != 0 {
		t.Errorf("detect on an empty directory = %+v, want nothing", got)
	}
}

func TestNewManagerIsNilWhenThereIsNothingToRun(t *testing.T) {
	// Nil is the feature switch: the tool is then never registered, so the
	// model is not offered a capability that can only fail.
	if m := NewManager(t.TempDir()); m != nil {
		t.Errorf("NewManager on a bare directory = %+v, want nil", m)
	}
	var off *Manager
	if got := off.Languages(); got != nil {
		t.Errorf("a nil manager reported languages: %v", got)
	}
	if _, err := off.For(context.Background(), "a.go"); err == nil {
		t.Error("a nil manager handed out a client")
	}
	off.Close()
}

func fakeManager(t *testing.T) (*Manager, string) {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	file := filepath.Join(dir, "main.go")
	if err := os.WriteFile(file, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(helperEnv, "1")
	t.Setenv(EnvServers, ".go="+self)

	m := NewManager(dir)
	if m == nil {
		t.Fatal("NewManager returned nil with an override pointing at a real binary")
	}
	t.Cleanup(m.Close)
	return m, file
}

func TestManagerStartsOnceAndReuses(t *testing.T) {
	m, file := fakeManager(t)

	first, err := m.For(context.Background(), file)
	if err != nil {
		t.Fatalf("For: %v", err)
	}
	second, err := m.For(context.Background(), file)
	if err != nil {
		t.Fatalf("For (second): %v", err)
	}
	// Starting a language server is expensive enough that doing it twice for
	// the same project would be a bug worth failing over.
	if first != second {
		t.Error("the second question started a second server")
	}
}

func TestManagerRefusesALanguageItDoesNotHandle(t *testing.T) {
	m, _ := fakeManager(t)
	_, err := m.For(context.Background(), "/tmp/thing.rb")
	if err == nil {
		t.Fatal("a file no server handles was accepted")
	}
	if !strings.Contains(err.Error(), ".rb") {
		t.Errorf("the error does not say which language: %v", err)
	}
}

// A broken install should cost one timeout per session, not one per question.
func TestManagerRemembersAServerThatWouldNotStart(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(EnvServers, ".go=/bin/echo")

	m := NewManager(dir)
	if m == nil {
		t.Skip("/bin/echo is not available to stand in for a broken server")
	}
	t.Cleanup(m.Close)

	first := errText(m.For(context.Background(), filepath.Join(dir, "main.go")))
	second := errText(m.For(context.Background(), filepath.Join(dir, "main.go")))
	if first == "" || second == "" {
		t.Fatal("a server that answers nothing was accepted")
	}
	if first != second {
		t.Errorf("the failure was not remembered:\nfirst  %s\nsecond %s", first, second)
	}
}

func errText(_ *Client, err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func TestLanguagesListsWhatCanBeAnswered(t *testing.T) {
	m, _ := fakeManager(t)
	if got := m.Languages(); len(got) != 1 || got[0] != ".go" {
		t.Errorf("Languages() = %v, want [.go]", got)
	}
}

func TestForAfterCloseDoesNotStartAServer(t *testing.T) {
	m, file := fakeManager(t)
	m.Close()
	if _, err := m.For(context.Background(), file); err == nil {
		t.Error("a closed manager started a server")
	}
}
