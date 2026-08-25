package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/proc"
)

func newBackgroundTools(t *testing.T, decider permission.Decider) (BackgroundBash, BackgroundOutput, KillShell) {
	t.Helper()
	broker := permission.NewBroker(permission.ModeAsk, decider)
	procs := proc.NewRegistry()
	t.Cleanup(procs.KillAll)
	dir := t.TempDir()
	return BackgroundBash{WorkDir: dir, Perms: broker, Procs: procs},
		BackgroundOutput{Procs: procs},
		KillShell{Procs: procs}
}

func runJSON(t *testing.T, tool interface {
	Run(context.Context, json.RawMessage) (string, error)
}, in map[string]any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	return tool.Run(context.Background(), raw)
}

func TestBackgroundBashReturnsAnIDImmediately(t *testing.T) {
	bg, _, kill := newBackgroundTools(t, permission.AllowAll{})
	start := time.Now()
	out, err := runJSON(t, bg, map[string]any{"command": "sleep 5"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 1*time.Second {
		t.Errorf("bash_background blocked for %s", elapsed)
	}
	if !strings.Contains(out, "Started as ") {
		t.Errorf("out = %q", out)
	}
	id := strings.TrimSuffix(strings.TrimPrefix(strings.Split(out, ".")[0], "Started as "), "")
	runJSON(t, kill, map[string]any{"id": id})
}

func TestBackgroundBashRequiresPermission(t *testing.T) {
	bg, _, _ := newBackgroundTools(t, permission.DenyAll{})
	_, err := runJSON(t, bg, map[string]any{"command": "echo hi"})
	if err != permission.ErrDenied {
		t.Errorf("err = %v, want ErrDenied", err)
	}
}

func TestBackgroundOutputAndKillRoundTrip(t *testing.T) {
	bg, out, kill := newBackgroundTools(t, permission.AllowAll{})

	started, err := runJSON(t, bg, map[string]any{"command": "echo ready; sleep 10"})
	if err != nil {
		t.Fatalf("bash_background: %v", err)
	}
	id := extractID(t, started)

	time.Sleep(150 * time.Millisecond)
	got, err := runJSON(t, out, map[string]any{"id": id})
	if err != nil {
		t.Fatalf("bash_output: %v", err)
	}
	if !strings.Contains(got, "ready") {
		t.Errorf("output = %q", got)
	}
	if !strings.Contains(got, "running") {
		t.Errorf("output = %q, want it to report running", got)
	}

	if _, err := runJSON(t, kill, map[string]any{"id": id}); err != nil {
		t.Fatalf("kill_shell: %v", err)
	}

	after, err := runJSON(t, out, map[string]any{"id": id})
	if err != nil {
		t.Fatalf("bash_output after kill: %v", err)
	}
	if !strings.Contains(after, "killed") {
		t.Errorf("output after kill = %q", after)
	}
}

func TestBackgroundOutputUnknownID(t *testing.T) {
	_, out, _ := newBackgroundTools(t, permission.AllowAll{})
	if _, err := runJSON(t, out, map[string]any{"id": "nope"}); err == nil {
		t.Error("expected an error for an unknown id")
	}
}

func extractID(t *testing.T, msg string) string {
	t.Helper()
	const prefix = "Started as "
	if !strings.HasPrefix(msg, prefix) {
		t.Fatalf("unexpected message: %q", msg)
	}
	rest := msg[len(prefix):]
	return strings.SplitN(rest, ".", 2)[0]
}
