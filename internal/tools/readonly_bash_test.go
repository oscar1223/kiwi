package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/permission"
)

func newReadOnlyBash(t *testing.T) ReadOnlyBash {
	t.Helper()
	return ReadOnlyBash{Bash: Bash{
		WorkDir: t.TempDir(),
		Perms:   permission.NewBroker(permission.ModeWork, permission.AllowAll{}),
	}}
}

func TestReadOnlyBashAllowsReadOnlyCommands(t *testing.T) {
	tool := newReadOnlyBash(t)
	raw, _ := json.Marshal(map[string]string{"command": "echo hi"})
	out, err := tool.Run(context.Background(), raw)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "hi" {
		t.Errorf("out = %q", out)
	}
}

// The whole point: even with a permissive broker that would approve
// anything, a mutating command must never reach the shell at all.
func TestReadOnlyBashRejectsMutatingCommandsEvenWithAnAllowAllBroker(t *testing.T) {
	tool := newReadOnlyBash(t)
	raw, _ := json.Marshal(map[string]string{"command": "rm -rf /tmp/should-not-run"})
	_, err := tool.Run(context.Background(), raw)
	if err == nil {
		t.Fatal("expected the mutating command to be rejected")
	}
	if !strings.Contains(err.Error(), "read-only") {
		t.Errorf("err = %v, should explain why", err)
	}
}

// The rejection happens before the permission broker is ever consulted —
// proven here with a decider that panics if it's invoked at all. A mutating
// command must not even reach the point of asking; it's refused outright.
func TestReadOnlyBashRejectsWithoutTouchingThePermissionBroker(t *testing.T) {
	tool := ReadOnlyBash{Bash: Bash{
		WorkDir: t.TempDir(),
		Perms:   permission.NewBroker(permission.ModeAsk, panicDecider{}),
	}}
	raw, _ := json.Marshal(map[string]string{"command": "rm -rf /tmp/x"})
	if _, err := tool.Run(context.Background(), raw); err == nil {
		t.Fatal("expected the mutating command to be rejected")
	}
}

type panicDecider struct{}

func (panicDecider) Decide(context.Context, *permission.Request) (bool, error) {
	panic("the permission broker must never be consulted for a rejected command")
}

func TestReadOnlyBashNameMatchesOrdinaryBash(t *testing.T) {
	if (ReadOnlyBash{}).Name() != "bash" {
		t.Errorf("Name() = %q, want %q so a subagent's prompt sees it as plain bash", (ReadOnlyBash{}).Name(), "bash")
	}
}

func TestReadOnlyBashSchemaInheritsFromBash(t *testing.T) {
	if (ReadOnlyBash{}).Schema() == nil {
		t.Error("Schema() should be inherited from the embedded Bash, not nil")
	}
}
