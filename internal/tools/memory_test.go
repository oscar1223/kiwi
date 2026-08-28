package tools

import (
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/memory"
	"github.com/oscar1223/kiwi/internal/permission"
)

func newRemember(t *testing.T, mode permission.Mode, d permission.Decider) Remember {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	return Remember{Store: memory.New(t.TempDir()), Perms: permission.NewBroker(mode, d)}
}

func TestRememberDefaultsToProjectScope(t *testing.T) {
	tool := newRemember(t, permission.ModeWork, permission.AllowAll{})

	if _, err := call(t, tool, map[string]any{"fact": "builds with go build ./..."}); err != nil {
		t.Fatalf("Run: %v", err)
	}

	project, _ := tool.Store.Read(memory.Project)
	if !strings.Contains(project, "builds with go build") {
		t.Errorf("the fact did not reach project memory: %q", project)
	}
	if global, _ := tool.Store.Read(memory.Global); global != "" {
		t.Errorf("an unscoped fact leaked into global memory: %q", global)
	}
}

func TestRememberHonoursGlobalScope(t *testing.T) {
	tool := newRemember(t, permission.ModeWork, permission.AllowAll{})

	if _, err := call(t, tool, map[string]any{"fact": "prefers Spanish", "scope": "global"}); err != nil {
		t.Fatalf("Run: %v", err)
	}
	global, _ := tool.Store.Read(memory.Global)
	if !strings.Contains(global, "prefers Spanish") {
		t.Errorf("the fact did not reach global memory: %q", global)
	}
}

// Plan mode is read-only, and "read-only except for the notes it keeps about
// you" is not a distinction a user should have to discover.
func TestRememberIsBlockedInPlanMode(t *testing.T) {
	tool := newRemember(t, permission.ModePlan, permission.NonInteractive{})

	if _, err := call(t, tool, map[string]any{"fact": "should never be saved"}); err == nil {
		t.Fatal("remember succeeded in plan mode; it must be refused like any other write")
	}
	if got, _ := tool.Store.Read(memory.Project); got != "" {
		t.Errorf("plan mode still wrote to disk: %q", got)
	}
}

func TestRememberRejectsUnknownScope(t *testing.T) {
	tool := newRemember(t, permission.ModeWork, permission.AllowAll{})

	if _, err := call(t, tool, map[string]any{"fact": "x", "scope": "session"}); err == nil {
		t.Error("an unknown scope should be reported, not silently written somewhere")
	}
}

func TestRememberRejectsAnEmptyFact(t *testing.T) {
	tool := newRemember(t, permission.ModeWork, permission.AllowAll{})

	if _, err := call(t, tool, map[string]any{"fact": "  "}); err == nil {
		t.Error("an empty fact should be rejected")
	}
}
