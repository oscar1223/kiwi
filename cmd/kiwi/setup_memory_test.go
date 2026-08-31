package main

import (
	"context"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/memory"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/proc"
	"github.com/oscar1223/kiwi/internal/tools"
)

func assembleForTest(t *testing.T, workDir string) *agent.Agent {
	t.Helper()
	fake := &llmtest.Fake{}
	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	procs := proc.NewRegistry()
	t.Cleanup(procs.KillAll)

	a, _, mgr := assembleAgent(context.Background(), fake, workDir, permission.ModeWork, broker, procs, nil)
	if mgr != nil {
		t.Cleanup(mgr.Close)
	}
	return a
}

// Saved notes are only worth anything if they reach the system prompt: the
// wiring between memory.Store and prompt.Options.Extra is exactly the seam no
// single package's tests can see.
func TestAssembleAgentPutsSavedNotesInTheSystemPrompt(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	workDir := t.TempDir()

	store := memory.New(workDir)
	if _, err := store.Append(memory.Global, "the user writes in Spanish"); err != nil {
		t.Fatal(err)
	}
	if _, err := store.Append(memory.Project, "this project ships one Go binary"); err != nil {
		t.Fatal(err)
	}

	a := assembleForTest(t, workDir)
	for _, want := range []string{"## Memory", "the user writes in Spanish", "this project ships one Go binary"} {
		if !strings.Contains(a.System, want) {
			t.Errorf("system prompt is missing %q", want)
		}
	}
}

// With nothing remembered the prompt must be exactly what it was before memory
// existed — an empty section header would cost tokens on every request and
// invite the model to fill it.
func TestAssembleAgentAddsNothingWhenNothingIsRemembered(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	a := assembleForTest(t, t.TempDir())
	if strings.Contains(a.System, "## Memory") {
		t.Errorf("an empty memory section reached the prompt:\n%s", a.System)
	}
}

// A subagent's context is thrown away when it returns, so nothing it decides
// belongs in a memory that every future session pays for.
func TestAssembleAgentKeepsRememberAwayFromSubagents(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	a := assembleForTest(t, t.TempDir())
	if _, ok := a.Tools.Get(tools.Remember{}.Name()); !ok {
		t.Fatal("the parent agent cannot remember anything")
	}

	taskTool, ok := a.Tools.Get(agent.TaskName)
	if !ok {
		t.Fatal("no task tool")
	}
	tt := taskTool.(agent.TaskTool)
	for name, toolset := range map[string]*tools.Registry{"explore": tt.ExploreTools, "general": tt.GeneralTools} {
		if _, ok := toolset.Get(tools.Remember{}.Name()); ok {
			t.Errorf("the %s subagent was handed the remember tool", name)
		}
	}
}
