package main

import (
	"context"
	"testing"

	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/proc"
)

// This is the specific wiring bug that a package-level unit test cannot see:
// assembleAgent registers the task tool into the parent's own registry
// *after* deriving GeneralTools from it. If that ordering were ever reversed
// (or Subset ever changed to alias instead of copy), a subagent would gain
// the ability to recurse without any single package's own tests catching it
// — only integration wiring at this level would.
func TestAssembleAgentTaskToolNeverLeaksIntoGeneralTools(t *testing.T) {
	fake := &llmtest.Fake{}
	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	procs := proc.NewRegistry()
	t.Cleanup(procs.KillAll)

	a, _, mgr := assembleAgent(context.Background(), fake, t.TempDir(), permission.ModeWork, broker, procs)
	if mgr != nil {
		t.Cleanup(mgr.Close)
	}

	taskTool, ok := a.Tools.Get(agent.TaskName)
	if !ok {
		t.Fatal("assembleAgent did not register the task tool at all")
	}
	tt, ok := taskTool.(agent.TaskTool)
	if !ok {
		t.Fatalf("task tool is %T, want agent.TaskTool", taskTool)
	}

	for _, name := range []string{"explore", "general"} {
		toolset := tt.ExploreTools
		if name == "general" {
			toolset = tt.GeneralTools
		}
		if _, ok := toolset.Get(agent.TaskName); ok {
			t.Errorf("%s toolset includes %q; a subagent could recurse", name, agent.TaskName)
		}
	}

	// The parent agent itself must still have task available.
	if _, ok := a.Tools.Get(agent.TaskName); !ok {
		t.Error("the top-level agent lost the task tool")
	}

	// And the background-process tools must be present and share the same
	// registry the session's Close() will clean up.
	for _, name := range []string{"bash_background", "bash_output", "kill_shell"} {
		if _, ok := a.Tools.Get(name); !ok {
			t.Errorf("missing tool %q", name)
		}
	}
}

// The explore toolset must be genuinely read-only: it should not include
// write_file, edit_file, or the mutable bash — only read_file and the
// hard-restricted ReadOnlyBash (registered under the name "bash").
func TestAssembleAgentExploreToolsAreReadOnly(t *testing.T) {
	fake := &llmtest.Fake{}
	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	procs := proc.NewRegistry()
	t.Cleanup(procs.KillAll)

	a, _, mgr := assembleAgent(context.Background(), fake, t.TempDir(), permission.ModeWork, broker, procs)
	if mgr != nil {
		t.Cleanup(mgr.Close)
	}

	taskTool, ok := a.Tools.Get(agent.TaskName)
	if !ok {
		t.Fatal("task tool not registered")
	}
	tt, ok := taskTool.(agent.TaskTool)
	if !ok {
		t.Fatalf("task tool is %T, want agent.TaskTool", taskTool)
	}

	for _, forbidden := range []string{"write_file", "edit_file", "bash_background", "kill_shell", "task"} {
		if _, ok := tt.ExploreTools.Get(forbidden); ok {
			t.Errorf("explore toolset includes %q, which is not read-only", forbidden)
		}
	}
	if _, ok := tt.ExploreTools.Get("read_file"); !ok {
		t.Error("explore toolset is missing read_file")
	}
	if _, ok := tt.ExploreTools.Get("bash"); !ok {
		t.Error("explore toolset is missing the read-only bash")
	}
}
