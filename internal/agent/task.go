package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/tools"
)

// AgentType names a restricted toolset a subagent may use.
type AgentType string

const (
	// AgentExplore is read-only: it can read files and run read-only shell
	// commands, but cannot edit anything or run a command that mutates
	// anything. The right choice for "find X" or "understand how Y works"
	// delegations that should never need a permission prompt.
	AgentExplore AgentType = "explore"
	// AgentGeneral gets everything the parent has, minus the task tool
	// itself — see TaskTool's doc comment for why.
	AgentGeneral AgentType = "general"
)

// TaskName is the tool name the model calls to delegate work, and what
// Agent.Run checks for to run a batch of delegations concurrently.
const TaskName = "task"

// TaskTool lets the model delegate a self-contained piece of work to a
// subagent: a fresh, isolated conversation and a restricted toolset, with
// only the final summary returned to the parent — not the individual steps
// the subagent took. That isolation is the entire point: a subagent's own
// back-and-forth (the files it read, the commands it tried, the dead ends)
// never pollutes the parent's context window.
//
// Subagents cannot spawn further subagents: TaskTool is deliberately absent
// from both ExploreTools and GeneralTools, which rules out runaway recursive
// forking by construction — there is no depth counter to reach, because the
// capability to recurse was never handed out in the first place.
type TaskTool struct {
	Provider llm.Provider
	// System is the subagent's base system prompt — normally the parent's
	// own, built without mode instructions, since a subagent has no
	// permission mode of its own; it uses whatever its tools were built
	// with.
	System string
	// ExploreTools and GeneralTools are pre-built per agent_type: Explore
	// pairs read_file with the hard-restricted ReadOnlyBash, never the
	// parent's real bash; General is the parent's full toolset minus task.
	ExploreTools *tools.Registry
	GeneralTools *tools.Registry
	// MaxSteps bounds a subagent's own turn the same way it bounds the
	// parent's; zero means DefaultMaxSteps.
	MaxSteps int
}

func (TaskTool) Name() string { return TaskName }

func (TaskTool) Description() string {
	return "Delegate a self-contained piece of work to a subagent with its own, isolated " +
		"conversation. Only its final summary comes back to you, not the individual steps " +
		"it took — use this to keep your own context focused on a large search or a chunk " +
		"of independent work. Write the prompt so the subagent can act on it alone: it " +
		"starts with no context beyond what you give it. Use agent_type \"explore\" for " +
		"read-only investigation (finding code, understanding how something works) — it " +
		"cannot edit or run mutating commands, and so never needs to interrupt you for " +
		"permission. Use \"general\" when the task also needs to edit files or run commands."
}

func (TaskTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"description": map[string]any{
				"type":        "string",
				"description": "A short (3-6 word) label for this delegation, shown to the user.",
			},
			"prompt": map[string]any{
				"type":        "string",
				"description": "The full task for the subagent. It has no context beyond this — be specific and self-contained.",
			},
			"agent_type": map[string]any{
				"type":        "string",
				"enum":        []string{string(AgentExplore), string(AgentGeneral)},
				"description": "\"explore\" for read-only investigation, \"general\" for work that edits files or runs commands.",
			},
		},
		"required": []string{"description", "prompt", "agent_type"},
	}
}

func (t TaskTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Description string `json:"description"`
		Prompt      string `json:"prompt"`
		AgentType   string `json:"agent_type"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return "", fmt.Errorf("prompt is required")
	}

	var toolset *tools.Registry
	switch AgentType(in.AgentType) {
	case AgentExplore:
		toolset = t.ExploreTools
	case AgentGeneral, "":
		toolset = t.GeneralTools
	default:
		return "", fmt.Errorf("unknown agent_type %q (want %q or %q)", in.AgentType, AgentExplore, AgentGeneral)
	}

	sub := &Agent{
		Provider: t.Provider,
		Tools:    toolset,
		System:   subagentSystemPrompt(t.System, in.Description),
		MaxSteps: t.MaxSteps,
	}

	res, err := sub.Run(ctx, in.Prompt, nil, NopObserver{})
	if err != nil {
		return "", fmt.Errorf("subagent %q: %w", in.Description, err)
	}
	return res.Text, nil
}

func subagentSystemPrompt(base, description string) string {
	var b strings.Builder
	b.WriteString(base)
	b.WriteString("\n\n## Delegated task\n\nYou are a subagent handling one delegated piece ")
	b.WriteString("of work: ")
	b.WriteString(description)
	b.WriteString(". Your conversation is isolated — only your final answer goes back to " +
		"the agent that delegated this. Investigate as thoroughly as the task needs, but " +
		"keep your final answer focused: a summary of what you found or did, not a " +
		"transcript of your steps.")
	return b.String()
}
