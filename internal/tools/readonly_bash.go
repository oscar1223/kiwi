package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/oscar1223/kiwi/internal/permission"
)

// ReadOnlyBash is Bash hard-restricted to commands permission.IsReadOnlyCommand
// accepts — regardless of the active permission mode.
//
// It exists for subagents delegated read-only investigation (see
// agent.AgentExplore): even when the parent session is in Work mode, where
// plain bash asks once and then runs anything, an explore subagent must
// never be able to mutate anything. That has to be true by construction —
// not because it would ask and get denied, but because the capability to
// ask for it was never on the table. A mode setting is state that could
// change out from under it mid-run; this type cannot.
type ReadOnlyBash struct {
	Bash
}

func (ReadOnlyBash) Name() string { return "bash" }

func (ReadOnlyBash) Description() string {
	return "Run a read-only shell command (ls, cat, grep, find, git status/log/diff, " +
		"pwd, wc, ...) and return its output. Commands that write, delete, or otherwise " +
		"change anything are rejected before they run."
}

func (t ReadOnlyBash) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if !permission.IsReadOnlyCommand(in.Command) {
		return "", fmt.Errorf("this subagent may only run read-only commands; rejected: %q", in.Command)
	}
	return t.Bash.Run(ctx, input)
}
