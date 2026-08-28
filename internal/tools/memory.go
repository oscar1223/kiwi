package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/oscar1223/kiwi/internal/memory"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Remember lets the model save a fact that outlives the session.
//
// It goes through the permission broker under ActionWrite, like any other
// file write, and for the same reason: in Plan mode Kiwi is read-only, and
// "read-only except for the notes it keeps about you" is not a distinction a
// user should have to discover. Work mode saves without asking; Ask mode
// shows the exact line first.
type Remember struct {
	Store *memory.Store
	Perms *permission.Broker
}

func (Remember) Name() string { return "remember" }

func (Remember) Description() string {
	return "Save one short fact for future sessions: how the user likes to work, a " +
		"durable decision, or a constraint that is not obvious from the code. Use " +
		"scope \"project\" for this codebase and \"global\" for the user themselves. " +
		"Do not use it for what the code, the README or git history already record, " +
		"or for anything that only matters until this conversation ends."
}

func (Remember) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"fact": map[string]any{
				"type":        "string",
				"description": "The fact to remember, as one self-contained sentence.",
			},
			"scope": map[string]any{
				"type":        "string",
				"enum":        []string{string(memory.Project), string(memory.Global)},
				"description": "project (default) applies to this codebase only; global follows the user everywhere.",
			},
		},
		"required": []string{"fact"},
	}
}

func (t Remember) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Fact  string `json:"fact"`
		Scope string `json:"scope"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	scope := memory.Scope(in.Scope)
	if in.Scope == "" {
		scope = memory.Project
	}
	if !scope.Valid() {
		return "", fmt.Errorf("unknown scope %q (want %q or %q)", in.Scope, memory.Project, memory.Global)
	}
	if in.Fact == "" {
		return "", fmt.Errorf("fact is required")
	}

	if err := t.Perms.Ask(ctx, permission.Action{
		Name:   permission.ActionWrite,
		Detail: fmt.Sprintf("remember (%s): %s", scope, in.Fact),
	}); err != nil {
		return "", err
	}

	dropped, err := t.Store.Append(scope, in.Fact)
	if err != nil {
		return "", err
	}
	if dropped > 0 {
		return fmt.Sprintf("Saved to %s memory. It was full, so the %d oldest note(s) were dropped — "+
			"tell the user if any of them mattered.", scope, dropped), nil
	}
	return fmt.Sprintf("Saved to %s memory.", scope), nil
}
