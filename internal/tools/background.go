package tools

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/proc"
)

// BackgroundBash starts a command that outlives the tool call — servers,
// watchers, anything that does not exit on its own. Ordinary bash already
// covers commands that finish; this exists specifically for the case that
// used to hang the Python prototype's turn forever with no timeout at all.
type BackgroundBash struct {
	WorkDir string
	Perms   *permission.Broker
	Procs   *proc.Registry
}

func (BackgroundBash) Name() string { return "bash_background" }

func (BackgroundBash) Description() string {
	return "Start a shell command that keeps running after this call returns — for dev " +
		"servers, watchers, or anything that does not exit on its own. Returns an id; " +
		"read its output with bash_output and stop it with kill_shell. Use bash instead " +
		"for a command that finishes by itself."
}

func (BackgroundBash) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command": map[string]any{"type": "string", "description": "The shell command to run in the background."},
		},
		"required": []string{"command"},
	}
}

func (t BackgroundBash) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Command string `json:"command"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if strings.TrimSpace(in.Command) == "" {
		return "", errors.New("command is required")
	}

	if err := t.Perms.Ask(ctx, permission.Action{
		Name:      permission.ActionBash,
		Detail:    in.Command,
		Dangerous: permission.IsDangerous(in.Command),
	}); err != nil {
		return "", err
	}

	p, err := t.Procs.Start(t.WorkDir, in.Command)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("Started as %s. Use bash_output with this id to read its output.", p.ID), nil
}

// BackgroundOutput reads what a background command has printed since the
// last read, and whether it is still running.
type BackgroundOutput struct {
	Procs *proc.Registry
}

func (BackgroundOutput) Name() string { return "bash_output" }

func (BackgroundOutput) Description() string {
	return "Read a background command's output since you last read it, and its status " +
		"(running, exited, or killed)."
}

func (BackgroundOutput) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{"type": "string", "description": "The id bash_background returned."},
		},
		"required": []string{"id"},
	}
}

func (t BackgroundOutput) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	p, err := t.Procs.Get(in.ID)
	if err != nil {
		return "", err
	}
	out, status := p.ReadNew()
	if strings.TrimSpace(out) == "" {
		out = "(no new output)"
	}
	return fmt.Sprintf("[%s] %s\n%s", status, p.Command, out), nil
}

// KillShell stops a background command.
type KillShell struct {
	Procs *proc.Registry
}

func (KillShell) Name() string { return "kill_shell" }
func (KillShell) Description() string {
	return "Stop a background command started with bash_background."
}

func (KillShell) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"id": map[string]any{"type": "string", "description": "The id bash_background returned."},
		},
		"required": []string{"id"},
	}
}

func (t KillShell) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if err := t.Procs.Kill(in.ID); err != nil {
		return "", err
	}
	return "Stopped " + in.ID + ".", nil
}
