package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os/exec"
	"strings"
	"syscall"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
)

const (
	// DefaultTimeout applies when the model does not ask for one.
	DefaultTimeout = 2 * time.Minute
	// MaxTimeout caps what the model may request.
	MaxTimeout = 10 * time.Minute
	// MaxOutputBytes bounds what a command can push into the context window.
	MaxOutputBytes = 30 * 1024
)

// Bash runs shell commands.
//
// Two things the Python prototype got wrong are fixed here by construction:
// every run has a timeout, and the process is bound to the turn's context, so
// cancelling a turn kills the child instead of orphaning it.
type Bash struct {
	WorkDir string
	Perms   *permission.Broker
}

func (Bash) Name() string { return "bash" }

func (Bash) Description() string {
	return "Run a shell command and return its combined output. " +
		"Commands that do not exit on their own (servers, watchers, tail -f) will " +
		"hit the timeout — start those with run_background instead."
}

func (Bash) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command":     map[string]any{"type": "string", "description": "The shell command to run."},
			"timeout_sec": map[string]any{"type": "integer", "description": "Seconds before the command is killed. Defaults to 120, capped at 600."},
		},
		"required": []string{"command"},
	}
}

func (t Bash) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Command    string `json:"command"`
		TimeoutSec int    `json:"timeout_sec"`
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

	timeout := DefaultTimeout
	if in.TimeoutSec > 0 {
		timeout = time.Duration(in.TimeoutSec) * time.Second
		if timeout > MaxTimeout {
			timeout = MaxTimeout
		}
	}

	// Two nested deadlines: the turn's ctx (cancelled by Esc or Ctrl-C) and
	// this command's own timeout. Whichever fires first kills the process.
	runCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(runCtx, "bash", "-c", in.Command)
	cmd.Dir = t.WorkDir

	// Give the child its own process group and kill the group, not just the
	// leader: `npm run dev` spawns children that would otherwise survive and
	// keep holding the port.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		if cmd.Process == nil {
			return nil
		}
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}

	var buf bytes.Buffer
	cmd.Stdout = &buf
	cmd.Stderr = &buf

	err := cmd.Run()
	output := truncateOutput(buf.String())

	switch {
	case errors.Is(runCtx.Err(), context.DeadlineExceeded):
		return "", fmt.Errorf("command timed out after %s and was killed%s",
			timeout, outputSuffix(output))
	case ctx.Err() != nil:
		// The turn was cancelled; propagate so the agent stops the loop.
		return "", ctx.Err()
	}

	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return "", fmt.Errorf("exit status %d%s", exitErr.ExitCode(), outputSuffix(output))
		}
		return "", err
	}

	if output == "" {
		return "(no output)", nil
	}
	return output, nil
}

func outputSuffix(output string) string {
	if output == "" {
		return " (no output)"
	}
	return ":\n" + output
}

func truncateOutput(s string) string {
	s = strings.TrimRight(s, "\n")
	if len(s) <= MaxOutputBytes {
		return s
	}
	// Keep both ends: the start usually says what ran, the end says how it
	// failed. Cutting only the tail throws away the error message.
	head := MaxOutputBytes * 2 / 3
	tail := MaxOutputBytes - head
	return s[:head] +
		fmt.Sprintf("\n\n… [%d bytes truncated] …\n\n", len(s)-MaxOutputBytes) +
		s[len(s)-tail:]
}
