package tools

import (
	"context"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
)

func allowAllBash(t *testing.T) Bash {
	t.Helper()
	return Bash{
		WorkDir: t.TempDir(),
		Perms:   permission.NewBroker(permission.ModeAsk, permission.AllowAll{}),
	}
}

func run(t *testing.T, ctx context.Context, b Bash, in map[string]any) (string, error) {
	t.Helper()
	raw, err := json.Marshal(in)
	if err != nil {
		t.Fatal(err)
	}
	return b.Run(ctx, raw)
}

func TestBashCapturesOutput(t *testing.T) {
	b := allowAllBash(t)
	out, err := run(t, context.Background(), b, map[string]any{"command": "echo hola"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "hola" {
		t.Errorf("out = %q, want %q", out, "hola")
	}
}

func TestBashCapturesStderrAndExitCode(t *testing.T) {
	b := allowAllBash(t)
	_, err := run(t, context.Background(), b, map[string]any{"command": "echo boom >&2; exit 3"})
	if err == nil {
		t.Fatal("expected an error for a non-zero exit")
	}
	// Both facts must reach the model: how it failed and what it printed.
	if !strings.Contains(err.Error(), "exit status 3") {
		t.Errorf("exit code missing from %q", err)
	}
	if !strings.Contains(err.Error(), "boom") {
		t.Errorf("stderr missing from %q", err)
	}
}

func TestBashRunsInWorkDir(t *testing.T) {
	dir := t.TempDir()
	b := Bash{WorkDir: dir, Perms: permission.NewBroker(permission.ModeAsk, permission.AllowAll{})}
	out, err := run(t, context.Background(), b, map[string]any{"command": "pwd"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	// macOS resolves /var to /private/var; compare the resolved paths.
	want, _ := filepath.EvalSymlinks(dir)
	got, _ := filepath.EvalSymlinks(strings.TrimSpace(out))
	if got != want {
		t.Errorf("pwd = %q, want %q", got, want)
	}
}

// The Python prototype's subprocess.run had no timeout: `npm run dev` hung the
// turn forever. This is the regression test for that.
func TestBashTimesOutInsteadOfHanging(t *testing.T) {
	b := allowAllBash(t)
	start := time.Now()
	_, err := run(t, context.Background(), b, map[string]any{
		"command":     "sleep 60",
		"timeout_sec": 1,
	})
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected a timeout error")
	}
	if !strings.Contains(err.Error(), "timed out") {
		t.Errorf("err = %v, want a timeout", err)
	}
	if elapsed > 10*time.Second {
		t.Errorf("took %s to give up; the timeout is not being enforced", elapsed)
	}
}

// Cancelling the turn must kill the child process, not orphan it. In the
// prototype, Esc only stopped rendering while the subprocess kept running.
func TestBashCancelKillsTheProcessTree(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "still-alive")
	b := Bash{WorkDir: dir, Perms: permission.NewBroker(permission.ModeAsk, permission.AllowAll{})}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		// A child that would create the marker file if it survived 3 seconds,
		// wrapped in a subshell so we also prove the whole group dies.
		_, err := run(t, ctx, b, map[string]any{
			"command":     "(sleep 3; touch " + marker + ") & wait",
			"timeout_sec": 30,
		})
		done <- err
	}()

	time.Sleep(300 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != context.Canceled {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("cancelling did not stop the command")
	}

	// Give an orphaned process time to reach its touch.
	time.Sleep(3500 * time.Millisecond)
	if _, err := os.Stat(marker); err == nil {
		t.Error("the child process survived cancellation and kept running")
	}
}

func TestBashRequiresPermission(t *testing.T) {
	b := Bash{
		WorkDir: t.TempDir(),
		Perms:   permission.NewBroker(permission.ModeAsk, permission.DenyAll{}),
	}
	_, err := run(t, context.Background(), b, map[string]any{"command": "echo nope"})
	if err != permission.ErrDenied {
		t.Errorf("err = %v, want ErrDenied", err)
	}
}

// Plan mode decides without a prompt: read-only commands run, others do not.
func TestBashPlanModeGate(t *testing.T) {
	b := Bash{
		WorkDir: t.TempDir(),
		Perms:   permission.NewBroker(permission.ModePlan, permission.DenyAll{}),
	}

	if _, err := run(t, context.Background(), b, map[string]any{"command": "echo readonly"}); err != nil {
		t.Errorf("read-only command should run in Plan mode: %v", err)
	}
	if _, err := run(t, context.Background(), b, map[string]any{"command": "mkdir newdir"}); err != permission.ErrDenied {
		t.Errorf("mutating command should be denied in Plan mode, got %v", err)
	}
}

func TestBashRejectsEmptyCommand(t *testing.T) {
	b := allowAllBash(t)
	if _, err := run(t, context.Background(), b, map[string]any{"command": "   "}); err == nil {
		t.Error("expected an error for an empty command")
	}
}

func TestTruncateOutputKeepsBothEnds(t *testing.T) {
	body := strings.Repeat("x", MaxOutputBytes*2)
	got := truncateOutput("HEAD" + body + "TAIL")
	if !strings.HasPrefix(got, "HEAD") {
		t.Error("truncation dropped the start of the output")
	}
	if !strings.HasSuffix(got, "TAIL") {
		t.Error("truncation dropped the end, where the error message usually is")
	}
	if !strings.Contains(got, "truncated") {
		t.Error("truncation was not announced")
	}
}

func TestBashNoOutputIsExplicit(t *testing.T) {
	b := allowAllBash(t)
	out, err := run(t, context.Background(), b, map[string]any{"command": "true"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "(no output)" {
		t.Errorf("out = %q; an empty string reads as a broken tool to the model", out)
	}
}

func TestBashAvailable(t *testing.T) {
	if _, err := exec.LookPath("bash"); err != nil {
		t.Skip("bash not available")
	}
}
