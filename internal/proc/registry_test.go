package proc

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func waitForStatus(t *testing.T, p *Process, want Status, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if p.Status() == want {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("process %s: status = %s after %s, want %s", p.ID, p.Status(), timeout, want)
}

func TestStartReturnsImmediately(t *testing.T) {
	r := NewRegistry()
	start := time.Now()
	p, err := r.Start(t.TempDir(), "sleep 5")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	if elapsed := time.Since(start); elapsed > 1*time.Second {
		t.Errorf("Start blocked for %s; it must return before the command finishes", elapsed)
	}
	r.Kill(p.ID) // cleanup
}

func TestOutputAccumulatesAndReadNewIsIncremental(t *testing.T) {
	r := NewRegistry()
	p, err := r.Start(t.TempDir(), "echo one; sleep 0.2; echo two")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	// Give "one" time to print before the first read.
	time.Sleep(100 * time.Millisecond)
	first, status := p.ReadNew()
	if !strings.Contains(first, "one") {
		t.Errorf("first read = %q, want it to contain %q", first, "one")
	}
	if strings.Contains(first, "two") {
		t.Errorf("first read = %q, should not yet contain %q", first, "two")
	}
	if status != StatusRunning {
		t.Errorf("status = %s, want running", status)
	}

	waitForStatus(t, p, StatusExited, 3*time.Second)
	second, _ := p.ReadNew()
	if !strings.Contains(second, "two") {
		t.Errorf("second read = %q, want it to contain %q", second, "two")
	}
	if strings.Contains(second, "one") {
		t.Errorf("second read = %q should not repeat %q", second, "one")
	}
}

func TestKillStopsARunningProcess(t *testing.T) {
	r := NewRegistry()
	p, err := r.Start(t.TempDir(), "sleep 30")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	if err := r.Kill(p.ID); err != nil {
		t.Fatalf("Kill: %v", err)
	}
	if got := p.Status(); got != StatusKilled {
		t.Errorf("status = %s, want killed", got)
	}
}

func TestKillOnAnAlreadyExitedProcessIsANoop(t *testing.T) {
	r := NewRegistry()
	p, err := r.Start(t.TempDir(), "true")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	waitForStatus(t, p, StatusExited, 3*time.Second)

	if err := r.Kill(p.ID); err != nil {
		t.Fatalf("Kill: %v", err)
	}
	if got := p.Status(); got != StatusExited {
		t.Errorf("status = %s; killing an exited process must not overwrite its exit status", got)
	}
}

func TestGetUnknownID(t *testing.T) {
	r := NewRegistry()
	if _, err := r.Get("nope"); !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

func TestKillUnknownID(t *testing.T) {
	r := NewRegistry()
	if err := r.Kill("nope"); !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

func TestStartRunsInWorkDir(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry()
	p, err := r.Start(dir, "pwd")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	waitForStatus(t, p, StatusExited, 3*time.Second)

	out, _ := p.ReadNew()
	want, _ := filepath.EvalSymlinks(dir)
	got, _ := filepath.EvalSymlinks(strings.TrimSpace(out))
	if got != want {
		t.Errorf("pwd = %q, want %q", got, want)
	}
}

// KillAll must actually terminate the OS-level process group, not just
// relabel Go-side state — a background dev server must not outlive the
// session that started it. This is the regression test for exactly the bug
// the Python prototype had: Esc only stopped rendering while the subprocess
// kept running.
func TestKillAllActuallyTerminatesTheProcessTree(t *testing.T) {
	dir := t.TempDir()
	marker := filepath.Join(dir, "still-alive")
	r := NewRegistry()

	// A child that, left alone, creates the marker after a delay — wrapped
	// in a subshell so this also proves the whole process group dies, not
	// just the immediate bash -c leader.
	_, err := r.Start(dir, "(sleep 2; touch "+marker+") & wait")
	if err != nil {
		t.Fatalf("Start: %v", err)
	}

	r.KillAll()

	time.Sleep(2500 * time.Millisecond)
	if _, err := os.Stat(marker); err == nil {
		t.Error("a background process survived KillAll and kept running")
	}
}

func TestKillAllIsSafeWithNoProcesses(t *testing.T) {
	NewRegistry().KillAll() // must not panic
}
