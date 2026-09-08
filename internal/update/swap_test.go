//go:build !windows

package update

import (
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"testing"
	"time"
)

// TestSwapOverRunningBinary is the claim the whole self-update rests on: on
// Unix a binary can be replaced while it is executing, because the running
// process holds the old inode open and only the directory entry moves.
//
// If this ever stops holding, `kiwi update` would kill the session that ran it.
func TestSwapOverRunningBinary(t *testing.T) {
	sleep, err := exec.LookPath("sleep")
	if err != nil {
		t.Skip("no sleep binary to copy")
	}

	dir := t.TempDir()
	target := filepath.Join(dir, "victim")

	// A real executable, not a script: a shell would re-read the file and
	// prove nothing about exec semantics.
	src, err := os.ReadFile(sleep)
	if err != nil {
		t.Fatalf("reading %s: %v", sleep, err)
	}
	if err := os.WriteFile(target, src, 0o755); err != nil {
		t.Fatalf("writing target: %v", err)
	}

	cmd := exec.Command(target, "30")
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting the victim: %v", err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})

	// Give it a moment to actually be executing, not just forked.
	time.Sleep(100 * time.Millisecond)

	staged := filepath.Join(dir, ".staged")
	want := []byte("the replacement\n")
	if err := os.WriteFile(staged, want, 0o755); err != nil {
		t.Fatalf("writing staged file: %v", err)
	}

	if err := swap(staged, target); err != nil {
		t.Fatalf("swap over a running binary: %v", err)
	}

	// The path now holds the new file...
	got, err := os.ReadFile(target)
	if err != nil {
		t.Fatalf("reading target after swap: %v", err)
	}
	if string(got) != string(want) {
		t.Errorf("target content = %q, want %q", got, want)
	}

	// ...and the process that was running the old one is still alive.
	// Signal 0 checks for existence without delivering anything.
	if err := cmd.Process.Signal(syscall.Signal(0)); err != nil {
		t.Errorf("the running process died when its binary was replaced: %v", err)
	}

	// swap must not leave the staged file behind.
	if _, err := os.Stat(staged); !os.IsNotExist(err) {
		t.Errorf("staged file still exists after swap")
	}
}

// TestStagePutsFileBesideTarget guards the reason stage() takes a directory:
// the rename in swap() is only atomic within one filesystem, so the staged
// file has to land next to the binary it will replace.
func TestStagePutsFileBesideTarget(t *testing.T) {
	dir := t.TempDir()

	path, err := stage(newReader("binary contents"), dir)
	if err != nil {
		t.Fatalf("stage: %v", err)
	}
	if got := filepath.Dir(path); got != dir {
		t.Errorf("staged into %q, want %q", got, dir)
	}

	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading staged file: %v", err)
	}
	if string(b) != "binary contents" {
		t.Errorf("staged content = %q", b)
	}
}
