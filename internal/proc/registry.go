// Package proc runs and tracks background shell commands — the ones that
// don't exit on their own (dev servers, watchers, tail -f) and would
// otherwise hang the foreground bash tool's timeout for nothing.
//
// A background process deliberately outlives the turn that started it: it is
// not tied to that turn's context, only to the Registry it was started from.
// It keeps running until something explicitly stops it — kill_shell, or the
// Registry's own KillAll when Kiwi exits.
package proc

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os/exec"
	"sync"
	"syscall"
)

// defaultBufferCap bounds how much output one process's ring buffer retains.
const defaultBufferCap = 256 * 1024

// Status is a background process's lifecycle state.
type Status string

const (
	StatusRunning Status = "running"
	StatusExited  Status = "exited"
	StatusKilled  Status = "killed"
)

// ErrNotFound is returned when an id matches no tracked process.
var ErrNotFound = errors.New("proc: not found")

// Process is one background command the Registry is tracking.
type Process struct {
	ID      string
	Command string

	mu      sync.Mutex
	buf     *ringBuffer
	cursor  int64
	status  Status
	exitErr error
	cmd     *exec.Cmd
}

// Status reports the process's current lifecycle state.
func (p *Process) Status() Status {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.status
}

// ReadNew returns output written since the last ReadNew call (or since start,
// on the first call), along with the current status.
func (p *Process) ReadNew() (string, Status) {
	p.mu.Lock()
	cursor := p.cursor
	status := p.status
	p.mu.Unlock()

	data, next := p.buf.Since(cursor)

	p.mu.Lock()
	p.cursor = next
	p.mu.Unlock()

	return string(data), status
}

// kill stops the process's whole process group. A process that has already
// stopped on its own is left alone — killing something already finished is
// not an error, it just has nothing left to do.
func (p *Process) kill() error {
	p.mu.Lock()
	if p.status != StatusRunning {
		p.mu.Unlock()
		return nil
	}
	p.status = StatusKilled
	proc := p.cmd.Process
	p.mu.Unlock()

	if proc == nil {
		return nil
	}
	// Negative pid targets the whole process group (see Start's Setpgid),
	// so a server's own child processes die with it instead of lingering.
	return syscall.Kill(-proc.Pid, syscall.SIGKILL)
}

// Registry tracks every background process started through it.
type Registry struct {
	mu    sync.Mutex
	procs map[string]*Process
}

func NewRegistry() *Registry {
	return &Registry{procs: map[string]*Process{}}
}

// Start launches command in its own process group and returns immediately —
// it does not wait for the command to finish. workDir sets the command's
// working directory.
func (r *Registry) Start(workDir, command string) (*Process, error) {
	id, err := newID()
	if err != nil {
		return nil, err
	}

	cmd := exec.Command("bash", "-c", command)
	cmd.Dir = workDir
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	buf := newRingBuffer(defaultBufferCap)
	cmd.Stdout = buf
	cmd.Stderr = buf

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("proc: starting %q: %w", command, err)
	}

	p := &Process{
		ID:      id,
		Command: command,
		buf:     buf,
		status:  StatusRunning,
		cmd:     cmd,
	}

	r.mu.Lock()
	r.procs[id] = p
	r.mu.Unlock()

	go func() {
		err := cmd.Wait()
		p.mu.Lock()
		// A process that was killed already has its terminal status set;
		// Wait returning here just means the SIGKILL took effect. Only a
		// process that stopped on its own transitions to Exited.
		if p.status == StatusRunning {
			p.status = StatusExited
			p.exitErr = err
		}
		p.mu.Unlock()
	}()

	return p, nil
}

// Get returns a tracked process by id.
func (r *Registry) Get(id string) (*Process, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	p, ok := r.procs[id]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrNotFound, id)
	}
	return p, nil
}

// Kill stops a tracked process by id.
func (r *Registry) Kill(id string) error {
	p, err := r.Get(id)
	if err != nil {
		return err
	}
	return p.kill()
}

// KillAll stops every process the Registry is tracking — called when Kiwi
// exits, so a background dev server does not outlive the session that
// started it.
func (r *Registry) KillAll() {
	r.mu.Lock()
	procs := make([]*Process, 0, len(r.procs))
	for _, p := range r.procs {
		procs = append(procs, p)
	}
	r.mu.Unlock()

	for _, p := range procs {
		p.kill()
	}
}

func newID() (string, error) {
	b := make([]byte, 4)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}
