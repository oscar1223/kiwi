//go:build !windows

// Package procgroup isolates the process-group handling that differs between
// Unix and Windows, so the packages that spawn shell commands don't have to
// carry build tags of their own.
package procgroup

import (
	"os"
	"os/exec"
	"syscall"
)

// Configure gives cmd its own process group, which is what makes Kill able to
// take down the whole tree: `npm run dev` spawns children that would otherwise
// survive and keep holding the port.
func Configure(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// Kill stops p's whole process group. A negative pid targets the group rather
// than the leader alone (see Configure). A process that already exited is not
// an error — there is simply nothing left to kill.
func Kill(p *os.Process) error {
	if p == nil {
		return nil
	}
	return syscall.Kill(-p.Pid, syscall.SIGKILL)
}
