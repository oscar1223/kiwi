//go:build windows

package procgroup

import (
	"os"
	"os/exec"
	"strconv"
	"syscall"
)

// Configure gives cmd its own process group. Windows has no Setpgid;
// CREATE_NEW_PROCESS_GROUP is the closest equivalent, and it keeps a Ctrl-C in
// the parent console from reaching the child.
func Configure(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: syscall.CREATE_NEW_PROCESS_GROUP}
}

// Kill stops p and everything it spawned. There is no group-wide signal on
// Windows, so this shells out to taskkill /T — the documented way to end a
// process tree — and falls back to killing the leader alone if that fails.
func Kill(p *os.Process) error {
	if p == nil {
		return nil
	}
	if err := exec.Command("taskkill", "/F", "/T", "/PID", strconv.Itoa(p.Pid)).Run(); err != nil {
		return p.Kill()
	}
	return nil
}
