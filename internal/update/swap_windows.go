package update

import (
	"os"
	"path/filepath"
)

// oldSuffix marks the displaced binary. Windows will not let us delete a file
// that is running, but it will let us rename it, so the previous binary lingers
// under this suffix until a later start clears it.
const oldSuffix = ".old"

// swap moves the staged binary over the target.
//
// Windows refuses to overwrite a running .exe, so the running one is renamed
// out of the way first. If putting the new binary in place then fails, the old
// name goes back: a failed update must not leave the user without a kiwi.
func swap(staged, target string) error {
	old := target + oldSuffix
	// A leftover from a previous update would block the rename.
	_ = os.Remove(old)

	if err := os.Rename(target, old); err != nil {
		return err
	}
	if err := os.Rename(staged, target); err != nil {
		_ = os.Rename(old, target)
		return err
	}

	// Fails while the old binary is still running, which is the normal case
	// when kiwi updates itself. CleanupOld finishes the job next time.
	_ = os.Remove(old)
	return nil
}

// CleanupOld removes the binary displaced by a previous update. It runs at
// startup, when the file is no longer executing and can finally be deleted.
func CleanupOld() {
	exe, err := os.Executable()
	if err != nil {
		return
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	_ = os.Remove(exe + oldSuffix)
}
