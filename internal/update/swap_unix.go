//go:build !windows

package update

import "os"

// swap moves the staged binary over the target.
//
// Renaming over a running binary is fine on Unix: the running process holds
// the old inode open and keeps executing it, while the directory entry points
// at the new file for every later exec. The rename itself is atomic, so there
// is no moment where the path exists but holds half a binary.
func swap(staged, target string) error {
	return os.Rename(staged, target)
}

// CleanupOld has nothing to do on Unix — no leftover file is ever created.
func CleanupOld() {}
