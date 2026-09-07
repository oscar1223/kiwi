// Package checkpoint snapshots the working tree before a turn that writes, so
// the work can be undone afterwards.
//
// The snapshots live in a git repository of Kiwi's own, kept under Kiwi's
// config directory and pointed at the project through GIT_WORK_TREE. Nothing
// Kiwi does here touches the project's own .git: not its index, not its stash,
// not its history, not its hooks — those live in the real GIT_DIR, and this
// package never names it. That separation is the whole design. A checkpoint
// implemented with `git stash` would be simpler and would also mean Kiwi and
// the user were fighting over the same index.
//
// The project must still be a git repository. Not because the shadow repo
// needs one, but because .gitignore is what keeps a snapshot from swallowing
// node_modules — and a directory nobody has bothered to make a repo is exactly
// the directory with no .gitignore.
package checkpoint

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/oscar1223/kiwi/internal/config"
)

// ErrNoRepo means the working directory is not inside a git repository, so
// there is nothing to snapshot safely. Callers are expected to degrade —
// saying the safety net is missing — rather than to fail.
var ErrNoRepo = errors.New("checkpoint: not a git repository")

// ErrNoGit means the git binary is not on PATH.
var ErrNoGit = errors.New("checkpoint: git is not installed")

// ErrNotFound means the requested checkpoint id is not in the shadow repo.
var ErrNotFound = errors.New("checkpoint: no such checkpoint")

// maxCheckpoints is how many snapshots one project keeps. Past this the
// history is re-rooted (see Prune): a shadow repo that only ever grows is a
// bug that shows up three months in, not on the first day.
const maxCheckpoints = 50

// commandTimeout bounds any single git invocation. A snapshot that hangs would
// hang the turn it was supposed to protect.
const commandTimeout = 30 * time.Second

// ref is the branch the snapshots live on. It is inside the shadow repo, so
// the name only has to be stable, not unusual.
const ref = "refs/heads/kiwi"

// Checkpoint is one snapshot.
type Checkpoint struct {
	// ID is the commit hash in the shadow repo.
	ID string
	// Label is what the snapshot was taken for — normally the user's message.
	Label string
	When  time.Time
	// Files is how many paths changed relative to the previous snapshot.
	Files int
}

// Short returns the abbreviated id used in messages to the user.
func (c Checkpoint) Short() string {
	if len(c.ID) > 8 {
		return c.ID[:8]
	}
	return c.ID
}

// Store is the snapshot history for one working directory.
type Store struct {
	workDir string
	gitDir  string
}

// New prepares the shadow repository for workDir, creating it on first use.
//
// It returns ErrNoRepo when workDir is not inside a git repository and
// ErrNoGit when git is missing; both are conditions the caller reports rather
// than fails on.
func New(ctx context.Context, workDir string) (*Store, error) {
	if _, err := exec.LookPath("git"); err != nil {
		return nil, ErrNoGit
	}
	abs, err := filepath.Abs(workDir)
	if err != nil {
		return nil, err
	}
	if !insideRepo(ctx, abs) {
		return nil, ErrNoRepo
	}

	root, err := config.Dir()
	if err != nil {
		return nil, err
	}
	// The directory is keyed by a hash of the absolute path rather than by the
	// path itself: two projects can share a basename, and a path used verbatim
	// would need escaping on every platform Kiwi runs on.
	sum := sha256.Sum256([]byte(abs))
	gitDir := filepath.Join(root, "checkpoints", hex.EncodeToString(sum[:])[:16])

	s := &Store{workDir: abs, gitDir: gitDir}
	if err := s.init(ctx); err != nil {
		return nil, err
	}
	return s, nil
}

// insideRepo reports whether dir is inside a git work tree. A plain directory
// is not an error anywhere else in Kiwi, so the failure is swallowed here and
// turned into a plain "no".
func insideRepo(ctx context.Context, dir string) bool {
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--is-inside-work-tree")
	cmd.Dir = dir
	out, err := cmd.Output()
	return err == nil && strings.TrimSpace(string(out)) == "true"
}

// init creates the shadow repo if it does not exist yet. It is idempotent, so
// every New can call it without checking first.
func (s *Store) init(ctx context.Context) error {
	if _, err := os.Stat(filepath.Join(s.gitDir, "HEAD")); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(s.gitDir), 0o700); err != nil {
		return err
	}
	if _, err := s.git(ctx, "init", "--quiet", "--initial-branch=kiwi"); err != nil {
		return err
	}
	// A committer identity, set on this repo only. Without it a user who has
	// never run `git config --global user.email` — a fresh machine, a CI
	// container — would find that snapshots fail for a reason that has
	// nothing to do with them.
	for _, kv := range [][2]string{
		{"user.name", "kiwi"},
		{"user.email", "kiwi@localhost"},
		{"commit.gpgsign", "false"},
		// The snapshot is a record, not a review: signing it, running the
		// project's hooks on it, or letting a global template add anything
		// would all be surprises.
		{"core.hooksPath", filepath.Join(s.gitDir, "no-hooks")},
	} {
		if _, err := s.git(ctx, "config", kv[0], kv[1]); err != nil {
			return err
		}
	}
	return nil
}

// git runs one git command against the shadow repo with the project as its
// work tree.
func (s *Store) git(ctx context.Context, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, commandTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "git", args...)
	cmd.Dir = s.workDir
	// GIT_DIR and GIT_WORK_TREE together are what redirect every read and
	// write to the shadow repo while still seeing the project's files — and,
	// importantly, the project's .gitignore, which git reads from the work
	// tree regardless of which repo it belongs to.
	cmd.Env = append(os.Environ(),
		"GIT_DIR="+s.gitDir,
		"GIT_WORK_TREE="+s.workDir,
		// Kiwi is not a terminal git can prompt through. Without this a repo
		// with a credential helper could block the turn on a hidden prompt.
		"GIT_TERMINAL_PROMPT=0",
		"GIT_OPTIONAL_LOCKS=0",
	)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			msg = err.Error()
		}
		return "", fmt.Errorf("git %s: %s", args[0], msg)
	}
	return strings.TrimSpace(string(out)), nil
}

// Take snapshots the working tree and returns the new checkpoint's id.
//
// label is what the user typed, trimmed to one line: a checkpoint the user
// cannot recognise is a checkpoint they will not dare restore.
func (s *Store) Take(ctx context.Context, label string) (string, error) {
	// Pruning before rather than after is deliberate. Re-rooting drops the
	// older snapshots, so doing it first means the checkpoint this call is
	// about to create still sits on top of a restorable parent — prune last
	// and the very snapshot you just took would be the one you could not undo.
	if err := s.Prune(ctx, maxCheckpoints); err != nil {
		return "", err
	}
	if _, err := s.git(ctx, "add", "-A"); err != nil {
		return "", err
	}
	args := []string{"commit", "--quiet", "--allow-empty", "--no-verify", "-m", oneLine(label)}
	if _, err := s.git(ctx, args...); err != nil {
		return "", err
	}
	return s.Head(ctx)
}

// Head is the id of the most recent checkpoint, or "" when none exist yet.
func (s *Store) Head(ctx context.Context) (string, error) {
	out, err := s.git(ctx, "rev-parse", "--verify", "--quiet", ref)
	if err != nil {
		// An unborn branch is the normal state before the first Take, not a
		// failure — `rev-parse --verify --quiet` exits non-zero for it.
		return "", nil
	}
	return out, nil
}

// Diff returns the unified diff between a checkpoint and the working tree as
// it stands now. An empty string means nothing changed.
func (s *Store) Diff(ctx context.Context, from string) (string, error) {
	if err := s.must(ctx, from); err != nil {
		return "", err
	}
	// Staging first is what makes files created since the snapshot show up:
	// `git diff <commit>` alone only reports paths the index already knows.
	if _, err := s.git(ctx, "add", "-A"); err != nil {
		return "", err
	}
	return s.git(ctx, "diff", "--cached", "--no-color", from)
}

// Files lists the paths that changed between a checkpoint and the working
// tree, for the confirmation prompt shown before restoring.
func (s *Store) Files(ctx context.Context, from string) ([]string, error) {
	if err := s.must(ctx, from); err != nil {
		return nil, err
	}
	if _, err := s.git(ctx, "add", "-A"); err != nil {
		return nil, err
	}
	out, err := s.git(ctx, "diff", "--cached", "--name-only", from)
	if err != nil || out == "" {
		return nil, err
	}
	return strings.Split(out, "\n"), nil
}

// Restore puts the working tree back to a checkpoint.
//
// Files created since it are removed, files deleted since it come back. That
// is the point, and it is also why the caller confirms with the user first:
// anything the user typed by hand in between goes with it.
func (s *Store) Restore(ctx context.Context, id string) error {
	if err := s.must(ctx, id); err != nil {
		return err
	}
	// Staging first is what lets reset --hard delete files that appeared after
	// the snapshot: reset only removes paths the index is carrying.
	if _, err := s.git(ctx, "add", "-A"); err != nil {
		return err
	}
	_, err := s.git(ctx, "reset", "--hard", "--quiet", id)
	return err
}

// List returns the checkpoints, newest first.
func (s *Store) List(ctx context.Context) ([]Checkpoint, error) {
	head, err := s.Head(ctx)
	if err != nil || head == "" {
		return nil, err
	}
	// The separator is a unit separator rather than anything typable, so a
	// label containing punctuation cannot split a row.
	out, err := s.git(ctx, "log", ref, "--format=%H\x1f%ct\x1f%s", "-n", strconv.Itoa(maxCheckpoints))
	if err != nil {
		return nil, err
	}
	var list []Checkpoint
	for _, line := range strings.Split(out, "\n") {
		parts := strings.SplitN(line, "\x1f", 3)
		if len(parts) != 3 {
			continue
		}
		secs, _ := strconv.ParseInt(parts[1], 10, 64)
		c := Checkpoint{ID: parts[0], When: time.Unix(secs, 0), Label: parts[2]}
		if n, err := s.git(ctx, "diff-tree", "--no-commit-id", "--name-only", "-r", c.ID); err == nil && n != "" {
			c.Files = len(strings.Split(n, "\n"))
		}
		list = append(list, c)
	}
	return list, nil
}

// Prune caps the history at keep snapshots by re-rooting it: the current tree
// is committed with no parent, the branch is moved onto it, and everything
// behind becomes unreachable and is collected.
//
// The older restore points are lost, which is what pruning means. The
// alternative — rewriting the kept range so it survives — needs filter-branch
// or a commit-by-commit replay, and gives every kept snapshot a new hash
// anyway, so the ids a session was holding would break either way. Given that,
// the simple version is the honest one.
func (s *Store) Prune(ctx context.Context, keep int) error {
	if keep < 1 {
		return nil
	}
	head, err := s.Head(ctx)
	if err != nil || head == "" {
		return err
	}
	out, err := s.git(ctx, "rev-list", "--count", ref)
	if err != nil {
		return err
	}
	count, err := strconv.Atoi(out)
	if err != nil || count < keep {
		return nil
	}

	if _, err := s.git(ctx, "add", "-A"); err != nil {
		return err
	}
	tree, err := s.git(ctx, "write-tree")
	if err != nil {
		return err
	}
	root, err := s.git(ctx, "commit-tree", tree, "-m", "checkpoint history trimmed")
	if err != nil {
		return err
	}
	if _, err := s.git(ctx, "update-ref", ref, root); err != nil {
		return err
	}
	// Best effort: the history is already correct at this point, and failing a
	// snapshot because housekeeping could not run would be the wrong trade.
	_, _ = s.git(ctx, "reflog", "expire", "--expire=now", "--all")
	_, _ = s.git(ctx, "gc", "--prune=now", "--quiet")
	return nil
}

// must verifies an id names a commit in the shadow repo, so a stale id from a
// previous session reports itself instead of producing a confusing git error.
func (s *Store) must(ctx context.Context, id string) error {
	if id == "" {
		return ErrNotFound
	}
	if _, err := s.git(ctx, "rev-parse", "--verify", "--quiet", id+"^{commit}"); err != nil {
		return ErrNotFound
	}
	return nil
}

// oneLine reduces a label to a single trimmed line, since it becomes a commit
// subject.
func oneLine(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return "checkpoint"
	}
	if i := strings.IndexAny(s, "\r\n"); i >= 0 {
		s = strings.TrimSpace(s[:i])
	}
	const max = 72
	if len(s) > max {
		s = s[:max] + "…"
	}
	if s == "" {
		return "checkpoint"
	}
	return s
}
