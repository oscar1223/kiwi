// Package memory keeps the few facts Kiwi should still know at the start of
// the next session.
//
// This is deliberately not a second conversation history. A session's messages
// live in internal/session and are compacted away as they age; memory is the
// opposite — a handful of durable lines (how the user likes to work, what this
// project is, a decision that outlived the conversation that produced it) that
// are cheap enough to sit in every system prompt.
//
// Two scopes, both stored under Kiwi's own config directory rather than inside
// the project: Global follows the user everywhere, Project is keyed by the
// working directory. Keeping project memory out of the repository is the same
// call internal/skills makes — notes Kiwi wrote to itself should never turn up
// in a user's `git status`, let alone in a commit. A project that *wants*
// committed, shared instructions already has KIWI.md/AGENTS.md, which Kiwi
// reads and never writes.
package memory

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/oscar1223/kiwi/internal/config"
)

// Scope is which of the two memory files a read or write is about.
type Scope string

const (
	// Global memory applies to this user in every project.
	Global Scope = "global"
	// Project memory applies only to the working directory it was written in.
	Project Scope = "project"
)

func (s Scope) Valid() bool { return s == Global || s == Project }

// MaxChars caps each scope's file. Memory is paid for on every single request,
// so it has to stay small enough that the user never has to think about the
// cost of having it switched on. Over the cap, the oldest lines are dropped
// (see Append) — a memory that grows without bound is a context leak, and
// silently unbounded is worse than visibly forgetful.
const MaxChars = 4000

// Store reads and writes the two memory files for one working directory.
// A zero WorkDir means no project scope is available: Global still works, and
// Project operations report that plainly instead of guessing at a directory.
type Store struct {
	WorkDir string
}

func New(workDir string) *Store { return &Store{WorkDir: workDir} }

// Dir is where memory files live.
func Dir() (string, error) {
	dir, err := config.Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "memory"), nil
}

// Path resolves a scope's file. It does not create anything.
func (s *Store) Path(scope Scope) (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	switch scope {
	case Global:
		return filepath.Join(dir, "global.md"), nil
	case Project:
		if s.WorkDir == "" {
			return "", fmt.Errorf("memory: no working directory, so there is no project memory")
		}
		return filepath.Join(dir, "projects", projectKey(s.WorkDir)+".md"), nil
	default:
		return "", fmt.Errorf("memory: unknown scope %q", scope)
	}
}

// projectKey turns an absolute path into one filename that a human can still
// recognise in a directory listing, with a short hash of the full path
// appended: sanitising alone is lossy ("/a/b" and "/a-b" collapse together),
// and two projects quietly sharing one memory file is exactly the failure
// nobody would think to look for.
func projectKey(dir string) string {
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	sum := sha256.Sum256([]byte(abs))

	var b strings.Builder
	for _, r := range abs {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '.', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	slug := strings.Trim(b.String(), "-")
	if len(slug) > 60 {
		slug = slug[len(slug)-60:]
	}
	return slug + "-" + hex.EncodeToString(sum[:4])
}

// Read returns a scope's contents, or "" when nothing has been remembered yet.
// A missing file is not an error: it is the normal state of a fresh install.
func (s *Store) Read(scope Scope) (string, error) {
	path, err := s.Path(scope)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

// Write replaces a scope's contents, trimming to MaxChars from the end (the
// oldest lines go first) so no path into the file can exceed the cap.
func (s *Store) Write(scope Scope, body string) (dropped int, err error) {
	path, err := s.Path(scope)
	if err != nil {
		return 0, err
	}
	body, dropped = capBody(body)

	if strings.TrimSpace(body) == "" {
		if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
			return dropped, err
		}
		return dropped, nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return dropped, err
	}
	return dropped, os.WriteFile(path, []byte(body+"\n"), 0o644)
}

// Append adds one fact as a bullet line. It reports how many old lines had to
// be dropped to stay under MaxChars, so the caller can say so out loud rather
// than letting the file quietly forget things.
func (s *Store) Append(scope Scope, fact string) (dropped int, err error) {
	fact = strings.TrimSpace(fact)
	if fact == "" {
		return 0, fmt.Errorf("memory: nothing to remember")
	}
	// One line per fact: memory is a list, and a multi-line entry makes the
	// oldest-first trimming below cut a fact in half.
	fact = strings.Join(strings.Fields(fact), " ")
	if !strings.HasPrefix(fact, "-") {
		fact = "- " + fact
	}

	existing, err := s.Read(scope)
	if err != nil {
		return 0, err
	}
	if existing == "" {
		return s.Write(scope, fact)
	}
	return s.Write(scope, existing+"\n"+fact)
}

// Clear forgets a whole scope.
func (s *Store) Clear(scope Scope) error {
	path, err := s.Path(scope)
	if err != nil {
		return err
	}
	if err := os.Remove(path); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

// capBody drops whole lines from the front until body fits in MaxChars.
func capBody(body string) (string, int) {
	body = strings.TrimSpace(body)
	if len(body) <= MaxChars {
		return body, 0
	}
	lines := strings.Split(body, "\n")
	dropped := 0
	for len(lines) > 1 && len(strings.Join(lines, "\n")) > MaxChars {
		lines = lines[1:]
		dropped++
	}
	out := strings.Join(lines, "\n")
	if len(out) > MaxChars {
		// A single line longer than the whole budget: keep its tail, which is
		// where a truncated sentence is least likely to invert its meaning.
		out = out[len(out)-MaxChars:]
	}
	return out, dropped
}

// Block renders the system-prompt fragment, or "" when nothing is remembered.
//
// The framing matters: these lines were written by the model itself in an
// earlier session, so they are presented as recollections that may have gone
// stale, not as instructions ranking alongside the user's own.
func (s *Store) Block() string {
	global, _ := s.Read(Global)
	project, _ := s.Read(Project)
	if global == "" && project == "" {
		return ""
	}

	var b strings.Builder
	b.WriteString("## Memory\n\n")
	b.WriteString("Notes you saved for yourself in earlier sessions. Treat them as background\n")
	b.WriteString("that may have gone stale — verify anything you are about to act on, and\n")
	b.WriteString("never let them override what the user asks for now. Use the remember tool\n")
	b.WriteString("when something durable comes up that you would want at the start of the\n")
	b.WriteString("next session; do not use it for what the code or git history already says.\n")
	if global != "" {
		b.WriteString("\n### About the user (all projects)\n\n")
		b.WriteString(global)
		b.WriteString("\n")
	}
	if project != "" {
		b.WriteString("\n### About this project\n\n")
		b.WriteString(project)
		b.WriteString("\n")
	}
	return b.String()
}
