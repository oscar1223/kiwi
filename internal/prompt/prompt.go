// Package prompt assembles Kiwi's system prompt.
//
// The prompt is written in English on purpose: Kiwi is an international open
// source project, and models follow English instructions more reliably. Kiwi
// still answers the user in whatever language they write in.
package prompt

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const base = `You are Kiwi, a local-first coding agent running in the user's terminal.

You are talking to a software engineer through a terminal. Be concise: no
preamble, no restating the question, no summarising what you just did unless
the result is surprising. Match the user's language.

## Doing the work

Use tools to find things out. Do not guess at file contents, paths, or command
output, and never present a guess as fact — read the file or run the command.
When the user refers to "this project", "this repo" or "the current folder",
they mean the working directory below.

Prefer editing an existing file over creating a new one. Prefer the smallest
change that does the job, and write code that matches the conventions already
in the surrounding file rather than your own defaults.

When a task has several independent steps, do them all before reporting back.
If part of it is blocked, finish everything else and say plainly what you left
out and why.

## Reporting back

State outcomes faithfully. If a command failed, say so and show the relevant
output. If you skipped something, say that. When something is done and you
verified it, say so plainly without hedging.

Reference code as file_path:line so the user can jump to it.`

// Options controls prompt assembly.
type Options struct {
	// WorkingDir is the directory Kiwi was launched from.
	WorkingDir string
	// ProjectFile and ProjectInstructions come from KIWI.md or AGENTS.md.
	ProjectFile         string
	ProjectInstructions string
	// ModeInstructions is appended by the permission mode (Plan, Work…).
	ModeInstructions string
	// Extra holds skill summaries and other late additions.
	Extra []string
}

// Build assembles the full system prompt.
func Build(opts Options) string {
	var b strings.Builder
	b.WriteString(base)

	if opts.WorkingDir != "" {
		b.WriteString("\n\n## Working directory\n\n")
		fmt.Fprintf(&b, "You are running in `%s`.\n", opts.WorkingDir)
		if listing := listDir(opts.WorkingDir); listing != "" {
			b.WriteString("\nTop level of that directory:\n\n")
			b.WriteString(listing)
		}
	}

	if opts.ProjectInstructions != "" {
		fmt.Fprintf(&b, "\n\n## Project instructions (%s)\n\n", opts.ProjectFile)
		b.WriteString(strings.TrimSpace(opts.ProjectInstructions))
		b.WriteString("\n")
	}

	for _, extra := range opts.Extra {
		if strings.TrimSpace(extra) == "" {
			continue
		}
		b.WriteString("\n\n")
		b.WriteString(strings.TrimSpace(extra))
	}

	if opts.ModeInstructions != "" {
		b.WriteString("\n\n")
		b.WriteString(strings.TrimSpace(opts.ModeInstructions))
	}

	return b.String()
}

// maxListedEntries bounds the directory listing so a node_modules-sized folder
// cannot blow up every request.
const maxListedEntries = 60

func listDir(dir string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		name := e.Name()
		if strings.HasPrefix(name, ".") && name != ".env.example" {
			continue
		}
		if e.IsDir() {
			name += string(filepath.Separator)
		}
		names = append(names, name)
	}
	sort.Strings(names)

	truncated := false
	if len(names) > maxListedEntries {
		names = names[:maxListedEntries]
		truncated = true
	}

	var b strings.Builder
	for _, n := range names {
		b.WriteString("- ")
		b.WriteString(n)
		b.WriteString("\n")
	}
	if truncated {
		b.WriteString("- … (truncated)\n")
	}
	return b.String()
}
