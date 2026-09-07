package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Search limits.
//
// Every one of these exists to protect the context window rather than the
// disk: an unbounded search across a dependency tree returns more text than a
// turn can hold, and the turn is lost to it. Truncating loudly — the model is
// told the result was cut — is better than a correct answer nobody can use.
const (
	maxGlobResults   = 200
	maxGrepMatches   = 200
	maxGrepLineBytes = 512
	maxListEntries   = 400
	// maxWalkEntries bounds the walk itself, so a search rooted somewhere
	// enormous by mistake still returns rather than grinding.
	maxWalkEntries = 50_000
	// maxGrepFileBytes skips files too large to be worth scanning line by
	// line; they are almost always generated or binary.
	maxGrepFileBytes = MaxFileBytes
)

// skipDirs are never descended into.
//
// Note this is a fixed list and not .gitignore. Honouring .gitignore properly
// means negations, nested files and precedence rules — a parser, not a
// lookup — and getting it half right would silently hide files the model was
// asked to find. A fixed list of famously heavy directories is honest about
// what it does.
var skipDirs = map[string]bool{
	".git": true, "node_modules": true, "vendor": true,
	".next": true, "dist": true, "build": true, "target": true,
	".venv": true, "venv": true, "__pycache__": true,
	".idea": true, ".cache": true, "coverage": true,
}

// walkFiles visits every file under root that is not inside a skipped
// directory, stopping once the walk has seen maxWalkEntries of them.
func walkFiles(root string, visit func(path string, d fs.DirEntry) error) error {
	seen := 0
	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			// An unreadable directory is not a reason to abandon the search.
			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			if path != root && skipDirs[d.Name()] {
				return filepath.SkipDir
			}
			return nil
		}
		seen++
		if seen > maxWalkEntries {
			return filepath.SkipAll
		}
		return visit(path, d)
	})
}

// globToRegexp compiles a glob into an anchored regexp.
//
// filepath.Match cannot do this on its own: it has no "**", and its "*" would
// happily cross a directory separator here. The distinction is the whole point
// of the pattern language — "*.go" is the files in one directory, "**/*.go" is
// the tree.
func globToRegexp(pattern string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("^")
	for i := 0; i < len(pattern); i++ {
		switch c := pattern[i]; c {
		case '*':
			switch {
			case strings.HasPrefix(pattern[i:], "**/"):
				// Any number of leading directories, including none, so
				// "**/*.go" also matches "main.go" at the root.
				b.WriteString("(?:[^/]*/)*")
				i += 2
			case strings.HasPrefix(pattern[i:], "**"):
				b.WriteString(".*")
				i++
			default:
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	b.WriteString("$")
	return regexp.Compile(b.String())
}

// searchRoot resolves the optional path argument the search tools share.
func (f *FS) searchRoot(path string) (string, error) {
	if strings.TrimSpace(path) == "" {
		return f.WorkDir, nil
	}
	return f.resolve(path)
}

// --- glob ---

// Glob finds files by name pattern.
type Glob struct{ *FS }

func (Glob) Name() string { return "glob" }

func (Glob) Description() string {
	return "Find files by name pattern, newest first. Use ** to cross directories: " +
		"'**/*.go' for every Go file, '*.md' for the ones in a single directory. " +
		"Prefer this over shelling out to find. Returns paths only — use grep to " +
		"search inside files."
}

func (Glob) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"pattern": map[string]any{"type": "string", "description": "Glob pattern, e.g. '**/*_test.go'."},
			"path":    map[string]any{"type": "string", "description": "Directory to search under. Defaults to the working directory."},
		},
		"required": []string{"pattern"},
	}
}

func (t Glob) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Pattern string `json:"pattern"`
		Path    string `json:"path"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if strings.TrimSpace(in.Pattern) == "" {
		return "", fmt.Errorf("pattern is required")
	}
	re, err := globToRegexp(in.Pattern)
	if err != nil {
		return "", fmt.Errorf("bad pattern %q: %w", in.Pattern, err)
	}
	root, err := t.searchRoot(in.Path)
	if err != nil {
		return "", err
	}

	type hit struct {
		path string
		mod  int64
	}
	var hits []hit
	err = walkFiles(root, func(path string, d fs.DirEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		rel, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}
		if !re.MatchString(filepath.ToSlash(rel)) {
			return nil
		}
		var mod int64
		if info, err := d.Info(); err == nil {
			mod = info.ModTime().UnixNano()
		}
		hits = append(hits, hit{path, mod})
		return nil
	})
	if err != nil {
		return "", err
	}
	if len(hits) == 0 {
		return fmt.Sprintf("no files match %q under %s", in.Pattern, t.display(root)), nil
	}

	// Newest first: in a codebase, what was touched recently is usually what
	// the question is about.
	sort.Slice(hits, func(i, j int) bool { return hits[i].mod > hits[j].mod })

	var b strings.Builder
	shown := hits
	if len(shown) > maxGlobResults {
		shown = shown[:maxGlobResults]
	}
	for _, h := range shown {
		b.WriteString(t.display(h.path))
		b.WriteString("\n")
	}
	if len(hits) > len(shown) {
		fmt.Fprintf(&b, "… %d more matches, not shown. Narrow the pattern.\n", len(hits)-len(shown))
	}
	return b.String(), nil
}

// --- grep ---

// Grep searches inside files.
type Grep struct{ *FS }

func (Grep) Name() string { return "grep" }

func (Grep) Description() string {
	return "Search file contents with a regular expression (Go/RE2 syntax). " +
		"Returns file:line matches. Narrow it with 'glob' to a subset of files. " +
		"Prefer this over shelling out to grep or rg."
}

func (Grep) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"pattern":          map[string]any{"type": "string", "description": "Regular expression to search for."},
			"path":             map[string]any{"type": "string", "description": "Directory to search under. Defaults to the working directory."},
			"glob":             map[string]any{"type": "string", "description": "Only search files matching this glob, e.g. '**/*.go'."},
			"context":          map[string]any{"type": "integer", "description": "Lines of context around each match, up to 5."},
			"case_insensitive": map[string]any{"type": "boolean", "description": "Match regardless of case."},
		},
		"required": []string{"pattern"},
	}
}

func (t Grep) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Pattern         string `json:"pattern"`
		Path            string `json:"path"`
		Glob            string `json:"glob"`
		Context         int    `json:"context"`
		CaseInsensitive bool   `json:"case_insensitive"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if strings.TrimSpace(in.Pattern) == "" {
		return "", fmt.Errorf("pattern is required")
	}

	expr := in.Pattern
	if in.CaseInsensitive {
		expr = "(?i)" + expr
	}
	re, err := regexp.Compile(expr)
	if err != nil {
		return "", fmt.Errorf("bad pattern %q: %w", in.Pattern, err)
	}

	var fileRe *regexp.Regexp
	if strings.TrimSpace(in.Glob) != "" {
		if fileRe, err = globToRegexp(in.Glob); err != nil {
			return "", fmt.Errorf("bad glob %q: %w", in.Glob, err)
		}
	}
	root, err := t.searchRoot(in.Path)
	if err != nil {
		return "", err
	}

	around := in.Context
	if around < 0 {
		around = 0
	}
	if around > 5 {
		around = 5
	}

	var b strings.Builder
	matches, files, truncated := 0, 0, false

	err = walkFiles(root, func(path string, d fs.DirEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if truncated {
			return filepath.SkipAll
		}
		if fileRe != nil {
			rel, err := filepath.Rel(root, path)
			if err != nil || !fileRe.MatchString(filepath.ToSlash(rel)) {
				return nil
			}
		}
		if info, err := d.Info(); err != nil || info.Size() > maxGrepFileBytes {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil || isBinary(data) {
			return nil
		}

		lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
		header := false
		for i, line := range lines {
			if !re.MatchString(line) {
				continue
			}
			if !header {
				if files > 0 {
					b.WriteString("\n")
				}
				b.WriteString(t.display(path))
				b.WriteString("\n")
				header, files = true, files+1
			}
			lo, hi := max(0, i-around), min(len(lines)-1, i+around)
			for n := lo; n <= hi; n++ {
				marker := " "
				if n == i {
					marker = ":"
				}
				fmt.Fprintf(&b, "%6d%s %s\n", n+1, marker, clip(lines[n], maxGrepLineBytes))
			}
			matches++
			if matches >= maxGrepMatches {
				truncated = true
				return filepath.SkipAll
			}
		}
		return nil
	})
	if err != nil {
		return "", err
	}

	if matches == 0 {
		return fmt.Sprintf("no matches for %q under %s", in.Pattern, t.display(root)), nil
	}
	if truncated {
		fmt.Fprintf(&b, "\n… stopped at %d matches. Narrow the pattern or pass a glob.\n", maxGrepMatches)
	}
	return b.String(), nil
}

// --- ls ---

// List shows the contents of one directory.
type List struct{ *FS }

func (List) Name() string { return "ls" }

func (List) Description() string {
	return "List the entries of a directory, directories first and marked with a " +
		"trailing slash. Use it to get your bearings; use glob to search a tree."
}

func (List) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "Directory to list. Defaults to the working directory."},
		},
	}
}

func (t List) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Path string `json:"path"`
	}
	// An argumentless call is legitimate here, so an empty or absent object is
	// not an error.
	if len(input) > 0 {
		if err := json.Unmarshal(input, &in); err != nil {
			return "", err
		}
	}
	dir, err := t.searchRoot(in.Path)
	if err != nil {
		return "", err
	}
	info, err := os.Stat(dir)
	if err != nil {
		return "", err
	}
	if !info.IsDir() {
		return "", fmt.Errorf("%s is a file, not a directory", t.display(dir))
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	if len(entries) == 0 {
		return fmt.Sprintf("(%s is empty)", t.display(dir)), nil
	}

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].IsDir() != entries[j].IsDir() {
			return entries[i].IsDir()
		}
		return entries[i].Name() < entries[j].Name()
	})

	var b strings.Builder
	shown := entries
	if len(shown) > maxListEntries {
		shown = shown[:maxListEntries]
	}
	for _, e := range shown {
		if e.IsDir() {
			fmt.Fprintf(&b, "%s/\n", e.Name())
			continue
		}
		size := int64(-1)
		if fi, err := e.Info(); err == nil {
			size = fi.Size()
		}
		if size >= 0 {
			fmt.Fprintf(&b, "%s (%s)\n", e.Name(), humanBytes(size))
			continue
		}
		fmt.Fprintf(&b, "%s\n", e.Name())
	}
	if len(entries) > len(shown) {
		fmt.Fprintf(&b, "… %d more entries\n", len(entries)-len(shown))
	}
	return b.String(), nil
}

// isBinary reports whether data looks like something there is no point
// grepping. A NUL byte in the first block is the same heuristic grep itself
// uses.
func isBinary(data []byte) bool {
	if len(data) > 8000 {
		data = data[:8000]
	}
	return bytes.IndexByte(data, 0) >= 0
}

// clip shortens a matched line so one runaway minified file cannot fill the
// context window on its own.
func clip(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "…"
}

func humanBytes(n int64) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%dB", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1fK", float64(n)/1024)
	default:
		return fmt.Sprintf("%.1fM", float64(n)/(1024*1024))
	}
}
