package tui

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// Typing "@" used to mean typing the whole path from memory. The search that
// makes a picker possible already exists — glob walks the tree and skips the
// heavy directories — so what was missing was only the list under the input,
// which is the same component the "/" autocomplete already draws.

// maxFileSuggestions caps the list. More rows than this stop being a menu and
// start being a directory listing, which is what the user was trying to avoid.
const maxFileSuggestions = 8

// filePickScanLimit bounds the walk. A repository with a hundred thousand
// files must not make each keystroke pause: past this the scan stops and
// offers what it has, which is nearly always the right answer anyway since the
// best matches are shallow.
const filePickScanLimit = 20000

// skipDirs are never walked. The same list glob uses, and for the same reason:
// nobody is looking for a file inside node_modules by name.
var skipDirs = map[string]bool{
	".git": true, "node_modules": true, "vendor": true, "dist": true,
	"build": true, "target": true, ".next": true, ".venv": true,
	"__pycache__": true, ".idea": true, ".cache": true,
}

// mentionQuery returns the partial path being typed after an "@", and whether
// the picker applies at all.
//
// It only applies to the last "@" in the input, and only while nothing has
// been typed after it but path characters: once there is a space, the mention
// is finished and the user is writing prose again.
func mentionQuery(input string) (string, bool) {
	at := strings.LastIndex(input, "@")
	if at < 0 {
		return "", false
	}
	// The "@" has to start a word, or an email address in a sentence would
	// open a file picker.
	if at > 0 && !isSpace(input[at-1]) {
		return "", false
	}
	query := input[at+1:]
	if strings.ContainsAny(query, " \t\n") {
		return "", false
	}
	return query, true
}

func isSpace(b byte) bool { return b == ' ' || b == '\t' || b == '\n' }

// scanFiles walks the project once and returns every path worth offering.
//
// Separated from the filtering because the walk is the expensive half and the
// filtering happens on every keystroke. Walking per keystroke would put a
// full-tree traversal behind each character typed after an "@", which on a
// large repository is exactly the pause this feature exists to avoid.
func scanFiles(workDir string) []string {
	var all []string
	seen := 0

	// Errors are skipped rather than returned: a directory that cannot be read
	// is a reason to offer fewer files, not to offer none.
	_ = filepath.WalkDir(workDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if seen > filePickScanLimit {
			return filepath.SkipAll
		}
		name := d.Name()
		if d.IsDir() {
			if path != workDir && (skipDirs[name] || strings.HasPrefix(name, ".")) {
				return filepath.SkipDir
			}
			return nil
		}
		seen++

		if rel, err := filepath.Rel(workDir, path); err == nil {
			all = append(all, rel)
		}
		return nil
	})
	return all
}

// rankFiles filters and orders an already-scanned list for one query.
func rankFiles(all []string, query string) []string {
	lower := strings.ToLower(query)
	var matches []string
	for _, rel := range all {
		if query == "" || subsequence(strings.ToLower(rel), lower) {
			matches = append(matches, rel)
		}
	}
	sort.SliceStable(matches, func(i, j int) bool {
		return fileRank(matches[i], query) < fileRank(matches[j], query)
	})
	if len(matches) > maxFileSuggestions {
		matches = matches[:maxFileSuggestions]
	}
	return matches
}

// subsequence reports whether every character of query appears in s in order.
//
// Not a substring match: people type "tumod" for internal/tui/model.go, and a
// picker that only finds literal substrings sends them back to typing the
// whole path, which is what it was supposed to replace.
func subsequence(s, query string) bool {
	if query == "" {
		return true
	}
	i := 0
	for j := 0; j < len(s) && i < len(query); j++ {
		if s[j] == query[i] {
			i++
		}
	}
	return i == len(query)
}

// fileRank orders matches: an exact hit on the file's own name first, then a
// prefix of it, then shallow paths before deep ones.
//
// Depth is the tie-breaker that matters. The file someone means is almost
// never the one buried six directories down with a similar name.
func fileRank(path, query string) int {
	if query == "" {
		return strings.Count(path, string(os.PathSeparator))
	}
	base := strings.ToLower(filepath.Base(path))
	q := strings.ToLower(query)
	depth := strings.Count(path, string(os.PathSeparator))

	switch {
	case base == q:
		return depth
	case strings.HasPrefix(base, q):
		return 100 + depth
	case strings.Contains(base, q):
		return 200 + depth
	case strings.Contains(strings.ToLower(path), q):
		return 300 + depth
	}
	return 400 + depth
}

// --- wiring ---

// fileSuggestionsFor returns the picker rows for the current input, or nil
// when the picker does not apply.
func (m *Model) fileSuggestionsFor() []string {
	if m.pending != nil || m.activePick != nil || m.activeText != nil || m.activeQuestion != nil {
		return nil
	}
	if m.promptHistory.search != nil {
		return nil
	}
	query, ok := mentionQuery(m.input.Value())
	if !ok {
		return nil
	}
	return rankFiles(m.projectFiles(), query)
}

// projectFiles is the scanned file list, refreshed when it goes stale.
//
// A short life rather than none: files appear and disappear while Kiwi is
// running — the agent itself creates them — and a picker that cannot see a
// file written two minutes ago is one people stop trusting.
func (m *Model) projectFiles() []string {
	if m.fileIndex != nil && time.Since(m.fileIndexAt) < fileIndexTTL {
		return m.fileIndex
	}
	m.fileIndex = scanFiles(m.opts.WorkDir)
	m.fileIndexAt = time.Now()
	return m.fileIndex
}

// fileIndexTTL is how long a scan is reused.
const fileIndexTTL = 30 * time.Second

// completeMention replaces the partial path after the "@" with a full one.
func (m *Model) completeMention(path string) {
	input := m.input.Value()
	at := strings.LastIndex(input, "@")
	if at < 0 {
		return
	}
	m.setInput(input[:at+1] + path + " ")
}

// renderFileSuggestions draws the picker under the input.
func renderFileSuggestions(paths []string, index, width int) string {
	if index < 0 || index >= len(paths) {
		index = 0
	}
	var b strings.Builder
	for i, p := range paths {
		if i == index {
			b.WriteString(fit(styleKiwi.Render("  ▸ "+p), width))
		} else {
			b.WriteString(fit(styleDim.Render("    "+p), width))
		}
		if i < len(paths)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// mentionIsComplete reports whether what has been typed after the "@" already
// names a file that exists.
//
// This is what lets enter submit rather than complete. Without it, typing a
// path in full and pressing enter would replace it with whichever suggestion
// was highlighted, which is the one thing a picker must never do.
func (m *Model) mentionIsComplete() bool {
	query, ok := mentionQuery(m.input.Value())
	if !ok || query == "" {
		return false
	}
	info, err := os.Stat(filepath.Join(m.opts.WorkDir, query))
	return err == nil && !info.IsDir()
}
