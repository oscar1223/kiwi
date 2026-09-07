package tui

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/config"
)

// Prompt history: the arrow keys recall what you sent before, and ctrl+r
// searches it. Every comparable tool has this and Kiwi had nothing — the up
// arrow scrolled the transcript and there was no way at all to resend the last
// thing you typed.
//
// It is kept per working directory rather than globally. The prompts for one
// project say nothing useful while working on another, and a shared list is
// mostly other projects' noise by the time it is long enough to be worth
// searching.

// maxHistory is how many prompts one directory keeps.
const maxHistory = 500

// maxHistoryEntryBytes drops anything longer from the file. A pasted stack
// trace is not something anyone recalls with the up arrow, and it would push
// out the entries that are.
const maxHistoryEntryBytes = 4096

// promptHistory is the recalled list plus where in it the user currently is.
type promptHistory struct {
	// entries is oldest-first, the order they were sent in.
	entries []string
	// pos is the index being shown, or len(entries) when the user is back on
	// their own draft rather than in the history.
	pos int
	// draft holds what was typed before recall started, so leaving the
	// history gives it back rather than losing it.
	draft string
	// path is where the list is persisted; empty disables saving.
	path string

	// search is the ctrl+r state: nil when not searching.
	search *historySearch
}

type historySearch struct {
	query string
	// match is the index of the entry currently shown, or -1 for no match.
	match int
}

// loadHistory reads the saved prompts for a working directory.
func loadHistory(workDir string) promptHistory {
	h := promptHistory{}
	root, err := config.DataDir()
	if err != nil {
		return h
	}
	abs, err := filepath.Abs(workDir)
	if err != nil {
		abs = workDir
	}
	// Keyed by a hash of the path for the same reason the checkpoints are: a
	// path used verbatim as a filename needs escaping on every platform.
	sum := sha256.Sum256([]byte(abs))
	h.path = filepath.Join(root, "history", hex.EncodeToString(sum[:])[:16])

	data, err := os.ReadFile(h.path)
	if err != nil {
		return h
	}
	for _, line := range strings.Split(string(data), "\n") {
		if entry := decodeEntry(line); entry != "" {
			h.entries = append(h.entries, entry)
		}
	}
	h.pos = len(h.entries)
	return h
}

// record adds a prompt and persists the list.
func (h *promptHistory) record(text string) {
	text = strings.TrimSpace(text)
	h.reset()
	if text == "" || len(text) > maxHistoryEntryBytes {
		return
	}
	// Re-sending the same thing twice should not take two presses of the up
	// arrow to get back past.
	if n := len(h.entries); n > 0 && h.entries[n-1] == text {
		h.pos = len(h.entries)
		return
	}
	h.entries = append(h.entries, text)
	if len(h.entries) > maxHistory {
		h.entries = h.entries[len(h.entries)-maxHistory:]
	}
	h.pos = len(h.entries)
	h.save()
}

func (h *promptHistory) save() {
	if h.path == "" {
		return
	}
	if err := os.MkdirAll(filepath.Dir(h.path), 0o700); err != nil {
		return
	}
	var b strings.Builder
	for _, e := range h.entries {
		b.WriteString(encodeEntry(e))
		b.WriteString("\n")
	}
	// Best effort. Losing the history is a papercut; failing a turn over it
	// would not be.
	_ = os.WriteFile(h.path, []byte(b.String()), 0o600)
}

// A multi-line prompt has to survive a line-based file, so newlines are
// escaped rather than the format being made cleverer.
func encodeEntry(s string) string {
	return strings.NewReplacer("\\", "\\\\", "\n", "\\n").Replace(s)
}

func decodeEntry(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] != '\\' || i+1 >= len(s) {
			b.WriteByte(s[i])
			continue
		}
		i++
		switch s[i] {
		case 'n':
			b.WriteByte('\n')
		case '\\':
			b.WriteByte('\\')
		default:
			b.WriteByte(s[i])
		}
	}
	return b.String()
}

// reset leaves history navigation and forgets the stashed draft.
func (h *promptHistory) reset() {
	h.pos = len(h.entries)
	h.draft = ""
	h.search = nil
}

// recall steps through the history and returns what the input should show.
// ok is false when there is nowhere further to go.
func (h *promptHistory) recall(back bool, current string) (string, bool) {
	if len(h.entries) == 0 {
		return "", false
	}
	if back {
		if h.pos == len(h.entries) {
			// Entering the history: keep whatever was being typed, so
			// stepping back out returns it instead of eating it.
			h.draft = current
		}
		if h.pos == 0 {
			return "", false
		}
		h.pos--
		return h.entries[h.pos], true
	}

	if h.pos >= len(h.entries) {
		return "", false
	}
	h.pos++
	if h.pos == len(h.entries) {
		return h.draft, true
	}
	return h.entries[h.pos], true
}

// find runs the ctrl+r search: the most recent entry containing the query.
func (h *promptHistory) find(query string) (string, bool) {
	for i := len(h.entries) - 1; i >= 0; i-- {
		if strings.Contains(strings.ToLower(h.entries[i]), strings.ToLower(query)) {
			return h.entries[i], true
		}
	}
	return "", false
}

// --- key handling ---

// wantsHistory decides whether an arrow key should recall a prompt rather than
// scroll the transcript or move within the input.
//
// The rule follows the one already used for the transcript: the input gets the
// key when it can actually use it. A multi-line draft navigates its own text;
// a single-line input at the top or bottom hands the key on.
func (m *Model) wantsHistory(key string) bool {
	if m.busy || m.flowBusy || len(m.promptHistory.entries) == 0 {
		return false
	}
	if m.inputWantsArrow(key) {
		return false
	}
	if key == "up" {
		// Going back is available whenever the input is not a multi-line
		// draft being navigated: from an empty prompt, or from something
		// half-typed, which is then stashed.
		return m.promptHistory.pos > 0
	}
	// Forward only makes sense once the history has been entered, or the down
	// arrow would eat the scroll on every press.
	return m.promptHistory.pos < len(m.promptHistory.entries)
}

// recallPrompt puts a remembered prompt into the input.
func (m *Model) recallPrompt(back bool) tea.Cmd {
	text, ok := m.promptHistory.recall(back, m.input.Value())
	if !ok {
		return nil
	}
	m.setInput(text)
	return nil
}

// setInput replaces the input's contents and puts the cursor at the end, which
// is where anyone editing a recalled prompt wants to start.
func (m *Model) setInput(text string) {
	m.input.SetValue(text)
	m.input.MoveToEnd()
	m.resize()
}

// startSearch opens reverse search.
func (m *Model) startSearch() tea.Cmd {
	if len(m.promptHistory.entries) == 0 {
		return m.println(styleDim.Render("  no history yet"))
	}
	m.promptHistory.draft = m.input.Value()
	m.promptHistory.search = &historySearch{match: -1}
	return nil
}

// onSearchKey handles typing while reverse search is open. It reports whether
// it consumed the key.
func (m *Model) onSearchKey(msg tea.KeyPressMsg, key string) (tea.Cmd, bool) {
	s := m.promptHistory.search
	if s == nil {
		return nil, false
	}
	switch key {
	case "esc", "ctrl+c":
		// Cancelling gives back what was being typed when the search opened.
		m.setInput(m.promptHistory.draft)
		m.promptHistory.reset()
		return nil, true
	case "enter":
		// Accepting leaves the found prompt in the input rather than sending
		// it: a recalled command is usually recalled to be edited.
		m.promptHistory.search = nil
		return nil, true
	case "backspace":
		if s.query != "" {
			s.query = s.query[:len(s.query)-1]
		}
	case "ctrl+r":
		// Kept simple on purpose: repeating ctrl+r cycling through older
		// matches is the one part of this people rarely reach for, and it
		// would need its own cursor to do correctly.
		return nil, true
	default:
		if len(msg.String()) != 1 {
			return nil, true
		}
		s.query += msg.String()
	}

	if match, ok := m.promptHistory.find(s.query); ok {
		m.setInput(match)
	}
	return nil, true
}

// searchLine renders the reverse-search prompt shown under the input.
func (m *Model) searchLine() string {
	s := m.promptHistory.search
	if s == nil {
		return ""
	}
	line := styleDim.Render("  search: ") + styleUser.Render(s.query)
	if s.query != "" {
		if _, ok := m.promptHistory.find(s.query); !ok {
			line += styleWarn.Render("  (no match)")
		}
	}
	return line + styleDim.Render("   enter: keep · esc: cancel")
}
