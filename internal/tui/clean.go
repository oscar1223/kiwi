package tui

import (
	"strings"

	"github.com/charmbracelet/x/ansi"
)

// tabWidth is how far apart the tab stops sit when a tab is expanded.
const tabWidth = 4

// cleanLine makes one line of text from the model or a command safe to lay
// out: what the frame measures has to be what the terminal draws.
//
//   - A carriage return sends the cursor back to column 0, so a progress bar
//     or a CRLF line would overwrite what is already on the row. Only what a
//     terminal would finally show is kept: the text after the last \r.
//   - A tab is drawn to the terminal's next tab stop but measured as zero
//     columns, so columns stop lining up and rows overflow. It is expanded to
//     spaces here, against stops the frame knows about.
//   - Escape sequences and other control characters are dropped: cut in half
//     by a truncation they garble the rest of the row.
func cleanLine(s string) string {
	s = strings.TrimRight(s, "\r")
	if i := strings.LastIndexByte(s, '\r'); i >= 0 {
		s = s[i+1:]
	}
	if strings.ContainsRune(s, '\x1b') {
		s = ansi.Strip(s)
	}
	if !needsCleaning(s) {
		return s
	}

	var b strings.Builder
	col := 0
	for _, r := range s {
		switch {
		case r == '\t':
			n := tabWidth - col%tabWidth
			b.WriteString(strings.Repeat(" ", n))
			col += n
		case r < 0x20 || r == 0x7f:
			// Other control characters draw nothing useful.
		default:
			b.WriteRune(r)
			col += ansi.StringWidth(string(r))
		}
	}
	return b.String()
}

func needsCleaning(s string) bool {
	for i := 0; i < len(s); i++ {
		if c := s[i]; c < 0x20 || c == 0x7f {
			return true
		}
	}
	return false
}

// cleanText is cleanLine over every line of s.
func cleanText(s string) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		lines[i] = cleanLine(l)
	}
	return strings.Join(lines, "\n")
}
