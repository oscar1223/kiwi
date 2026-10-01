package tui

import (
	"strings"
	"unicode/utf8"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/atotto/clipboard"
	"github.com/charmbracelet/x/ansi"
)

// Selecting text with the mouse.
//
// The mouse is captured so the wheel scrolls the transcript (see View), and a
// captured mouse is one the terminal no longer selects with: a drag arrives
// here as events instead of highlighting anything. So the selection is drawn
// and copied by Kiwi itself — press, drag, release, and what was covered is on
// the clipboard.

// selPoint is one cell of the transcript: a row of transcriptRows and a column
// in it. Rows are counted from the top of the transcript rather than of the
// window, so a selection stays on its text while the view scrolls under it.
type selPoint struct{ row, col int }

type selection struct {
	anchor, head selPoint
	// dragging is true from the press until the release.
	dragging bool
	// copied is how many characters the release put on the clipboard, for the
	// status line.
	copied int
}

// styleSelection has no colour of its own: reverse video is legible on every
// theme and over text of any colour.
var styleSelection = lipgloss.NewStyle().Reverse(true)

// writeClipboard is the system clipboard, replaceable so the tests do not
// overwrite whatever the person running them had copied.
var writeClipboard = clipboard.WriteAll

// bounds orders the two ends, since a drag can run backwards.
func (s *selection) bounds() (from, to selPoint) {
	from, to = s.anchor, s.head
	if to.row < from.row || (to.row == from.row && to.col < from.col) {
		from, to = to, from
	}
	return from, to
}

// span is the columns of one row the selection covers, as [left, right).
func (s *selection) span(row, width int) (left, right int, ok bool) {
	from, to := s.bounds()
	if from == to || row < from.row || row > to.row {
		return 0, 0, false
	}
	left, right = 0, width
	if row == from.row {
		left = from.col
	}
	if row == to.row {
		// The cell under the pointer is part of the selection.
		right = min(width, to.col+1)
	}
	return left, right, left < right
}

// highlight draws the selected part of one rendered row.
func (s *selection) highlight(row string, index int) string {
	width := ansi.StringWidth(row)
	left, right, ok := s.span(index, width)
	if !ok {
		return row
	}
	// The selected cells lose their own styling: reverse video over a
	// coloured run would otherwise change colour half-way along.
	out := ansi.Cut(row, 0, left) + styleSelection.Render(ansi.Strip(ansi.Cut(row, left, right)))
	if right < width {
		out += ansi.Cut(row, right, width)
	}
	return out
}

// text is what the selection covers, as plain text.
//
// The indentation every selected line shares is dropped: it is the gutter the
// transcript is drawn behind, not part of what was written.
func (s *selection) text(rows []string) string {
	from, to := s.bounds()
	var lines []string
	for i := max(0, from.row); i <= to.row && i < len(rows); i++ {
		line := ""
		if left, right, ok := s.span(i, ansi.StringWidth(rows[i])); ok {
			line = strings.TrimRight(ansi.Strip(ansi.Cut(rows[i], left, right)), " ")
		}
		lines = append(lines, line)
	}

	indent := -1
	for _, l := range lines {
		if l == "" {
			continue
		}
		if n := len(l) - len(strings.TrimLeft(l, " ")); indent < 0 || n < indent {
			indent = n
		}
	}
	if indent > 0 && len(lines) > 1 {
		for i, l := range lines {
			if l != "" {
				lines[i] = l[indent:]
			}
		}
	}
	return strings.Trim(strings.Join(lines, "\n"), "\n")
}

// cellAt maps a position in the window to a cell of the transcript. The second
// result is false below the transcript, where there is nothing to select.
func (m *Model) cellAt(x, y int) (selPoint, bool) {
	height := m.viewportHeight()
	if y < 0 || y >= height {
		return selPoint{}, false
	}
	total := len(m.transcriptRows(m.termWidth()))
	return selPoint{row: m.scrollOffset(total, height) + y, col: max(0, x)}, true
}

// onMouseClick starts a selection. Any click drops the previous one, so a
// click on its own is also how a highlight is dismissed.
func (m *Model) onMouseClick(msg tea.MouseClickMsg) {
	m.sel = nil
	if msg.Button != tea.MouseLeft {
		return
	}
	if p, ok := m.cellAt(msg.X, msg.Y); ok {
		m.sel = &selection{anchor: p, head: p, dragging: true}
	}
}

// onMouseMotion extends the selection to the pointer. Dragging past either
// edge of the transcript scrolls it, so a selection can be longer than the
// window.
func (m *Model) onMouseMotion(msg tea.MouseMotionMsg) {
	if m.sel == nil || !m.sel.dragging {
		return
	}
	height := m.viewportHeight()
	y := msg.Y
	switch {
	case y <= 0:
		m.scrollBy(-1)
		y = 0
	case y >= height:
		m.scrollBy(1)
		y = height - 1
	}
	if p, ok := m.cellAt(msg.X, y); ok {
		m.sel.head = p
	}
}

// onMouseRelease ends the drag and copies what it covered.
//
// The text goes out twice. OSC 52 asks the terminal to set the clipboard,
// which is the only route that works over SSH; the system clipboard is written
// directly as well, because Terminal.app ignores OSC 52.
func (m *Model) onMouseRelease() tea.Cmd {
	if m.sel == nil || !m.sel.dragging {
		return nil
	}
	m.sel.dragging = false
	text := m.sel.text(m.transcriptRows(m.termWidth()))
	if text == "" {
		m.sel = nil
		return nil
	}
	m.sel.copied = utf8.RuneCountInString(text)
	_ = writeClipboard(text)
	return tea.SetClipboard(text)
}
