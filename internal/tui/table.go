package tui

import (
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// Markdown tables.
//
// A table cannot be drawn a line at a time: how wide each column is depends
// on every row. So its rows are held back as they stream in (see record) and
// filed as one entry when the table ends, which renders them aligned against
// whatever width the window has. Drawn raw instead, the pipes never line up
// and a row too long for the window wraps into the middle of the next one.

// A column is squeezed no narrower than its longest word, so words are not
// broken in half, and never below minColumn or than maxWordColumn needs. A
// table that still does not fit lists each row as field: value pairs instead.
const (
	minColumn     = 8
	maxWordColumn = 16
)

// isTableRow reports whether a line is a row of a markdown table.
func isTableRow(line string) bool {
	t := strings.TrimSpace(line)
	return len(t) >= 2 && strings.HasPrefix(t, "|") && strings.HasSuffix(t, "|")
}

// splitRow breaks a table row into its cells. An escaped pipe stays text.
func splitRow(line string) []string {
	t := strings.TrimSpace(line)
	t = strings.TrimSuffix(strings.TrimPrefix(t, "|"), "|")
	t = strings.ReplaceAll(t, `\|`, "\x00")
	cells := strings.Split(t, "|")
	for i, c := range cells {
		cells[i] = strings.ReplaceAll(strings.TrimSpace(c), "\x00", "|")
	}
	return cells
}

// isDelimiterRow reports whether a row is the |---|:--:| line under a header.
func isDelimiterRow(cells []string) bool {
	for _, c := range cells {
		c = strings.Trim(c, ":")
		if c == "" || strings.Trim(c, "-") != "" {
			return false
		}
	}
	return len(cells) > 0
}

// renderTable lays out the rows of a table for a terminal of the given width.
func renderTable(lines []string, prefix string, width int) []string {
	var header []string
	var body [][]string
	for i, l := range lines {
		cells := splitRow(l)
		if i == 1 && isDelimiterRow(cells) {
			header, body = body[0], nil
			continue
		}
		body = append(body, cells)
	}
	if header == nil && len(body) == 1 && isDelimiterRow(body[0]) {
		body = nil
	}

	cols := len(header)
	for _, r := range body {
		cols = max(cols, len(r))
	}
	if cols == 0 {
		return nil
	}

	style := func(cells []string, base lipgloss.Style) []string {
		out := make([]string, cols)
		for i := range out {
			if i < len(cells) {
				out[i] = renderInline(cells[i], base)
			}
		}
		return out
	}
	var head []string
	if header != nil {
		head = style(header, styleHeading)
	}
	rows := make([][]string, len(body))
	for i, r := range body {
		rows[i] = style(r, styleKiwi)
	}

	indent := strings.Repeat(" ", lipgloss.Width(prefix))
	avail := width - lipgloss.Width(prefix)

	widths := make([]int, cols)
	floors := make([]int, cols)
	for _, r := range append([][]string{head}, rows...) {
		for i, c := range r {
			widths[i] = max(widths[i], lipgloss.Width(c))
			for _, word := range strings.Fields(ansi.Strip(c)) {
				floors[i] = max(floors[i], min(ansi.StringWidth(word), maxWordColumn))
			}
		}
	}
	for i := range floors {
		floors[i] = min(widths[i], max(floors[i], minColumn))
	}
	// Squeeze the widest column a cell at a time until the table fits. Cells
	// in a squeezed column wrap within it.
	sep := styleDim.Render(" │ ")
	gaps := 3 * (cols - 1)
	for sum(widths)+gaps > avail {
		i := widestAbove(widths, floors)
		if i < 0 {
			return renderTableAsList(head, rows, prefix, indent, avail)
		}
		widths[i]--
	}

	var out []string
	first := true
	emit := func(line string) {
		if first {
			out = append(out, prefix+line)
			first = false
			return
		}
		out = append(out, indent+line)
	}
	drawRow := func(cells []string) {
		wrapped := make([][]string, cols)
		height := 1
		for i, c := range cells {
			wrapped[i] = strings.Split(ansi.Wrap(c, widths[i], ""), "\n")
			height = max(height, len(wrapped[i]))
		}
		for h := 0; h < height; h++ {
			parts := make([]string, cols)
			for i := range parts {
				var cell string
				if h < len(wrapped[i]) {
					cell = wrapped[i][h]
				}
				parts[i] = cell + strings.Repeat(" ", max(0, widths[i]-lipgloss.Width(cell)))
			}
			emit(strings.TrimRight(strings.Join(parts, sep), " "))
		}
	}

	if head != nil {
		drawRow(head)
		rule := make([]string, cols)
		for i, w := range widths {
			rule[i] = strings.Repeat("─", w)
		}
		emit(styleDim.Render(strings.Join(rule, "─┼─")))
	}
	for _, r := range rows {
		drawRow(r)
	}
	return out
}

// renderTableAsList is the fallback for a table with more columns than the
// window has room for: each row becomes a block of "header: value" lines,
// which stays readable at any width.
func renderTableAsList(head []string, rows [][]string, prefix, indent string, avail int) []string {
	var out []string
	for n, r := range rows {
		if n > 0 {
			out = append(out, "")
		}
		for i, c := range r {
			if c == "" {
				continue
			}
			label := ""
			if i < len(head) && head[i] != "" {
				label = head[i] + styleDim.Render(": ")
			}
			line := label + c
			for _, w := range strings.Split(ansi.Wrap(line, max(1, avail), ""), "\n") {
				out = append(out, indent+w)
			}
		}
	}
	if len(out) > 0 {
		out[0] = prefix + strings.TrimPrefix(out[0], indent)
	}
	return out
}

func sum(xs []int) int {
	n := 0
	for _, x := range xs {
		n += x
	}
	return n
}

// widestAbove is the widest column that can still give up a cell, or -1.
func widestAbove(widths, floors []int) int {
	best := -1
	for i, w := range widths {
		if w > floors[i] && (best < 0 || w > widths[best]) {
			best = i
		}
	}
	return best
}
