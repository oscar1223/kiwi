package tui

import (
	"encoding/json"
	"fmt"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

func sprintf(format string, args ...any) string { return fmt.Sprintf(format, args...) }

// oneLine collapses text to a single line and caps its length, for the
// summaries shown next to tool calls.
func oneLine(s string, max int) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = strings.TrimSpace(s[:i]) + " …"
	}
	if runes := []rune(s); len(runes) > max {
		return string(runes[:max]) + "…"
	}
	return s
}

// toolSummary renders a tool call's arguments compactly: the value that
// matters most for that tool, rather than raw JSON.
func toolSummary(name string, input []byte) string {
	var args map[string]any
	if err := jsonUnmarshal(input, &args); err != nil {
		return oneLine(string(input), 100)
	}

	// Show the argument that identifies the action for each known tool.
	for _, key := range []string{"command", "path", "pattern", "query", "name"} {
		if v, ok := args[key].(string); ok && v != "" {
			return oneLine(v, 100)
		}
	}
	if len(args) == 0 {
		return ""
	}
	return oneLine(string(input), 100)
}

func jsonUnmarshal(data []byte, v any) error { return json.Unmarshal(data, v) }

// Layout widths.
//
// Everything the interface draws is measured against the terminal's current
// width. The frame owns the whole alt screen, so a line wider than the window
// costs the renderer a row it did not account for and shoves the rest of the
// frame off the bottom.
const (
	// fallbackWidth is used until the first WindowSizeMsg arrives, and by the
	// tests, which never get one.
	fallbackWidth = 80
	// gutter is the two columns every line of output is indented by, for the
	// "●" marker on the first line of an answer.
	gutter = 2
	// minWidth keeps the arithmetic below sane on a window too narrow to
	// really use.
	minWidth = 20
	// fallbackTailRows caps the streaming preview when the window height is
	// not known yet, and keeps it from taking the whole screen when it is.
	fallbackTailRows = 12
	// promptMarker sits in front of the input. Its width is also how far the
	// cursor has to be pushed right, so the two cannot drift apart.
	promptMarker = "› "
)

// width returns the terminal width to lay out against.
func (m *Model) termWidth() int {
	if m.width <= 0 {
		return fallbackWidth
	}
	return max(minWidth, m.width)
}

// textWidth is how wide a line of output may be: the window minus the gutter.
func (m *Model) textWidth() int { return textWidthFor(m.termWidth()) }

// textWidthFor is textWidth against a width the Model does not hold, so the
// transcript can lay itself out for any window rather than only the current
// one.
func textWidthFor(width int) int {
	return max(minWidth-gutter, width-gutter)
}

// fit truncates a rendered line to width so it can never spill onto a second
// row. Used for the rows of a list or a status line, where wrapping would
// change how many rows the frame occupies and truncating does not.
func fit(s string, width int) string {
	if width <= 0 || lipgloss.Width(s) <= width {
		return s
	}
	return ansi.Truncate(s, width, "…")
}

// wrapIndent wraps already-styled text to width and hangs the continuation
// lines under the first, so a wrapped paragraph lines up with itself instead
// of with the marker in front of it. hang pushes them in further still, to
// clear a marker that belongs to the text — a list bullet, a quote bar.
func wrapIndent(marker string, hang int, styled string, width int) string {
	lines := strings.Split(ansi.Wrap(styled, max(1, width-hang), ""), "\n")
	indent := strings.Repeat(" ", lipgloss.Width(marker)+hang)
	for i := range lines {
		if i == 0 {
			lines[i] = marker + lines[i]
			continue
		}
		lines[i] = indent + lines[i]
	}
	return strings.Join(lines, "\n")
}
