package tui

import (
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/charmbracelet/x/ansi"
)

// entryKind says how an entry is turned back into rows at render time.
type entryKind int

const (
	// entryStyled is text that was already styled when it was recorded: the
	// banner, a tool call, a notice. Rendering only has to wrap it.
	entryStyled entryKind = iota
	// entryProse is one line of assistant prose, stored raw so its markdown
	// can be restyled and rewrapped at whatever width the window currently
	// has.
	entryProse
	// entryCode is one line inside a fence, reproduced as written.
	entryCode
	// entryFence is the ``` line that opens or closes a code block.
	entryFence
)

// entry is one logical line of the session, stored unwrapped.
//
// prefix is the marker the line is printed behind — "● " on the first line of
// an answer, two spaces on the rest — resolved when the entry is recorded
// because it depends on turn state the renderer no longer has.
type entry struct {
	kind   entryKind
	text   string
	prefix string
}

// transcript holds everything the session has printed, unwrapped.
//
// Storing logical lines rather than wrapped ones is what makes a resize
// correct: the whole transcript is rewrapped to the new width, instead of
// keeping the breaks it was given at the old one. Wrapping at print time —
// which is what Kiwi did while it drew into the terminal's own scrollback —
// cannot survive a resize in either direction, because once a line has been
// printed it belongs to the terminal and not to us.
type transcript struct {
	entries []entry

	// Rendering the whole transcript costs a rewrap of every entry, and the
	// view is rebuilt on every streamed delta, so the last result is kept and
	// reused until the width changes or a new entry arrives.
	cacheWidth int
	cacheRows  []string
	cacheValid bool
}

// add records one entry. The fence state and the prefix are already resolved
// by the caller: they depend on the order lines arrive in, which is exactly
// what a re-render must not recompute.
func (t *transcript) add(e entry) {
	t.entries = append(t.entries, e)
	t.cacheValid = false
}

// addStyled records text that is already styled, as printed by println.
func (t *transcript) addStyled(s string) { t.add(entry{kind: entryStyled, text: s}) }

// reset empties the transcript, for /clear.
func (t *transcript) reset() {
	t.entries = nil
	t.cacheValid = false
}

// render lays the whole transcript out for a terminal of the given width.
func (t *transcript) render(width int) []string {
	if t.cacheValid && t.cacheWidth == width {
		return t.cacheRows
	}
	rows := make([]string, 0, len(t.entries))
	for _, e := range t.entries {
		rows = append(rows, e.rows(width)...)
	}
	t.cacheWidth, t.cacheRows, t.cacheValid = width, rows, true
	return rows
}

// rows renders one entry against a terminal of the given width.
//
// It is a pure function of the entry and the width. That is the whole point:
// calling it twice at the same width must give the same thing, and calling it
// at a new width must give a correct layout for that width rather than a
// patched-up version of the old one.
func (e entry) rows(width int) []string {
	switch e.kind {
	case entryFence:
		return []string{e.prefix + styleDim.Render(e.text)}
	case entryCode:
		// Code is reproduced as written: wrapping it would put line breaks
		// into what the user copies back out.
		return []string{e.prefix + styleCode.Render(e.text)}
	case entryProse:
		wrapped := wrapIndent(e.prefix, hangingIndent(e.text),
			renderMarkdown(e.text, styleKiwi), textWidthFor(width))
		return strings.Split(wrapped, "\n")
	default:
		return wrapStyled(e.text, width)
	}
}

// wrapStyled breaks already-styled text to width without touching its styling.
func wrapStyled(s string, width int) []string {
	lines := strings.Split(s, "\n")
	out := make([]string, 0, len(lines))
	for _, line := range lines {
		if width > 0 && lipgloss.Width(line) > width {
			out = append(out, strings.Split(ansi.Wrap(line, width, ""), "\n")...)
			continue
		}
		out = append(out, line)
	}
	return out
}
