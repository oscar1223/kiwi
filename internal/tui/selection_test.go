package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// captureClipboard swaps the system clipboard for a variable, so a test neither
// depends on one existing nor overwrites what was on it.
func captureClipboard(t *testing.T) *string {
	t.Helper()
	var got string
	prev := writeClipboard
	writeClipboard = func(s string) error { got = s; return nil }
	t.Cleanup(func() { writeClipboard = prev })
	return &got
}

// answered is a model whose transcript holds one short answer, at the top of
// a window tall enough not to scroll.
func answered(t *testing.T, lines ...string) *Model {
	t.Helper()
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 60, Height: 20})
	m.stream(strings.Join(lines, "\n") + "\n")
	return m
}

func drag(m *Model, fromX, fromY, toX, toY int) tea.Cmd {
	m.Update(tea.MouseClickMsg{X: fromX, Y: fromY, Button: tea.MouseLeft})
	m.Update(tea.MouseMotionMsg{X: toX, Y: toY, Button: tea.MouseLeft})
	_, cmd := m.Update(tea.MouseReleaseMsg{X: toX, Y: toY, Button: tea.MouseLeft})
	return cmd
}

// Dragging over the answer copies it: the mouse is captured, so the terminal
// will not.
func TestDragCopiesTheSelection(t *testing.T) {
	got := captureClipboard(t)
	m := answered(t, "primera línea", "segunda línea")

	// "● primera línea": the text starts after the two-column marker.
	cmd := drag(m, 2, 0, 8, 0)
	if *got != "primera" {
		t.Errorf("clipboard = %q, want %q", *got, "primera")
	}
	if cmd == nil {
		t.Error("no OSC 52 command: the copy would not reach a remote terminal")
	}
	if view := plain(m.View().Content); !strings.Contains(view, "copied 7 chars") {
		t.Errorf("the status line does not confirm the copy:\n%s", view)
	}
}

// Several rows copy as lines, without the gutter they are drawn behind and
// with accented characters intact.
func TestDragAcrossRowsDropsTheGutter(t *testing.T) {
	got := captureClipboard(t)
	m := answered(t, "```", "func año() {", "    return", "}", "```")

	drag(m, 0, 1, 59, 3)
	if want := "func año() {\n    return\n}"; *got != want {
		t.Errorf("clipboard = %q, want %q", *got, want)
	}
}

// A drag that runs backwards selects the same text.
func TestBackwardsDrag(t *testing.T) {
	got := captureClipboard(t)
	m := answered(t, "primera línea")

	drag(m, 8, 0, 2, 0)
	if *got != "primera" {
		t.Errorf("clipboard = %q, want %q", *got, "primera")
	}
}

// A click with no drag copies nothing and clears the highlight; so does typing.
func TestSelectionIsDismissed(t *testing.T) {
	got := captureClipboard(t)
	m := answered(t, "primera línea")

	drag(m, 2, 0, 8, 0)
	*got = ""
	m.Update(tea.MouseClickMsg{X: 4, Y: 0, Button: tea.MouseLeft})
	m.Update(tea.MouseReleaseMsg{X: 4, Y: 0, Button: tea.MouseLeft})
	if *got != "" || m.sel != nil {
		t.Errorf("a plain click copied %q and left sel = %+v", *got, m.sel)
	}

	drag(m, 2, 0, 8, 0)
	m.Update(key("a"))
	if m.sel != nil {
		t.Error("typing did not clear the selection")
	}
}

// The highlight covers the selected cells and leaves the row's text and width
// as they were.
func TestHighlightKeepsTheRow(t *testing.T) {
	captureClipboard(t)
	m := answered(t, "primera línea")
	before := strings.Split(m.View().Content, "\n")[0]

	m.Update(tea.MouseClickMsg{X: 2, Y: 0, Button: tea.MouseLeft})
	m.Update(tea.MouseMotionMsg{X: 8, Y: 0, Button: tea.MouseLeft})
	after := strings.Split(m.View().Content, "\n")[0]

	if after == before {
		t.Error("the selection is not drawn")
	}
	if plain(after) != plain(before) {
		t.Errorf("highlighting changed the text: %q → %q", plain(before), plain(after))
	}
}

// A press below the transcript — on the prompt or the status line — selects
// nothing.
func TestClickOutsideTheTranscript(t *testing.T) {
	got := captureClipboard(t)
	m := answered(t, "primera línea")

	drag(m, 0, 19, 10, 19)
	if *got != "" || m.sel != nil {
		t.Errorf("a drag on the status line copied %q", *got)
	}
}
