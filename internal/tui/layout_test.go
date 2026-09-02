package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Nothing the interface draws may be wider than the window: an overflowing
// line costs the renderer a row it has not accounted for, and pushes the rest
// of the frame off the bottom of the screen.
func TestViewFitsEveryTerminalWidth(t *testing.T) {
	for _, width := range []int{30, 45, 80, 120} {
		m, _ := newTestModel(t, permission.ModeAsk)
		m.Update(tea.WindowSizeMsg{Width: width, Height: 24})
		m.busy = true
		m.tail = strings.Repeat("streaming prose that keeps going ", 8)
		m.input.SetValue("/")

		for i, line := range strings.Split(m.View().Content, "\n") {
			if w := lipgloss.Width(line); w > width {
				t.Errorf("width %d: line %d is %d cells wide: %q", width, i, w, plain(line))
			}
		}
	}
}

// A window too short for a tall input would otherwise push the prompt, the
// status line and everything else off the screen.
func TestResizeCapsInputHeight(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.input.SetValue(strings.Repeat("line\n", 20))

	m.Update(tea.WindowSizeMsg{Width: 80, Height: 8})
	if got := m.input.MaxHeight; got > 8 {
		t.Errorf("input MaxHeight = %d in an 8-row window", got)
	}
	if got := m.input.Height(); got > 8 {
		t.Errorf("input height = %d in an 8-row window", got)
	}
}

// The frame owns the alt screen, so it must fill it exactly: short of the
// height and the block below floats; over it and the status line is pushed
// off. A streamed paragraph has no length limit until its newline arrives,
// which is the case most likely to overrun.
func TestViewFillsTerminalHeightExactly(t *testing.T) {
	for _, height := range []int{8, 12, 40} {
		m, _ := newTestModel(t, permission.ModeAsk)
		m.Update(tea.WindowSizeMsg{Width: 60, Height: height})
		m.busy = true
		m.tail = strings.Repeat("una frase larguísima que no termina nunca ", 30)

		if rows := len(strings.Split(m.View().Content, "\n")); rows != height {
			t.Errorf("the frame is %d rows in a %d-row window", rows, height)
		}
	}
}

// An empty session must still fill the screen, with the prompt pinned to the
// bottom rather than floating under the banner.
func TestViewFillsTerminalHeightWhenEmpty(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 30})

	if rows := len(strings.Split(m.View().Content, "\n")); rows != 30 {
		t.Errorf("an empty session drew %d rows in a 30-row window", rows)
	}
}

// A picker with more options than the window has rows must still show the
// highlighted one.
func TestPickerKeepsTheHighlightedRowVisible(t *testing.T) {
	const count, limit = 40, 8
	for _, index := range []int{0, 20, count - 1} {
		start, end := listWindow(count, index, limit)
		if end-start > limit {
			t.Errorf("index %d: window %d..%d exceeds %d rows", index, start, end, limit)
		}
		if index < start || index >= end {
			t.Errorf("index %d fell outside the window %d..%d", index, start, end)
		}
	}
	if start, end := listWindow(3, 1, 8); start != 0 || end != 3 {
		t.Errorf("a list that fits was windowed: %d..%d", start, end)
	}
}

// A cursor is positioned against the frame, not against the widget that owns
// it. The prompt sits at the bottom of the screen behind a marker, so both
// offsets have to be applied — getting either wrong puts the caret somewhere
// the user is not typing.
func TestCursorLandsOnThePrompt(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.input.Focus()
	m.Update(tea.WindowSizeMsg{Width: 60, Height: 20})
	for i := 0; i < 40; i++ {
		m.record("una línea cualquiera de relleno")
	}
	m.input.SetValue("hola")
	m.input.CursorEnd()

	v := m.View()
	if v.Cursor == nil {
		t.Fatal("a focused prompt has no cursor")
	}
	rows := strings.Split(v.Content, "\n")
	if v.Cursor.Y < 0 || v.Cursor.Y >= len(rows) {
		t.Fatalf("cursor row %d is outside a %d-row frame", v.Cursor.Y, len(rows))
	}
	if got := plain(rows[v.Cursor.Y]); !strings.Contains(got, promptMarker) {
		t.Errorf("the cursor is on row %d, which is not the prompt: %q", v.Cursor.Y, got)
	}
	// Four characters typed behind the marker.
	if want := lipgloss.Width(promptMarker) + len("hola"); v.Cursor.X != want {
		t.Errorf("cursor column = %d, want %d", v.Cursor.X, want)
	}
}
