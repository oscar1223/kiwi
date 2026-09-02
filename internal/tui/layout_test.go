package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Nothing the interface draws may be wider than the window: in inline mode an
// overflowing line costs the renderer a row it has not accounted for, and the
// next frame lands on top of the wrong one — which is what a resize exposed.
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

func TestOutputWrapsToTerminalWidth(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 40, Height: 24})

	lines := strings.Split(m.renderLine(strings.Repeat("palabra ", 30)), "\n")
	if len(lines) < 2 {
		t.Fatalf("a 240-character line was not wrapped at width 40: %q", plain(lines[0]))
	}
	for i, line := range lines {
		if w := lipgloss.Width(line); w > 40 {
			t.Errorf("wrapped line %d is %d cells wide: %q", i, w, plain(line))
		}
		// Continuation lines hang under the text, not under the bullet.
		if i > 0 && !strings.HasPrefix(plain(line), "  ") {
			t.Errorf("wrapped line %d is not indented: %q", i, plain(line))
		}
	}
}

// Code inside a fence is reproduced as written: wrapping it would put line
// breaks into whatever the user copies back out of the scrollback.
func TestFencedCodeIsNotWrapped(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 40, Height: 24})

	m.renderLine("```go")
	code := "x := " + strings.Repeat("a", 100)
	if got := m.renderLine(code); strings.Contains(got, "\n") {
		t.Errorf("fenced code was wrapped: %q", plain(got))
	}
}

// Markup inside a fence is code, not markup.
func TestFencedCodeKeepsItsAsterisks(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})

	m.renderLine("```c")
	if got := plain(m.renderLine("int **p = &q;")); !strings.Contains(got, "**p") {
		t.Errorf("fenced code lost its asterisks: %q", got)
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

// A streamed paragraph has no length limit until its newline arrives, so the
// live preview of it must not be allowed to grow past the window.
func TestViewFitsTerminalHeight(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 60, Height: 12})
	m.busy = true
	m.tail = strings.Repeat("una frase larguísima que no termina nunca ", 30)

	if rows := len(strings.Split(m.View().Content, "\n")); rows > 12 {
		t.Errorf("the live area is %d rows in a 12-row window", rows)
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
