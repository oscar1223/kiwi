package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// The input grows with what is typed. A box stuck at one row scrolls instead,
// and a wrapped or broken line pushes the one above it out of sight.
func TestInputGrowsAsItWraps(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 40, Height: 24})

	for _, r := range strings.Repeat("palabra ", 12) {
		m.Update(key(string(r)))
	}
	if got := m.input.Height(); got < 2 {
		t.Fatalf("input is %d rows tall after wrapping", got)
	}
	if view := plain(m.View().Content); !strings.Contains(view, "› palabra") {
		t.Errorf("the first line of the prompt scrolled out of view:\n%s", view)
	}
}

// shift+enter and ctrl+j break the line instead of sending, and the box makes
// room for the new one.
func TestNewlineKeysInsertALine(t *testing.T) {
	for _, k := range []tea.KeyPressMsg{
		{Code: tea.KeyEnter, Mod: tea.ModShift},
		{Code: 'j', Mod: tea.ModCtrl},
	} {
		m, _ := newTestModel(t, permission.ModeAsk)
		m.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
		m.Update(key("a"))
		m.Update(k)
		m.Update(key("b"))

		if got := m.input.Value(); got != "a\nb" {
			t.Errorf("%s: input = %q, want a newline between the lines", k, got)
		}
		if got := m.input.Height(); got != 2 {
			t.Errorf("%s: input is %d rows tall with two lines", k, got)
		}
	}
}

// A tall prompt with a picker open in a short window must not push the frame
// past the top of the screen: the extras give way, the prompt does not.
func TestTallBottomBlockStillFitsTheWindow(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 60, Height: 10})
	m.showShortcuts = true
	m.queued = []string{"one", "two", "three", "four"}
	m.input.SetValue("a\nb\nc\nd\ne\nf")

	rows := strings.Split(m.View().Content, "\n")
	if len(rows) != 10 {
		t.Fatalf("the frame is %d rows in a 10-row window", len(rows))
	}
	if view := plain(strings.Join(rows, "\n")); !strings.Contains(view, "› a") || !strings.Contains(view, "f") {
		t.Errorf("the prompt was trimmed:\n%s", view)
	}
}

// A line of code wider than the window is broken on screen, or the terminal
// wraps it on its own and the frame gains rows it did not count.
func TestLongCodeFitsOnScreen(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 40, Height: 12})
	code := "x := " + strings.Repeat("a", 100)
	m.transcript.add(entry{kind: entryCode, text: code, prefix: "  "})

	content := m.View().Content
	for i, line := range strings.Split(content, "\n") {
		if w := lipgloss.Width(line); w > 40 {
			t.Errorf("line %d is %d cells wide", i, w)
		}
	}
	if rows := len(strings.Split(content, "\n")); rows != 12 {
		t.Errorf("the frame is %d rows in a 12-row window", rows)
	}
	// Kept whole for the scrollback printed on exit, so it copies as written.
	if !strings.Contains(plain(m.Transcript()), code) {
		t.Error("the exit transcript broke the code line")
	}
}

// The wheel always scrolls the conversation, even when there are earlier
// prompts an up arrow would recall.
func TestWheelScrollsEvenWithHistory(t *testing.T) {
	m := scrolled(t)
	m.promptHistory.record("an earlier prompt")
	m.promptHistory.reset()

	m.Update(tea.MouseWheelMsg{Button: tea.MouseWheelUp})
	if m.follow {
		t.Fatal("wheel up did not leave the bottom")
	}
	if m.input.Value() != "" {
		t.Errorf("wheel up recalled a prompt: %q", m.input.Value())
	}
	m.Update(tea.MouseWheelMsg{Button: tea.MouseWheelDown})
	if !m.follow {
		t.Error("wheel down did not return to the bottom")
	}
}

// The terminal tab carries the kiwi and the project.
func TestWindowTitle(t *testing.T) {
	if got := windowTitle("/home/me/code/shop"); got != "🥝 kiwi · shop" {
		t.Errorf("windowTitle = %q", got)
	}
}
