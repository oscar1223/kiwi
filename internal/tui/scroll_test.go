package tui

import (
	"fmt"
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// scrolled builds a model whose transcript is far taller than its window.
func scrolled(t *testing.T) *Model {
	t.Helper()
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 12})
	for i := 0; i < 60; i++ {
		m.record(fmt.Sprintf("line %02d", i))
	}
	return m
}

// visible is the transcript as it currently appears on screen.
func visible(m *Model) string {
	return plain(strings.Join(m.viewportBlock(m.termWidth(), m.viewportHeight()), "\n"))
}

// A transcript taller than the window shows its end, not its beginning: new
// output is the point.
func TestViewFollowsTheNewestOutput(t *testing.T) {
	m := scrolled(t)
	if got := visible(m); !strings.Contains(got, "line 59") {
		t.Errorf("the newest line is not on screen:\n%s", got)
	}
}

// Scrolling up must stick. Without this the view is yanked back to the bottom
// on the next delta, which makes reading anything during a turn impossible.
func TestScrollingUpSurvivesNewOutput(t *testing.T) {
	m := scrolled(t)
	m.scrollBy(-20)
	if m.follow {
		t.Fatal("scrolling up left follow mode armed")
	}
	before := visible(m)

	m.record("line 60")
	if after := visible(m); after != before {
		t.Errorf("new output moved a view the user had scrolled:\n%s\n---\n%s", before, after)
	}
}

// Catching back up with the bottom resumes tracking, so the reader does not
// have to keep scrolling by hand for the rest of the turn.
func TestReachingTheBottomResumesFollowing(t *testing.T) {
	m := scrolled(t)
	m.scrollBy(-20)
	m.scrollBy(20)

	if !m.follow {
		t.Fatal("scrolling back to the bottom did not resume following")
	}
	m.record("line 60")
	if got := visible(m); !strings.Contains(got, "line 60") {
		t.Errorf("following did not resume:\n%s", got)
	}
}

func TestScrollStopsAtBothEnds(t *testing.T) {
	m := scrolled(t)

	m.scrollBy(-1000)
	if got := visible(m); !strings.Contains(got, "line 00") {
		t.Errorf("scrolling to the top did not reach the first line:\n%s", got)
	}
	if m.scroll < 0 {
		t.Errorf("scroll went negative: %d", m.scroll)
	}

	m.scrollBy(1000)
	if got := visible(m); !strings.Contains(got, "line 59") {
		t.Errorf("scrolling to the bottom did not reach the last line:\n%s", got)
	}
}

// Sending something means you want to see the reply.
func TestSubmitReturnsToTheBottom(t *testing.T) {
	m := scrolled(t)
	m.scrollBy(-30)
	m.submit("/help")

	if !m.follow {
		t.Error("submitting did not return the view to the newest output")
	}
}

// The wheel reaches us as bare arrows, so they scroll — unless the prompt has
// another line of its own to move onto.
func TestArrowsScrollOnlyWhenTheInputCannotUseThem(t *testing.T) {
	m := scrolled(t)

	if m.inputWantsArrow("up") || m.inputWantsArrow("down") {
		t.Error("a single-line prompt claimed the arrows")
	}

	m.input.SetValue("one\ntwo\nthree")
	m.input.CursorEnd()
	if !m.inputWantsArrow("up") {
		t.Error("a multi-line prompt with room above did not claim the up arrow")
	}
	if m.inputWantsArrow("down") {
		t.Error("a prompt whose cursor is on its last line claimed the down arrow")
	}
}

// Resizing must not strand the view somewhere the transcript no longer
// reaches.
func TestScrollIsClampedAfterAResize(t *testing.T) {
	m := scrolled(t)
	m.scrollBy(-40)
	m.Update(tea.WindowSizeMsg{Width: 80, Height: 50})

	rows := m.viewportBlock(m.termWidth(), m.viewportHeight())
	if len(rows) != m.viewportHeight() {
		t.Errorf("the viewport is %d rows, want %d", len(rows), m.viewportHeight())
	}
	if got := plain(strings.Join(rows, "\n")); strings.Count(got, "line 00") > 1 {
		t.Errorf("the viewport repeated content after a resize:\n%s", got)
	}
}
