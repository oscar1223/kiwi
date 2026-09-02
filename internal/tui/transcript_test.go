package tui

import (
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
)

// sample builds a transcript with one of everything: styled notices, prose
// long enough to wrap, and a fenced block.
func sample() *transcript {
	t := &transcript{}
	t.addStyled(styleDim.Render("  banner line"))
	t.add(entry{kind: entryProse, text: strings.Repeat("palabra ", 30), prefix: "● "})
	t.add(entry{kind: entryProse, text: "- un bullet que también es largo " + strings.Repeat("x ", 20), prefix: "  "})
	t.add(entry{kind: entryFence, text: "```go", prefix: "  "})
	t.add(entry{kind: entryCode, text: "x := " + strings.Repeat("a", 100), prefix: "  "})
	t.add(entry{kind: entryFence, text: "```", prefix: "  "})
	t.addStyled(styleWarn.Render("  cancelled"))
	return t
}

// The whole reason the transcript exists: narrowing the window and widening it
// back must land exactly where it started. While Kiwi wrapped lines as it
// printed them into the terminal's scrollback this was impossible — the breaks
// were measured once, against a width that no longer applied.
func TestTranscriptSurvivesResizeRoundTrip(t *testing.T) {
	tr := sample()
	want := append([]string(nil), tr.render(100)...)

	tr.render(40)
	tr.render(200)
	tr.render(37)
	got := tr.render(100)

	if len(got) != len(want) {
		t.Fatalf("round trip changed the row count: %d then %d", len(want), len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("row %d changed across a resize round trip:\n old %q\n new %q",
				i, plain(want[i]), plain(got[i]))
		}
	}
}

// Rendering must be a pure function of the entry and the width. It is called
// again on every streamed delta, so anything it mutated would drift.
func TestTranscriptRenderIsRepeatable(t *testing.T) {
	tr := sample()
	first := append([]string(nil), tr.render(60)...)

	// Defeat the cache: a different width, then back.
	tr.render(61)
	second := tr.render(60)

	for i := range first {
		if first[i] != second[i] {
			t.Errorf("row %d differs between two renders at the same width:\n %q\n %q",
				i, plain(first[i]), plain(second[i]))
		}
	}
}

// Nothing may be wider than the window it was rendered for, except the fenced
// code that is deliberately left as written.
func TestTranscriptFitsItsWidth(t *testing.T) {
	for _, width := range []int{30, 45, 80, 120} {
		tr := &transcript{}
		tr.addStyled(styleDim.Render(strings.Repeat("aviso ", 40)))
		tr.add(entry{kind: entryProse, text: strings.Repeat("palabra ", 40), prefix: "● "})

		for i, row := range tr.render(width) {
			if w := lipgloss.Width(row); w > width {
				t.Errorf("width %d: row %d is %d cells: %q", width, i, w, plain(row))
			}
		}
	}
}

// Continuation lines hang under the text, not under the bullet.
func TestTranscriptHangsWrappedProse(t *testing.T) {
	tr := &transcript{}
	tr.add(entry{kind: entryProse, text: strings.Repeat("palabra ", 30), prefix: "● "})

	rows := tr.render(40)
	if len(rows) < 2 {
		t.Fatalf("a 240-character line was not wrapped at width 40: %q", plain(rows[0]))
	}
	for i, row := range rows[1:] {
		if !strings.HasPrefix(plain(row), "  ") {
			t.Errorf("continuation row %d is not indented: %q", i+1, plain(row))
		}
	}
}

// Code inside a fence is reproduced as written: wrapping it would put line
// breaks into whatever the user copies back out.
func TestTranscriptDoesNotWrapFencedCode(t *testing.T) {
	tr := &transcript{}
	code := "x := " + strings.Repeat("a", 100)
	tr.add(entry{kind: entryCode, text: code, prefix: "  "})

	rows := tr.render(40)
	if len(rows) != 1 {
		t.Fatalf("fenced code was wrapped into %d rows", len(rows))
	}
	if !strings.Contains(plain(rows[0]), code) {
		t.Errorf("fenced code was altered: %q", plain(rows[0]))
	}
}

// Markup inside a fence is code, not markup.
func TestTranscriptKeepsFencedAsterisks(t *testing.T) {
	tr := &transcript{}
	tr.add(entry{kind: entryCode, text: "int **p = &q;", prefix: "  "})

	if got := plain(tr.render(80)[0]); !strings.Contains(got, "**p") {
		t.Errorf("fenced code lost its asterisks: %q", got)
	}
}

func TestTranscriptResetEmptiesIt(t *testing.T) {
	tr := sample()
	tr.render(80) // prime the cache, so reset has something to invalidate
	tr.reset()
	if rows := tr.render(80); len(rows) != 0 {
		t.Errorf("reset left %d rows behind", len(rows))
	}
}
