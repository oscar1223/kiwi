package tui

import (
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
)

// plain strips the styling so a test can assert on what the user reads rather
// than on which escape sequences carry it.
func plain(s string) string { return ansiRE.ReplaceAllString(s, "") }

func TestMarkdownDropsEmphasisMarkers(t *testing.T) {
	base := lipgloss.NewStyle()
	cases := []struct{ in, want string }{
		{"**bold** text", "bold text"},
		{"__bold__ text", "bold text"},
		{"*italic* text", "italic text"},
		{"_italic_ text", "italic text"},
		{"***both*** text", "both text"},
		{"a `code` span", "a code span"},
		{"**bold with `code`**", "bold with code"},
		{"# Heading", "Heading"},
		{"### Deeper heading", "Deeper heading"},
		{"- an item", "• an item"},
		{"* an item", "• an item"},
		{"2. second", "2. second"},
		{"> quoted", "│ quoted"},
		{`escaped \*not emphasis\*`, "escaped *not emphasis*"},
	}
	for _, c := range cases {
		if got := plain(renderMarkdown(c.in, base)); got != c.want {
			t.Errorf("renderMarkdown(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// Markup that is not markup must survive: prose is full of asterisks and
// underscores that mean something else.
func TestMarkdownLeavesProseAlone(t *testing.T) {
	base := lipgloss.NewStyle()
	for _, in := range []string{
		"2 * 3 * 4",
		"a snake_case_name here",
		"an unclosed **span",
		"a lone * asterisk",
		"plain sentence",
		"    indented text",
	} {
		if got := plain(renderMarkdown(in, base)); got != in {
			t.Errorf("renderMarkdown(%q) = %q, want it unchanged", in, got)
		}
	}
}

func TestMarkdownActuallyEmboldens(t *testing.T) {
	got := renderMarkdown("**loud**", lipgloss.NewStyle())
	if !strings.Contains(got, "\x1b[1m") {
		t.Errorf("renderMarkdown(**loud**) = %q, want the bold escape", got)
	}
	if strings.Contains(got, "*") {
		t.Errorf("renderMarkdown(**loud**) = %q, want no literal asterisks", got)
	}
}

// A code span is literal: markup inside it is content, not markup.
func TestCodeSpanIsLiteral(t *testing.T) {
	got := plain(renderMarkdown("use `a*b` here", lipgloss.NewStyle()))
	if got != "use a*b here" {
		t.Errorf("got %q, want %q", got, "use a*b here")
	}
}
