package tui

import (
	"strings"
	"testing"

	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

const sampleTable = "| Tool | What it does | Mode |\n|---|:---:|---|\n| `web_search` | searches the web and returns a title, URL and snippet for each result | plan and work |\n| `web_fetch` | downloads a page | same |\n"

func streamed(t *testing.T, text string, width int) []string {
	t.Helper()
	m, _ := newTestModel(t, permission.ModeAsk)
	m.stream(text)
	m.flushTail()
	var out []string
	for _, r := range m.transcript.screen(width) {
		out = append(out, plain(r))
	}
	return out
}

// A table is drawn with its columns lined up and nothing wider than the
// window, instead of raw pipes that wrap into the next row.
func TestTableColumnsLineUp(t *testing.T) {
	for _, width := range []int{100, 50} {
		rows := streamed(t, sampleTable, width)
		sepCol := -1
		for _, r := range rows {
			if strings.Contains(r, "|") {
				t.Errorf("width %d: raw pipe left in %q", width, r)
			}
			if w := lipgloss.Width(r); w > width {
				t.Errorf("width %d: row is %d wide: %q", width, w, r)
			}
			if i := strings.Index(r, "│"); i >= 0 {
				i = lipgloss.Width(r[:i]) // columns, not bytes: "●" is three
				if sepCol >= 0 && i != sepCol {
					t.Errorf("width %d: columns do not line up:\n%s", width, strings.Join(rows, "\n"))
					break
				}
				sepCol = i
			}
		}
		joined := strings.Join(rows, "\n")
		for _, want := range []string{"Tool", "web_search", "downloads a page", "same"} {
			if !strings.Contains(joined, want) {
				t.Errorf("width %d: %q missing:\n%s", width, want, joined)
			}
		}
	}
}

// Too many columns for the window: each row becomes field: value lines.
func TestNarrowTableBecomesAList(t *testing.T) {
	rows := streamed(t, sampleTable, 26)
	joined := strings.Join(rows, "\n")
	if strings.Contains(joined, "│") {
		t.Fatalf("a 26-column window still drew columns:\n%s", joined)
	}
	if !strings.Contains(joined, "Tool: web_search") || !strings.Contains(joined, "Mode: same") {
		t.Errorf("rows are not listed as field: value:\n%s", joined)
	}
}

// Text after a table goes back to being prose.
func TestTableEndsAtTheFirstOrdinaryLine(t *testing.T) {
	rows := streamed(t, sampleTable+"after the table\n", 100)
	if last := rows[len(rows)-1]; strings.TrimSpace(last) != "after the table" {
		t.Errorf("last row = %q", last)
	}
}

// A table inside a code fence is code, and stays as written.
func TestTableInsideAFenceIsLeftAlone(t *testing.T) {
	rows := streamed(t, "```\n| a | b |\n```\n", 100)
	if !strings.Contains(strings.Join(rows, "\n"), "| a | b |") {
		t.Errorf("fenced table was reformatted: %q", rows)
	}
}

func TestCleanLine(t *testing.T) {
	for in, want := range map[string]string{
		"a\tb":                        "a   b",
		"abcd\te":                     "abcd    e",
		"progress 10%\rprogress 100%": "progress 100%",
		"windows line\r":              "windows line",
		"\x1b[31mred\x1b[0m text":     "red text",
		"bell\a here":                 "bell here",
	} {
		if got := cleanLine(in); got != want {
			t.Errorf("cleanLine(%q) = %q, want %q", in, got, want)
		}
	}
}

// Tool output is cleaned before it is cut to one line, so a truncation can
// never leave half an escape sequence behind.
func TestToolResultIsCleaned(t *testing.T) {
	out := plain(renderToolResult("\x1b[32mok\x1b[0m\tdone\rDONE", false))
	if strings.ContainsAny(out, "\x1b\t\r") || !strings.Contains(out, "DONE") {
		t.Errorf("tool result not cleaned: %q", out)
	}
}
