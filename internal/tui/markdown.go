package tui

import (
	"strings"
	"unicode"

	"charm.land/lipgloss/v2"
)

// Markdown, one line at a time.
//
// An answer is streamed and printed line by line (see Model.stream), so a
// block-level renderer — which needs the whole document before it can decide
// anything — has nothing to work with here. What is left is everything
// markdown expresses *within* a line: emphasis, inline code, a heading, a list
// marker, a quote. That covers what actually shows up mid-sentence in an
// answer, and it is what left literal "**" on screen before.
//
// Anything unbalanced is passed through untouched: a lone "*" is
// multiplication, "snake_case" is an identifier, and "**" with no closer is
// half of a span the next chunk has not delivered yet.

// renderMarkdown styles one line, with base as the style of ordinary text
// around the markup.
func renderMarkdown(line string, base lipgloss.Style) string {
	trimmed := strings.TrimLeft(line, " \t")
	indent := line[:len(line)-len(trimmed)]

	switch {
	case trimmed == "":
		return line

	// Headings: the hashes are markup, not content, so the weight carries the
	// meaning instead of the punctuation.
	case headingLevel(trimmed) > 0:
		n := headingLevel(trimmed)
		return indent + renderInline(strings.TrimSpace(trimmed[n:]), styleHeading)

	// A thematic break has no text to style; dimming it keeps it from reading
	// as content.
	case isThematicBreak(trimmed):
		return indent + styleDim.Render(trimmed)

	case strings.HasPrefix(trimmed, "> "):
		return indent + styleQuote.Render("│ ") + renderInline(trimmed[2:], styleQuote)
	}

	if marker, rest, ok := listMarker(trimmed); ok {
		return indent + styleBullet.Render(marker) + " " + renderInline(rest, base)
	}
	return indent + renderInline(trimmed, base)
}

// renderInline styles the emphasis and code spans inside a run of text.
func renderInline(s string, base lipgloss.Style) string {
	var out, lit strings.Builder
	flush := func() {
		if lit.Len() > 0 {
			out.WriteString(base.Render(lit.String()))
			lit.Reset()
		}
	}

	r := []rune(s)
	for i := 0; i < len(r); {
		switch {
		// A backslash escape is the author saying "this asterisk is an
		// asterisk". The backslash itself is markup and goes away.
		case r[i] == '\\' && i+1 < len(r) && isMarkdownPunct(r[i+1]):
			lit.WriteRune(r[i+1])
			i += 2

		case r[i] == '`':
			if j := indexRuneFrom(r, '`', i+1); j > i+1 {
				flush()
				// Code spans are literal by definition: no recursion, so a
				// `*` inside one stays a `*`.
				out.WriteString(styleCode.Render(string(r[i+1 : j])))
				i = j + 1
				continue
			}
			lit.WriteRune(r[i])
			i++

		case r[i] == '*' || r[i] == '_':
			if content, next, style, ok := emphasisAt(r, i, base); ok {
				flush()
				// Recursing is what makes "**bold with `code`**" work.
				out.WriteString(renderInline(content, style))
				i = next
				continue
			}
			lit.WriteRune(r[i])
			i++

		default:
			lit.WriteRune(r[i])
			i++
		}
	}
	flush()
	return out.String()
}

// emphasisAt reads an emphasis span opening at i, returning its content, the
// index just past the closing delimiter, and the style to render it with.
//
// The flanking rules are what keep prose intact: an opener has to be attached
// to the word it emphasises (so "2 * 3" is arithmetic), and "_" additionally
// may not sit inside a word (so "snake_case_name" is one identifier).
func emphasisAt(r []rune, i int, base lipgloss.Style) (content string, next int, style lipgloss.Style, ok bool) {
	ch := r[i]
	run := 1
	for i+run < len(r) && r[i+run] == ch {
		run++
	}
	if run > 3 {
		return "", 0, base, false
	}
	open := i + run
	if open >= len(r) || unicode.IsSpace(r[open]) {
		return "", 0, base, false
	}
	if ch == '_' && i > 0 && isWordRune(r[i-1]) {
		return "", 0, base, false
	}

	for j := open; j < len(r); j++ {
		if r[j] != ch {
			continue
		}
		k := 1
		for j+k < len(r) && r[j+k] == ch {
			k++
		}
		switch {
		case k < run, unicode.IsSpace(r[j-1]):
			j += k - 1 // Skip the whole run; a piece of it cannot close either.
		case ch == '_' && j+run < len(r) && isWordRune(r[j+run]):
			j += k - 1
		default:
			return string(r[open:j]), j + run, emphasisStyle(base, run), true
		}
	}
	return "", 0, base, false
}

// emphasisStyle maps delimiter length to weight: one is italic, two is bold,
// three is both.
func emphasisStyle(base lipgloss.Style, run int) lipgloss.Style {
	switch run {
	case 1:
		return base.Italic(true)
	case 2:
		return base.Bold(true)
	default:
		return base.Bold(true).Italic(true)
	}
}

// headingLevel returns the number of leading hashes for an ATX heading, or 0
// for anything else — "#tag" included, since a hash with no space after it is
// not a heading.
func headingLevel(s string) int {
	n := 0
	for n < len(s) && s[n] == '#' {
		n++
	}
	if n == 0 || n > 6 || n >= len(s) || s[n] != ' ' {
		return 0
	}
	return n
}

// listMarker splits a list item into its marker and the rest of the line.
// Bullets are normalised to "•" because "-", "*" and "+" all mean the same
// thing and none of them reads as a bullet; numbers keep their own value,
// since renumbering them would misquote the answer.
func listMarker(s string) (marker, rest string, ok bool) {
	if len(s) > 1 && (s[0] == '-' || s[0] == '*' || s[0] == '+') && s[1] == ' ' {
		return "•", strings.TrimLeft(s[2:], " "), true
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i > 0 && i+1 < len(s) && (s[i] == '.' || s[i] == ')') && s[i+1] == ' ' {
		return s[:i+1], strings.TrimLeft(s[i+2:], " "), true
	}
	return "", "", false
}

// isThematicBreak reports a "---" style rule: three or more of one marker and
// nothing else.
func isThematicBreak(s string) bool {
	if len(s) < 3 {
		return false
	}
	ch := s[0]
	if ch != '-' && ch != '*' && ch != '_' {
		return false
	}
	return strings.Trim(s, string(ch)) == ""
}

// hangingIndent is how far the continuation lines of a wrapped line have to be
// pushed in to sit under its text rather than under its marker.
func hangingIndent(line string) int {
	trimmed := strings.TrimLeft(line, " \t")
	indent := len(line) - len(trimmed)
	if strings.HasPrefix(trimmed, "> ") {
		return indent + 2
	}
	if marker, _, ok := listMarker(trimmed); ok {
		return indent + lipgloss.Width(marker) + 1
	}
	return indent
}

func isMarkdownPunct(r rune) bool {
	return strings.ContainsRune("\\`*_{}[]()#+-.!>|~", r)
}

func isWordRune(r rune) bool {
	return r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
}

func indexRuneFrom(r []rune, target rune, from int) int {
	for i := from; i < len(r); i++ {
		if r[i] == target {
			return i
		}
	}
	return -1
}
