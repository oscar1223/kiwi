package remote

import (
	"strings"
	"unicode/utf8"
)

// maxMessage is Telegram's limit per message, in UTF-16 code units: characters
// outside the BMP (most emoji) count twice. Longer text is rejected whole, not
// truncated.
const maxMessage = 4096

const fence = "```"

// splitMessage cuts text into messages of at most limit UTF-16 units.
//
// It cuts between paragraphs where it can, between lines where it must, and
// inside a line only when a single line is longer than a message. A fenced
// code block is never cut while it fits in one message; one that does not is
// split by lines, each piece closed and reopened with the same fence so every
// message stays a well-formed block.
func splitMessage(text string, limit int) []string {
	if utf16Len(text) <= limit {
		return []string{text}
	}

	var (
		chunks []string
		cur    strings.Builder
		curLen int
	)
	flush := func() {
		if s := strings.Trim(cur.String(), "\n"); strings.TrimSpace(s) != "" {
			chunks = append(chunks, s)
		}
		cur.Reset()
		curLen = 0
	}

	for _, u := range units(text) {
		for _, piece := range fit(u, limit) {
			n := utf16Len(piece)
			if curLen+n > limit {
				flush()
			}
			cur.WriteString(piece)
			curLen += n
		}
	}
	flush()
	return chunks
}

// unit is a paragraph or a whole fenced code block, with the blank lines that
// follow it. Joining every unit gives back the original text.
type unit struct {
	text string
	code bool
}

func units(text string) []unit {
	lines := strings.SplitAfter(text, "\n")
	var out []unit

	for i := 0; i < len(lines); {
		start := i
		code := isFence(lines[i])
		switch {
		case code:
			i++ // the opening fence
			for i < len(lines) && !isFence(lines[i]) {
				i++
			}
			if i < len(lines) {
				i++ // the closing fence; an unclosed block runs to the end
			}
		case isBlank(lines[i]):
			// Blank lines before anything else: a unit of their own.
		default:
			for i < len(lines) && !isBlank(lines[i]) && !isFence(lines[i]) {
				i++
			}
		}
		for i < len(lines) && isBlank(lines[i]) {
			i++
		}
		out = append(out, unit{text: strings.Join(lines[start:i], ""), code: code})
	}
	return out
}

// fit splits a unit that is too long for one message.
func fit(u unit, limit int) []string {
	if utf16Len(u.text) <= limit {
		return []string{u.text}
	}
	if u.code {
		return fitCode(u.text, limit)
	}
	return packLines(strings.SplitAfter(u.text, "\n"), limit)
}

// fitCode splits a code block by lines, wrapping each piece in the block's own
// opening fence (language included) and a closing one.
func fitCode(block string, limit int) []string {
	lines := strings.SplitAfter(strings.TrimRight(block, "\n"), "\n")
	open := strings.TrimRight(lines[0], "\n") + "\n"
	body := lines[1:]
	if n := len(body); n > 0 && isFence(body[n-1]) {
		body = body[:n-1]
	}
	if n := len(body); n > 0 {
		body[n-1] = strings.TrimRight(body[n-1], "\n") + "\n"
	}

	closing := fence + "\n"
	budget := limit - utf16Len(open) - utf16Len(closing)
	var out []string
	for _, p := range packLines(body, budget) {
		out = append(out, open+p+closing)
	}
	return out
}

// packLines groups lines into pieces of at most limit units, cutting a line
// only when it is longer than limit on its own.
func packLines(lines []string, limit int) []string {
	var (
		out    []string
		cur    strings.Builder
		curLen int
	)
	for _, line := range lines {
		for _, part := range hardSplit(line, limit) {
			n := utf16Len(part)
			if curLen+n > limit && curLen > 0 {
				out = append(out, cur.String())
				cur.Reset()
				curLen = 0
			}
			cur.WriteString(part)
			curLen += n
		}
	}
	if curLen > 0 {
		out = append(out, cur.String())
	}
	return out
}

// hardSplit cuts s into parts of at most limit units, preferring to cut after
// a space when there is one in the second half of the window.
func hardSplit(s string, limit int) []string {
	var out []string
	for utf16Len(s) > limit {
		cut, units, lastSpace := 0, 0, -1
		for i, c := range s {
			if units+utf16Units(c) > limit {
				cut = i
				break
			}
			units += utf16Units(c)
			if c == ' ' && units > limit/2 {
				lastSpace = i + 1
			}
		}
		if lastSpace > 0 {
			cut = lastSpace
		}
		if cut == 0 {
			// limit is smaller than the first character: send it alone
			// rather than loop forever.
			_, size := utf8.DecodeRuneInString(s)
			cut = size
		}
		out = append(out, s[:cut])
		s = s[cut:]
	}
	return append(out, s)
}

func isFence(line string) bool { return strings.HasPrefix(strings.TrimSpace(line), fence) }
func isBlank(line string) bool { return strings.TrimSpace(line) == "" }

func utf16Len(s string) int {
	n := 0
	for _, c := range s {
		n += utf16Units(c)
	}
	return n
}

func utf16Units(c rune) int {
	if c > 0xFFFF {
		return 2
	}
	return 1
}
