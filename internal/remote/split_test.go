package remote

import (
	"fmt"
	"strings"
	"testing"
)

// checkChunks asserts the properties every split must have: no chunk over the
// limit, none empty, and every fence opened in a chunk closed in it.
func checkChunks(t *testing.T, chunks []string, limit int) {
	t.Helper()
	for i, c := range chunks {
		if n := utf16Len(c); n > limit {
			t.Errorf("chunk %d is %d units, over the limit of %d", i, n, limit)
		}
		if strings.TrimSpace(c) == "" {
			t.Errorf("chunk %d is empty; Telegram rejects empty messages", i)
		}
		fences := 0
		for _, line := range strings.Split(c, "\n") {
			if isFence(line) {
				fences++
			}
		}
		if fences%2 != 0 {
			t.Errorf("chunk %d leaves a code block open:\n%s", i, c)
		}
	}
}

// words strips everything but the words, to compare content across a split.
// Fence lines are dropped: a split code block repeats them on purpose.
func words(s string) string {
	var kept []string
	for _, line := range strings.Split(s, "\n") {
		if !isFence(line) {
			kept = append(kept, line)
		}
	}
	return strings.Join(strings.Fields(strings.Join(kept, " ")), " ")
}

func TestSplitAroundTheLimit(t *testing.T) {
	for _, n := range []int{4095, 4096, 4097} {
		text := strings.Repeat("a", n)
		chunks := splitMessage(text, maxMessage)
		checkChunks(t, chunks, maxMessage)

		want := 1
		if n > maxMessage {
			want = 2
		}
		if len(chunks) != want {
			t.Errorf("%d characters: %d chunks, want %d", n, len(chunks), want)
		}
		if strings.Join(chunks, "") != text {
			t.Errorf("%d characters: the text changed when split", n)
		}
	}
}

func TestShortTextIsUntouched(t *testing.T) {
	text := "hola\n\n```go\nfmt.Println(1)\n```\n"
	if got := splitMessage(text, maxMessage); len(got) != 1 || got[0] != text {
		t.Errorf("got %q", got)
	}
}

func TestCodeBlockAcrossTheLimitStaysWhole(t *testing.T) {
	limit := 200
	block := "```go\n" + strings.Repeat("x := 1\n", 15) + "```\n" // ~111 units
	text := strings.Repeat("palabra ", 15) + "\n\n" + block + "\nfin\n"
	// The prose plus the block do not fit together, so the block has to move
	// to the next message whole rather than be cut where the limit falls.
	chunks := splitMessage(text, limit)
	checkChunks(t, chunks, limit)

	found := false
	for _, c := range chunks {
		if strings.Contains(c, strings.TrimRight(block, "\n")) {
			found = true
		}
	}
	if !found {
		t.Errorf("the code block was split although it fits in one message:\n%q", chunks)
	}
}

func TestCodeBlockLongerThanAMessage(t *testing.T) {
	limit := 100
	var b strings.Builder
	b.WriteString("Mira:\n\n```python\n")
	for i := range 40 {
		fmt.Fprintf(&b, "print(%d)\n", i)
	}
	b.WriteString("```\nYa está.")
	text := b.String()

	chunks := splitMessage(text, limit)
	checkChunks(t, chunks, limit)
	if len(chunks) < 3 {
		t.Fatalf("got %d chunks, want the block spread over several", len(chunks))
	}
	for _, c := range chunks {
		if strings.Contains(c, "print(") && !strings.Contains(c, "```python\n") {
			t.Errorf("a piece of the block lost its opening fence (and language):\n%s", c)
		}
	}
	if words(strings.Join(chunks, "\n")) != words(text) {
		t.Error("content was lost or reordered")
	}
}

func TestPrefersParagraphBreaks(t *testing.T) {
	limit := 100
	p1 := strings.Repeat("uno ", 15) // 60
	p2 := strings.Repeat("dos ", 15) // 60
	text := p1 + "\n\n" + p2
	chunks := splitMessage(text, limit)
	checkChunks(t, chunks, limit)
	if len(chunks) != 2 || strings.TrimSpace(chunks[0]) != strings.TrimSpace(p1) {
		t.Errorf("want one paragraph per message, got %q", chunks)
	}
}

func TestLongLineIsCutAtASpace(t *testing.T) {
	limit := 50
	text := strings.Repeat("palabra ", 30) // one 240-unit line
	chunks := splitMessage(text, limit)
	checkChunks(t, chunks, limit)
	for i, c := range chunks[:len(chunks)-1] {
		if !strings.HasSuffix(strings.TrimSpace(c), "palabra") {
			t.Errorf("chunk %d cuts a word: %q", i, c)
		}
	}
	if words(strings.Join(chunks, " ")) != words(text) {
		t.Error("content was lost")
	}
}

func TestEmojiCountDouble(t *testing.T) {
	text := strings.Repeat("🥝", 3000) // 3000 runes, 6000 UTF-16 units
	chunks := splitMessage(text, maxMessage)
	checkChunks(t, chunks, maxMessage)
	if len(chunks) != 2 {
		t.Errorf("got %d chunks; 3000 emoji are 6000 units and need two", len(chunks))
	}
	if strings.Join(chunks, "") != text {
		t.Error("the text changed when split")
	}
}

func TestUnclosedFence(t *testing.T) {
	limit := 60
	text := "```\n" + strings.Repeat("línea\n", 30) // the model never closed it
	chunks := splitMessage(text, limit)
	checkChunks(t, chunks, limit)
}

func TestSplitProperties(t *testing.T) {
	// A mix of everything, at several limits.
	var b strings.Builder
	for i := range 30 {
		fmt.Fprintf(&b, "## Paso %d\n\nTexto con tildes, ñ y algún 🥝 en la línea %d.\n\n", i, i)
		if i%4 == 0 {
			b.WriteString("```bash\n")
			for j := range i + 3 {
				fmt.Fprintf(&b, "echo %d-%d\n", i, j)
			}
			b.WriteString("```\n\n")
		}
		if i%7 == 0 {
			b.WriteString(strings.Repeat("larguísima ", 40) + "\n\n")
		}
	}
	text := b.String()

	for _, limit := range []int{80, 150, 500, 4096} {
		t.Run(fmt.Sprint(limit), func(t *testing.T) {
			chunks := splitMessage(text, limit)
			checkChunks(t, chunks, limit)
			if words(strings.Join(chunks, "\n")) != words(text) {
				t.Error("content was lost or reordered")
			}
		})
	}
}
