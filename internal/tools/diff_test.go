package tools

import (
	"strings"
	"testing"
)

func TestUnifiedDiffBasics(t *testing.T) {
	got := UnifiedDiff("f.txt", "a\nb\nc\n", "a\nB\nc\n")
	for _, want := range []string{"--- f.txt", "+++ f.txt", "@@", " a", "-b", "+B", " c"} {
		if !strings.Contains(got, want) {
			t.Errorf("diff missing %q:\n%s", want, got)
		}
	}
}

func TestUnifiedDiffIdenticalIsEmpty(t *testing.T) {
	if got := UnifiedDiff("f.txt", "same\n", "same\n"); got != "" {
		t.Errorf("expected no diff, got:\n%s", got)
	}
}

func TestUnifiedDiffNewFile(t *testing.T) {
	got := UnifiedDiff("new.txt", "", "hello\nworld\n")
	if !strings.Contains(got, "+hello") || !strings.Contains(got, "+world") {
		t.Errorf("new file diff should be all additions:\n%s", got)
	}
	if strings.Contains(got, "\n-") {
		t.Errorf("new file diff should have no deletions:\n%s", got)
	}
}

func TestUnifiedDiffDeletion(t *testing.T) {
	got := UnifiedDiff("f.txt", "a\nb\nc\n", "a\nc\n")
	if !strings.Contains(got, "-b") {
		t.Errorf("deletion not shown:\n%s", got)
	}
	if strings.Contains(got, "+") && !strings.Contains(got, "+++") {
		t.Errorf("pure deletion should add nothing:\n%s", got)
	}
}

// A one-line change in a large file must preview as a few lines, not the
// whole file — otherwise the approval prompt is unreadable.
func TestUnifiedDiffOnlyShowsContextAroundChanges(t *testing.T) {
	var before, after strings.Builder
	for i := range 500 {
		line := "line\n"
		before.WriteString(line)
		if i == 250 {
			after.WriteString("CHANGED\n")
		} else {
			after.WriteString(line)
		}
	}

	got := UnifiedDiff("big.txt", before.String(), after.String())
	lines := strings.Count(got, "\n")
	if lines > 12 {
		t.Errorf("diff is %d lines; context should keep it small:\n%s", lines, got)
	}
	if !strings.Contains(got, "+CHANGED") {
		t.Errorf("the actual change is missing:\n%s", got)
	}
}

// Two changes far apart belong in separate hunks.
func TestUnifiedDiffSeparateHunks(t *testing.T) {
	var before, after strings.Builder
	for i := range 100 {
		before.WriteString("line\n")
		switch i {
		case 10:
			after.WriteString("FIRST\n")
		case 80:
			after.WriteString("SECOND\n")
		default:
			after.WriteString("line\n")
		}
	}

	got := UnifiedDiff("f.txt", before.String(), after.String())
	if n := countHunks(got); n != 2 {
		t.Errorf("got %d hunks, want 2:\n%s", n, got)
	}
}

// Repeated lines — closing braces, blank lines, import blocks — let the LCS
// pick any of several equally-short edit scripts. Without change compaction it
// scatters one logical edit across several hunks.
func TestUnifiedDiffCompactsChangesInRepetitiveCode(t *testing.T) {
	before := "func a() {\n\treturn 1\n}\n\nfunc b() {\n\treturn 2\n}\n\nfunc c() {\n\treturn 3\n}\n"
	after := "func a() {\n\treturn 1\n}\n\nfunc b() {\n\treturn 22\n}\n\nfunc c() {\n\treturn 3\n}\n"

	got := UnifiedDiff("f.go", before, after)
	if n := countHunks(got); n != 1 {
		t.Errorf("one edit should be one hunk, got %d:\n%s", n, got)
	}
	if !strings.Contains(got, "-\treturn 2") || !strings.Contains(got, "+\treturn 22") {
		t.Errorf("the edit is not shown as a single replacement:\n%s", got)
	}
}

func countHunks(diff string) int {
	n := 0
	for _, line := range strings.Split(diff, "\n") {
		if strings.HasPrefix(line, "@@") {
			n++
		}
	}
	return n
}

func TestUnifiedDiffHunkHeaderLineNumbers(t *testing.T) {
	got := UnifiedDiff("f.txt", "a\nb\nc\nd\ne\nf\ng\nh\n", "a\nb\nc\nd\nE\nf\ng\nh\n")
	// The change is on line 5, so with 3 lines of context the hunk starts at 2.
	if !strings.Contains(got, "@@ -2,7 +2,7 @@") {
		t.Errorf("unexpected hunk header:\n%s", got)
	}
}

func TestCountLines(t *testing.T) {
	cases := map[string]int{"": 0, "a": 1, "a\n": 1, "a\nb": 2, "a\nb\n": 2}
	for in, want := range cases {
		if got := countLines(in); got != want {
			t.Errorf("countLines(%q) = %d, want %d", in, got, want)
		}
	}
}
