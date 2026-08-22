package tools

import (
	"fmt"
	"strings"
)

// diffContext is how many unchanged lines to show around each change.
const diffContext = 3

// UnifiedDiff renders a unified diff of two file versions.
//
// The user sees this before approving an edit, so it is deliberately plain: no
// colour codes, no syntax highlighting. Whoever renders it decides how it
// looks.
func UnifiedDiff(path, before, after string) string {
	if before == after {
		return ""
	}
	a := splitLines(before)
	b := splitLines(after)

	ops := diffOps(a, b)
	hunks := groupHunks(ops, diffContext)
	if len(hunks) == 0 {
		return ""
	}

	var out strings.Builder
	fmt.Fprintf(&out, "--- %s\n+++ %s\n", path, path)
	for _, h := range hunks {
		fmt.Fprintf(&out, "@@ -%d,%d +%d,%d @@\n", h.aStart, h.aLen, h.bStart, h.bLen)
		for _, op := range h.ops {
			out.WriteString(op.marker())
			out.WriteString(op.text)
			out.WriteString("\n")
		}
	}
	return out.String()
}

func splitLines(s string) []string {
	if s == "" {
		return nil
	}
	return strings.Split(strings.TrimSuffix(s, "\n"), "\n")
}

type opKind int

const (
	opEqual opKind = iota
	opDelete
	opInsert
)

type op struct {
	kind opKind
	text string
}

func (o op) marker() string {
	switch o.kind {
	case opDelete:
		return "-"
	case opInsert:
		return "+"
	default:
		return " "
	}
}

// diffOps computes a line diff via the classic LCS dynamic program.
//
// O(n*m) is fine here: these are single source files being previewed for a
// human, not a version control system.
func diffOps(a, b []string) []op {
	// Trim the common prefix and suffix first — for a typical edit_file that
	// leaves an LCS table of a handful of lines instead of the whole file.
	prefix := 0
	for prefix < len(a) && prefix < len(b) && a[prefix] == b[prefix] {
		prefix++
	}
	suffix := 0
	for suffix < len(a)-prefix && suffix < len(b)-prefix &&
		a[len(a)-1-suffix] == b[len(b)-1-suffix] {
		suffix++
	}

	midA := a[prefix : len(a)-suffix]
	midB := b[prefix : len(b)-suffix]

	ops := make([]op, 0, len(a)+len(b))
	for _, line := range a[:prefix] {
		ops = append(ops, op{opEqual, line})
	}
	ops = append(ops, lcsOps(midA, midB)...)
	for _, line := range a[len(a)-suffix:] {
		ops = append(ops, op{opEqual, line})
	}
	return normalize(ops)
}

func lcsOps(a, b []string) []op {
	n, m := len(a), len(b)
	if n == 0 {
		ops := make([]op, 0, m)
		for _, line := range b {
			ops = append(ops, op{opInsert, line})
		}
		return ops
	}
	if m == 0 {
		ops := make([]op, 0, n)
		for _, line := range a {
			ops = append(ops, op{opDelete, line})
		}
		return ops
	}

	table := make([][]int, n+1)
	for i := range table {
		table[i] = make([]int, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if a[i] == b[j] {
				table[i][j] = table[i+1][j+1] + 1
			} else {
				table[i][j] = max(table[i+1][j], table[i][j+1])
			}
		}
	}

	var ops []op
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case a[i] == b[j]:
			ops = append(ops, op{opEqual, a[i]})
			i++
			j++
		case table[i+1][j] >= table[i][j+1]:
			ops = append(ops, op{opDelete, a[i]})
			i++
		default:
			ops = append(ops, op{opInsert, b[j]})
			j++
		}
	}
	for ; i < n; i++ {
		ops = append(ops, op{opDelete, a[i]})
	}
	for ; j < m; j++ {
		ops = append(ops, op{opInsert, b[j]})
	}
	return ops
}

// normalize slides runs of changes forward past identical lines.
//
// When a file has repeated lines — closing braces, blank lines, a block of
// imports — the LCS walk is free to pick any of several equally-short edit
// scripts, and it tends to scatter one logical change across the file. Sliding
// each run as far forward as it can go without changing the result pulls the
// pieces back together, so a single edit previews as a single hunk. This is
// what git calls change compaction.
func normalize(ops []op) []op {
	for i := 0; i < len(ops); {
		if ops[i].kind == opEqual {
			i++
			continue
		}

		// Find the extent of this run of same-kind changes.
		kind := ops[i].kind
		start, end := i, i
		for end < len(ops) && ops[end].kind == kind {
			end++
		}

		// Slide it forward while the line entering the run equals the line
		// leaving it: removing (or adding) either one produces the same file.
		for end < len(ops) && ops[end].kind == opEqual && ops[start].text == ops[end].text {
			moved := ops[end]
			copy(ops[start+1:end+1], ops[start:end])
			ops[start] = moved
			start++
			end++
		}

		i = end
	}
	return ops
}

type hunk struct {
	aStart, aLen int
	bStart, bLen int
	ops          []op
}

// groupHunks slices the op stream into hunks with `context` unchanged lines
// around each run of changes, so a one-line edit in a 5000-line file previews
// as seven lines rather than five thousand.
func groupHunks(ops []op, context int) []hunk {
	changed := make([]bool, len(ops))
	any := false
	for i, o := range ops {
		if o.kind != opEqual {
			changed[i] = true
			any = true
		}
	}
	if !any {
		return nil
	}

	keep := make([]bool, len(ops))
	for i, c := range changed {
		if !c {
			continue
		}
		lo := max(0, i-context)
		hi := min(len(ops)-1, i+context)
		for j := lo; j <= hi; j++ {
			keep[j] = true
		}
	}

	var hunks []hunk
	aLine, bLine := 1, 1
	i := 0
	for i < len(ops) {
		if !keep[i] {
			switch ops[i].kind {
			case opEqual:
				aLine++
				bLine++
			case opDelete:
				aLine++
			case opInsert:
				bLine++
			}
			i++
			continue
		}

		h := hunk{aStart: aLine, bStart: bLine}
		for i < len(ops) && keep[i] {
			o := ops[i]
			h.ops = append(h.ops, o)
			switch o.kind {
			case opEqual:
				h.aLen++
				h.bLen++
				aLine++
				bLine++
			case opDelete:
				h.aLen++
				aLine++
			case opInsert:
				h.bLen++
				bLine++
			}
			i++
		}
		hunks = append(hunks, h)
	}
	return hunks
}
