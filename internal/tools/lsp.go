package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/oscar1223/kiwi/internal/lsp"
)

// maxLSPResults caps one answer. "Used in 400 places" is a fact worth
// reporting; four hundred lines of it is a fact that costs the rest of the
// turn its context window.
const maxLSPResults = 60

// LSP answers questions about code through the project's language server.
//
// It earns its place on the one question grep cannot answer honestly: which of
// these matches is the same symbol. grep finds a name; a language server
// resolves it, so a method on one type does not drag in the identically named
// method on another, and a shadowed local does not look like a package-level
// definition.
type LSP struct {
	Manager *lsp.Manager
	FS      *FS
}

func (LSP) Name() string { return "lsp" }

func (t LSP) Description() string {
	langs := strings.Join(t.Manager.Languages(), ", ")
	return "Ask the language server about code: where a symbol is defined " +
		"(definition), everywhere it is used (references), or what is wrong " +
		"with a file (diagnostics). Resolves symbols properly, so it does not " +
		"confuse same-named methods on different types the way grep does. " +
		"Available for: " + langs + "."
}

func (LSP) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"operation": map[string]any{
				"type":        "string",
				"enum":        []string{"definition", "references", "diagnostics"},
				"description": "What to ask.",
			},
			"path": map[string]any{"type": "string", "description": "File to ask about."},
			"line": map[string]any{
				"type":        "integer",
				"description": "1-based line the symbol is on. Required for definition and references.",
			},
			"symbol": map[string]any{
				"type": "string",
				"description": "The symbol's name. Used to find it on the line, so you do not " +
					"have to count columns. Strongly recommended.",
			},
		},
		"required": []string{"operation", "path"},
	}
}

func (t LSP) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Operation string `json:"operation"`
		Path      string `json:"path"`
		Line      int    `json:"line"`
		Symbol    string `json:"symbol"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	abs, err := t.FS.resolve(in.Path)
	if err != nil {
		return "", err
	}
	data, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}
	content := string(data)

	client, err := t.Manager.For(ctx, abs)
	if err != nil {
		return "", err
	}

	if in.Operation == "diagnostics" {
		found, err := client.Diagnostics(ctx, abs, content)
		if err != nil {
			return "", err
		}
		return t.renderDiagnostics(abs, found), nil
	}

	if in.Line < 1 {
		return "", fmt.Errorf("line is required for %s (1-based)", in.Operation)
	}
	col, err := columnOf(content, in.Line, in.Symbol)
	if err != nil {
		return "", err
	}

	var locs []lsp.Location
	switch in.Operation {
	case "definition":
		locs, err = client.Definition(ctx, abs, content, in.Line, col)
	case "references":
		locs, err = client.References(ctx, abs, content, in.Line, col)
	default:
		return "", fmt.Errorf("unknown operation %q: use definition, references or diagnostics", in.Operation)
	}
	if err != nil {
		return "", err
	}
	return t.renderLocations(in.Operation, in.Symbol, locs), nil
}

// columnOf finds where on a line the symbol sits.
//
// Asking the model for a column would be asking it to count characters in text
// it has only read, which it gets wrong often enough to make the whole tool
// untrustworthy. Naming the symbol is something it can always do correctly.
func columnOf(content string, line int, symbol string) (int, error) {
	lines := strings.Split(content, "\n")
	if line > len(lines) {
		return 0, fmt.Errorf("line %d is past the end of the file (%d lines)", line, len(lines))
	}
	text := lines[line-1]
	if symbol == "" {
		// No symbol named: the first non-space character is the best guess
		// available, and it is right for a declaration.
		for i, r := range text {
			if r != ' ' && r != '\t' {
				return i + 1, nil
			}
		}
		return 1, nil
	}
	i := strings.Index(text, symbol)
	if i < 0 {
		return 0, fmt.Errorf("%q does not appear on line %d: %q", symbol, line, strings.TrimSpace(text))
	}
	return i + 1, nil
}

func (t LSP) renderLocations(op, symbol string, locs []lsp.Location) string {
	label := symbol
	if label == "" {
		label = "that symbol"
	}
	if len(locs) == 0 {
		if op == "definition" {
			return fmt.Sprintf("No definition found for %s.", label)
		}
		return fmt.Sprintf("No references found for %s.", label)
	}

	// Servers repeat a location when a definition is also a reference; the
	// duplicate says nothing and costs a line.
	seen := map[string]bool{}
	var rows []string
	for _, l := range locs {
		row := fmt.Sprintf("%s:%d:%d", t.FS.display(l.Path), l.Line, l.Col)
		if seen[row] {
			continue
		}
		seen[row] = true
		rows = append(rows, row)
	}
	sort.Strings(rows)

	var b strings.Builder
	fmt.Fprintf(&b, "%d %s for %s:\n", len(rows), plural(len(rows), op), label)
	for i, row := range rows {
		if i == maxLSPResults {
			fmt.Fprintf(&b, "… and %d more\n", len(rows)-maxLSPResults)
			break
		}
		b.WriteString("  " + row + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

func plural(n int, op string) string {
	word := "reference"
	if op == "definition" {
		word = "definition"
	}
	if n == 1 {
		return word
	}
	return word + "s"
}

func (t LSP) renderDiagnostics(path string, found []lsp.Diagnostic) string {
	if len(found) == 0 {
		return fmt.Sprintf("No problems reported in %s.", t.FS.display(path))
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%d problem(s) in %s:\n", len(found), t.FS.display(path))
	for i, d := range found {
		if i == maxLSPResults {
			fmt.Fprintf(&b, "… and %d more\n", len(found)-maxLSPResults)
			break
		}
		fmt.Fprintf(&b, "  %d:%d: %s: %s\n", d.Line, d.Col, d.Severity, d.Message)
	}
	return strings.TrimRight(b.String(), "\n")
}
