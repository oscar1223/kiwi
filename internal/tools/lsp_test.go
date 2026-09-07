package tools

import (
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/lsp"
)

func lspTool(dir string) LSP {
	return LSP{FS: &FS{WorkDir: dir}}
}

// Asking the model for a column would be asking it to count characters in text
// it has only read. Naming the symbol is something it can always do correctly.
func TestColumnOfFindsTheSymbolOnTheLine(t *testing.T) {
	content := "package main\n\nfunc main() {\n\tresult := compute(x)\n}\n"

	col, err := columnOf(content, 4, "compute")
	if err != nil {
		t.Fatalf("columnOf: %v", err)
	}
	// The tab counts as one character, so compute starts at column 12.
	if got := content[strings.Index(content, "\tresult")+col-1:]; !strings.HasPrefix(got, "compute") {
		t.Errorf("column %d does not land on the symbol: %q", col, got[:10])
	}
}

func TestColumnOfFallsBackToTheFirstRealCharacter(t *testing.T) {
	col, err := columnOf("\t\tindented := 1\n", 1, "")
	if err != nil {
		t.Fatal(err)
	}
	if col != 3 {
		t.Errorf("column = %d, want the first non-space character at 3", col)
	}
}

func TestColumnOfReportsASymbolThatIsNotThere(t *testing.T) {
	_, err := columnOf("func main() {}\n", 1, "compute")
	if err == nil {
		t.Fatal("a symbol that is not on the line was accepted")
	}
	// The error shows the line, so the model can correct itself rather than
	// guess again.
	if !strings.Contains(err.Error(), "func main()") {
		t.Errorf("the error does not show the line: %v", err)
	}
}

func TestColumnOfReportsALineOffTheEnd(t *testing.T) {
	if _, err := columnOf("one\ntwo\n", 99, "x"); err == nil {
		t.Error("a line past the end of the file was accepted")
	}
}

func TestRenderLocationsDropsDuplicates(t *testing.T) {
	dir := t.TempDir()
	tool := lspTool(dir)

	// A server reports the definition as a reference too, so the same place
	// arrives twice. Repeating it says nothing and costs a line.
	got := tool.renderLocations("references", "Compute", []lsp.Location{
		{Path: dir + "/a.go", Line: 10, Col: 6},
		{Path: dir + "/b.go", Line: 3, Col: 1},
		{Path: dir + "/a.go", Line: 10, Col: 6},
	})

	if strings.Count(got, "a.go:10:6") != 1 {
		t.Errorf("the duplicate location was kept:\n%s", got)
	}
	if !strings.HasPrefix(got, "2 references") {
		t.Errorf("the count does not match what is listed:\n%s", got)
	}
}

func TestRenderLocationsCapsALongList(t *testing.T) {
	dir := t.TempDir()
	var locs []lsp.Location
	for i := 1; i <= maxLSPResults+7; i++ {
		locs = append(locs, lsp.Location{Path: dir + "/a.go", Line: i, Col: 1})
	}

	got := lspTool(dir).renderLocations("references", "Widely", locs)
	if strings.Count(got, "a.go:") != maxLSPResults {
		t.Errorf("listed %d locations, want %d", strings.Count(got, "a.go:"), maxLSPResults)
	}
	if !strings.Contains(got, "and 7 more") {
		t.Errorf("the report does not say how many were left out:\n%s", got)
	}
}

func TestRenderLocationsSaysNothingFoundPlainly(t *testing.T) {
	tool := lspTool(t.TempDir())
	if got := tool.renderLocations("definition", "Missing", nil); !strings.Contains(got, "No definition") {
		t.Errorf("empty definition result = %q", got)
	}
	if got := tool.renderLocations("references", "Missing", nil); !strings.Contains(got, "No references") {
		t.Errorf("empty references result = %q", got)
	}
	// With no symbol named, the message still has to read as a sentence.
	if got := tool.renderLocations("definition", "", nil); !strings.Contains(got, "that symbol") {
		t.Errorf("unnamed symbol result = %q", got)
	}
}

func TestRenderDiagnostics(t *testing.T) {
	dir := t.TempDir()
	tool := lspTool(dir)

	if got := tool.renderDiagnostics(dir+"/a.go", nil); !strings.Contains(got, "No problems") {
		t.Errorf("a clean file reported %q", got)
	}

	got := tool.renderDiagnostics(dir+"/a.go", []lsp.Diagnostic{
		{Line: 4, Col: 2, Severity: "error", Message: "undefined: Foo"},
	})
	for _, want := range []string{"a.go", "4:2", "error", "undefined: Foo"} {
		if !strings.Contains(got, want) {
			t.Errorf("the report is missing %q:\n%s", want, got)
		}
	}
}

func TestPlural(t *testing.T) {
	cases := []struct {
		n    int
		op   string
		want string
	}{
		{1, "definition", "definition"},
		{2, "definition", "definitions"},
		{1, "references", "reference"},
		{0, "references", "references"},
	}
	for _, c := range cases {
		if got := plural(c.n, c.op); got != c.want {
			t.Errorf("plural(%d, %q) = %q, want %q", c.n, c.op, got, c.want)
		}
	}
}

func TestLSPSchemaOffersOnlyTheThreeOperations(t *testing.T) {
	schema := LSP{}.Schema()
	props, _ := schema["properties"].(map[string]any)
	op, _ := props["operation"].(map[string]any)
	enum, _ := op["enum"].([]string)

	if strings.Join(enum, ",") != "definition,references,diagnostics" {
		t.Errorf("operations = %v, want the three the client implements", enum)
	}
}

// A nil manager must not panic while the description is being assembled: the
// tool is not registered in that case, but Description is cheap to reach.
func TestDescriptionSurvivesANilManager(t *testing.T) {
	if got := (LSP{}).Description(); !strings.Contains(got, "definition") {
		t.Errorf("description = %q", got)
	}
}
