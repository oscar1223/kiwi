package lsp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The tests drive a real server over a real pipe, because the parts most
// likely to be wrong here — message framing, matching a response to its
// caller, and diagnostics that arrive unasked — only exist between two
// processes. The server is this same test binary re-executed in server mode,
// which is the cheapest way to get one that actually speaks the protocol.
const helperEnv = "KIWI_LSP_TEST_SERVER"

func TestMain(m *testing.M) {
	if os.Getenv(helperEnv) == "" {
		os.Exit(m.Run())
	}
	fakeServer()
	os.Exit(0)
}

// fakeServer answers the handful of requests the client actually makes.
func fakeServer() {
	in := bufio.NewReaderSize(os.Stdin, 1<<16)
	send := func(msg any) {
		body, _ := json.Marshal(msg)
		fmt.Fprintf(os.Stdout, "Content-Length: %d\r\n\r\n%s", len(body), body)
	}
	loc := func(uri string, line, col int) map[string]any {
		return map[string]any{
			"uri": uri,
			"range": map[string]any{
				"start": map[string]any{"line": line, "character": col},
				"end":   map[string]any{"line": line, "character": col + 3},
			},
		}
	}

	for {
		body, err := readMessage(in)
		if err != nil {
			return
		}
		var msg struct {
			ID     *int   `json:"id"`
			Method string `json:"method"`
			Params struct {
				TextDocument struct {
					URI string `json:"uri"`
				} `json:"textDocument"`
			} `json:"params"`
		}
		if json.Unmarshal(body, &msg) != nil {
			continue
		}
		uri := msg.Params.TextDocument.URI

		switch msg.Method {
		case "initialize":
			send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": map[string]any{
				"capabilities": map[string]any{},
			}})
		case "textDocument/didOpen", "textDocument/didChange":
			send(map[string]any{"jsonrpc": "2.0", "method": "textDocument/publishDiagnostics",
				"params": map[string]any{
					"uri": uri,
					"diagnostics": []map[string]any{{
						"range":    map[string]any{"start": map[string]any{"line": 3, "character": 1}},
						"severity": 1,
						"message":  "undefined: Foo",
					}},
				}})
		case "textDocument/definition":
			send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": loc(uri, 9, 5)})
		case "textDocument/references":
			// The same location twice, which is what a server does when a
			// definition is also counted as a reference.
			send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": []map[string]any{
				loc(uri, 9, 5), loc(uri, 20, 2), loc(uri, 9, 5),
			}})
		case "shutdown":
			send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": nil})
		case "exit":
			return
		default:
			if msg.ID != nil {
				send(map[string]any{"jsonrpc": "2.0", "id": msg.ID, "result": nil})
			}
		}
	}
}

func startFake(t *testing.T) (*Client, string) {
	t.Helper()
	dir := t.TempDir()
	file := filepath.Join(dir, "main.go")
	if err := os.WriteFile(file, []byte("package main\n\nfunc main() {}\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	self, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(helperEnv, "1")

	c, err := Start(context.Background(), "fake", []string{self}, dir)
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	t.Cleanup(c.Close)
	return c, file
}

func TestDefinitionResolvesToAPlaceInTheCode(t *testing.T) {
	c, file := startFake(t)

	locs, err := c.Definition(context.Background(), file, "package main\n", 3, 6)
	if err != nil {
		t.Fatalf("Definition: %v", err)
	}
	if len(locs) != 1 {
		t.Fatalf("got %d locations, want 1: %+v", len(locs), locs)
	}
	// The protocol counts from zero and everyone reading the answer counts
	// from one; the conversion has to happen exactly once.
	if locs[0].Line != 10 || locs[0].Col != 6 {
		t.Errorf("location = %+v, want line 10 col 6", locs[0])
	}
	if locs[0].Path != file {
		t.Errorf("path = %q, want %q", locs[0].Path, file)
	}
}

func TestReferencesReturnsEveryUse(t *testing.T) {
	c, file := startFake(t)

	locs, err := c.References(context.Background(), file, "package main\n", 3, 6)
	if err != nil {
		t.Fatalf("References: %v", err)
	}
	if len(locs) != 3 {
		t.Fatalf("got %d locations, want the server's three: %+v", len(locs), locs)
	}
}

func TestDiagnosticsArriveWithoutBeingAskedFor(t *testing.T) {
	c, file := startFake(t)

	found, err := c.Diagnostics(context.Background(), file, "package main\n")
	if err != nil {
		t.Fatalf("Diagnostics: %v", err)
	}
	if len(found) != 1 {
		t.Fatalf("got %d diagnostics, want 1: %+v", len(found), found)
	}
	if found[0].Line != 4 || found[0].Severity != "error" {
		t.Errorf("diagnostic = %+v, want line 4 and severity error", found[0])
	}
	if found[0].Message != "undefined: Foo" {
		t.Errorf("message = %q", found[0].Message)
	}
}

// The second question about a file must be answered about the file as it is
// now, not as it was when it was first opened three tool calls ago.
func TestAskingTwiceReopensTheDocumentAsChanged(t *testing.T) {
	c, file := startFake(t)

	if _, err := c.Diagnostics(context.Background(), file, "first"); err != nil {
		t.Fatal(err)
	}
	found, err := c.Diagnostics(context.Background(), file, "second")
	if err != nil {
		t.Fatalf("the second question failed: %v", err)
	}
	if len(found) != 1 {
		t.Errorf("the second question got %d diagnostics, want 1", len(found))
	}
}

func TestRequestsAreMatchedToTheirOwnCallers(t *testing.T) {
	c, file := startFake(t)

	// Concurrent questions of different kinds: the reader dispatches by id,
	// and a mix-up would show up as one caller getting the other's answer.
	type result struct {
		locs []Location
		err  error
	}
	defs := make(chan result, 1)
	refs := make(chan result, 1)
	go func() {
		l, err := c.Definition(context.Background(), file, "x", 3, 6)
		defs <- result{l, err}
	}()
	go func() {
		l, err := c.References(context.Background(), file, "x", 3, 6)
		refs <- result{l, err}
	}()

	d, r := <-defs, <-refs
	if d.err != nil || r.err != nil {
		t.Fatalf("definition: %v, references: %v", d.err, r.err)
	}
	if len(d.locs) != 1 {
		t.Errorf("definition got %d locations, want 1", len(d.locs))
	}
	if len(r.locs) != 3 {
		t.Errorf("references got %d locations, want 3", len(r.locs))
	}
}

func TestCloseIsSafeToCallTwice(t *testing.T) {
	c, _ := startFake(t)
	c.Close()
	c.Close()
}

func TestAQuestionAfterTheServerDiesFailsRatherThanHangs(t *testing.T) {
	c, file := startFake(t)
	c.Close()

	done := make(chan error, 1)
	go func() {
		_, err := c.Definition(context.Background(), file, "x", 3, 6)
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Error("a question to a dead server succeeded")
		}
	case <-time.After(10 * time.Second):
		t.Error("a question to a dead server hung")
	}
}

func TestStartReportsAServerThatIsNotThere(t *testing.T) {
	_, err := Start(context.Background(), "nope", []string{"definitely-not-a-server-4c1f"}, t.TempDir())
	if err == nil {
		t.Fatal("starting a missing binary succeeded")
	}
	if !strings.Contains(err.Error(), "nope") {
		t.Errorf("the error does not name the server: %v", err)
	}
}

func TestParseLocationsHandlesEveryShapeAServerMayAnswerWith(t *testing.T) {
	cases := map[string]string{
		"single Location":  `{"uri":"file:///a.go","range":{"start":{"line":1,"character":2}}}`,
		"array":            `[{"uri":"file:///a.go","range":{"start":{"line":1,"character":2}}}]`,
		"LocationLink":     `[{"targetUri":"file:///a.go","targetSelectionRange":{"start":{"line":1,"character":2}}}]`,
		"LocationLink alt": `[{"targetUri":"file:///a.go","targetRange":{"start":{"line":1,"character":2}}}]`,
	}
	for name, raw := range cases {
		got, err := parseLocations(json.RawMessage(raw))
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if len(got) != 1 {
			t.Errorf("%s: got %d locations, want 1", name, len(got))
			continue
		}
		if got[0].Line != 2 || got[0].Col != 3 {
			t.Errorf("%s: location = %+v, want line 2 col 3", name, got[0])
		}
	}

	for _, empty := range []string{"null", "[]", ""} {
		if got, err := parseLocations(json.RawMessage(empty)); err != nil || len(got) != 0 {
			t.Errorf("parseLocations(%q) = %+v, %v; want nothing", empty, got, err)
		}
	}
}

func TestURIRoundTrip(t *testing.T) {
	for _, path := range []string{"/tmp/a.go", "/tmp/with space/b.go", "/tmp/percent%.go"} {
		if got := uriToPath(pathToURI(path)); got != path {
			t.Errorf("round trip of %q = %q", path, got)
		}
	}
	// Anything that is not a file URI comes back untouched rather than
	// mangled into a path that does not exist.
	if got := uriToPath("untitled:Untitled-1"); got != "untitled:Untitled-1" {
		t.Errorf("uriToPath on a non-file URI = %q", got)
	}
}

func TestLanguageID(t *testing.T) {
	cases := map[string]string{
		"a.go": "go", "a.ts": "typescript", "a.tsx": "typescriptreact",
		"a.js": "javascript", "a.py": "python", "a.rs": "rust", "a.zig": "zig",
	}
	for file, want := range cases {
		if got := languageID(file); got != want {
			t.Errorf("languageID(%q) = %q, want %q", file, got, want)
		}
	}
}

func TestReadMessageRejectsAnUnframedPayload(t *testing.T) {
	r := bufio.NewReader(strings.NewReader("no headers here\r\n\r\n{}"))
	if _, err := readMessage(r); err == nil {
		t.Error("a message with no Content-Length was accepted")
	}
}

func TestSeverityName(t *testing.T) {
	for n, want := range map[int]string{1: "error", 2: "warning", 3: "info", 4: "hint", 9: "info"} {
		if got := severityName(n); got != want {
			t.Errorf("severityName(%d) = %q, want %q", n, got, want)
		}
	}
}
