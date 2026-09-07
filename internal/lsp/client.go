// Package lsp talks to language servers over stdio, so Kiwi can answer
// questions about code the way an editor does rather than by reading files and
// guessing.
//
// The scope is deliberately three operations — where is this defined, what
// uses it, and what is wrong with this file. Hover, rename and completion are
// the ones that cost the most code and give an agent the least: a model that
// can already read the whole file does not need a type signature summarised
// for it, but it cannot grep its way to "every caller of this method" without
// getting import cycles and shadowed names wrong.
package lsp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// startTimeout bounds the initialize handshake. A server that has not answered
// by then is one the user is better off being told about than waiting on.
const startTimeout = 30 * time.Second

// requestTimeout bounds one query.
const requestTimeout = 15 * time.Second

// diagnosticsWait is how long to wait for a server to publish its verdict on a
// freshly opened file. Diagnostics are pushed, not requested, so there is no
// response to block on — only a deadline to give up at.
const diagnosticsWait = 3 * time.Second

// Location is one place in the code.
type Location struct {
	Path string
	Line int
	Col  int
}

// Diagnostic is one problem a server reports.
type Diagnostic struct {
	Line     int
	Col      int
	Severity string
	Message  string
}

// Client is one running language server.
type Client struct {
	name string
	root string

	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader

	mu      sync.Mutex
	nextID  int
	pending map[int]chan rpcResponse

	diagMu sync.Mutex
	// diags is the latest verdict per document, and waiters are the callers
	// currently blocked on one arriving.
	diags   map[string][]Diagnostic
	waiters map[string][]chan struct{}
	// opened tracks which documents the server has been told about, so a
	// second question about the same file does not re-open it.
	opened map[string]bool

	closeOnce sync.Once
	done      chan struct{}
}

type rpcResponse struct {
	Result json.RawMessage
	Err    error
}

// Start launches a server and completes the initialize handshake.
func Start(ctx context.Context, name string, argv []string, root string) (*Client, error) {
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Dir = root
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	// The server's own logging is discarded rather than inherited: a language
	// server writing progress to stderr would land in the middle of the TUI's
	// alt screen and corrupt the frame.
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("lsp: starting %s: %w", name, err)
	}

	c := &Client{
		name:    name,
		root:    root,
		cmd:     cmd,
		stdin:   stdin,
		stdout:  bufio.NewReaderSize(stdout, 1<<16),
		pending: map[int]chan rpcResponse{},
		diags:   map[string][]Diagnostic{},
		waiters: map[string][]chan struct{}{},
		opened:  map[string]bool{},
		done:    make(chan struct{}),
	}
	go c.read()

	initCtx, cancel := context.WithTimeout(ctx, startTimeout)
	defer cancel()
	if err := c.initialize(initCtx); err != nil {
		c.Close()
		return nil, err
	}
	return c, nil
}

// Name is the server's identifier, for messages to the user.
func (c *Client) Name() string { return c.name }

func (c *Client) initialize(ctx context.Context) error {
	params := map[string]any{
		"processId": nil,
		"rootUri":   pathToURI(c.root),
		"workspaceFolders": []map[string]any{
			{"uri": pathToURI(c.root), "name": filepath.Base(c.root)},
		},
		// Only what is actually used is advertised. A client that claims
		// capabilities it does not implement gets sent requests it will never
		// answer, and some servers then wait on them.
		"capabilities": map[string]any{
			"textDocument": map[string]any{
				"synchronization": map[string]any{"didSave": false, "willSave": false},
				"definition":      map[string]any{"linkSupport": false},
				"references":      map[string]any{},
				"publishDiagnostics": map[string]any{
					"relatedInformation": false,
				},
			},
			"workspace": map[string]any{"workspaceFolders": true},
		},
	}
	if _, err := c.call(ctx, "initialize", params); err != nil {
		return err
	}
	return c.notify("initialized", map[string]any{})
}

// read pumps the server's output, matching responses to their waiting callers
// and filing diagnostics as they are pushed.
func (c *Client) read() {
	defer close(c.done)
	for {
		body, err := readMessage(c.stdout)
		if err != nil {
			c.failPending(err)
			return
		}
		var msg struct {
			ID     *int            `json:"id"`
			Method string          `json:"method"`
			Result json.RawMessage `json:"result"`
			Params json.RawMessage `json:"params"`
			Error  *struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		if json.Unmarshal(body, &msg) != nil {
			continue
		}

		switch {
		case msg.ID != nil && msg.Method == "":
			resp := rpcResponse{Result: msg.Result}
			if msg.Error != nil {
				resp.Err = fmt.Errorf("lsp: %s: %s", c.name, msg.Error.Message)
			}
			c.mu.Lock()
			ch, ok := c.pending[*msg.ID]
			delete(c.pending, *msg.ID)
			c.mu.Unlock()
			if ok {
				ch <- resp
			}
		case msg.Method == "textDocument/publishDiagnostics":
			c.onDiagnostics(msg.Params)
		case msg.ID != nil:
			// A server request. Nothing here can answer one, but leaving it
			// unanswered makes some servers wait forever, so it is refused
			// promptly instead.
			c.respondUnsupported(*msg.ID)
		}
	}
}

func (c *Client) onDiagnostics(params json.RawMessage) {
	var p struct {
		URI         string `json:"uri"`
		Diagnostics []struct {
			Range struct {
				Start struct {
					Line      int `json:"line"`
					Character int `json:"character"`
				} `json:"start"`
			} `json:"range"`
			Severity int    `json:"severity"`
			Message  string `json:"message"`
		} `json:"diagnostics"`
	}
	if json.Unmarshal(params, &p) != nil {
		return
	}
	out := make([]Diagnostic, 0, len(p.Diagnostics))
	for _, d := range p.Diagnostics {
		out = append(out, Diagnostic{
			// LSP counts from zero; everything a person or a model reads
			// counts from one, and the conversion belongs here rather than in
			// every caller.
			Line:     d.Range.Start.Line + 1,
			Col:      d.Range.Start.Character + 1,
			Severity: severityName(d.Severity),
			Message:  d.Message,
		})
	}

	c.diagMu.Lock()
	c.diags[p.URI] = out
	waiting := c.waiters[p.URI]
	delete(c.waiters, p.URI)
	c.diagMu.Unlock()
	for _, ch := range waiting {
		close(ch)
	}
}

func severityName(n int) string {
	switch n {
	case 1:
		return "error"
	case 2:
		return "warning"
	case 3:
		return "info"
	case 4:
		return "hint"
	}
	return "info"
}

func (c *Client) failPending(err error) {
	c.mu.Lock()
	pending := c.pending
	c.pending = map[int]chan rpcResponse{}
	c.mu.Unlock()
	for _, ch := range pending {
		ch <- rpcResponse{Err: fmt.Errorf("lsp: %s stopped: %w", c.name, err)}
	}
}

func (c *Client) respondUnsupported(id int) {
	c.write(map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"error":   map[string]any{"code": -32601, "message": "not supported"},
	})
}

// call sends a request and waits for its response.
func (c *Client) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	c.mu.Lock()
	c.nextID++
	id := c.nextID
	ch := make(chan rpcResponse, 1)
	c.pending[id] = ch
	c.mu.Unlock()

	if err := c.write(map[string]any{
		"jsonrpc": "2.0", "id": id, "method": method, "params": params,
	}); err != nil {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		return nil, err
	}

	select {
	case resp := <-ch:
		return resp.Result, resp.Err
	case <-ctx.Done():
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
		return nil, ctx.Err()
	case <-c.done:
		return nil, fmt.Errorf("lsp: %s exited", c.name)
	}
}

func (c *Client) notify(method string, params any) error {
	return c.write(map[string]any{"jsonrpc": "2.0", "method": method, "params": params})
}

func (c *Client) write(msg any) error {
	body, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	_, err = fmt.Fprintf(c.stdin, "Content-Length: %d\r\n\r\n%s", len(body), body)
	return err
}

// readMessage reads one framed JSON-RPC message.
func readMessage(r *bufio.Reader) ([]byte, error) {
	var length int
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if name, value, ok := strings.Cut(line, ":"); ok &&
			strings.EqualFold(strings.TrimSpace(name), "Content-Length") {
			fmt.Sscanf(strings.TrimSpace(value), "%d", &length)
		}
	}
	if length <= 0 {
		return nil, fmt.Errorf("lsp: message with no Content-Length")
	}
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, err
	}
	return body, nil
}

// open tells the server about a file, once.
//
// Servers answer questions about documents the client has opened, not about
// whatever happens to be on disk, so this is a precondition for every query
// rather than an optimisation.
func (c *Client) open(ctx context.Context, path string, content string) error {
	uri := pathToURI(path)
	c.diagMu.Lock()
	already := c.opened[uri]
	c.opened[uri] = true
	c.diagMu.Unlock()
	if already {
		// Already open: tell the server the contents changed instead, so a
		// query after an edit is answered about the current file rather than
		// the one opened three tool calls ago.
		return c.notify("textDocument/didChange", map[string]any{
			"textDocument": map[string]any{"uri": uri, "version": time.Now().UnixNano()},
			"contentChanges": []map[string]any{
				{"text": content},
			},
		})
	}
	return c.notify("textDocument/didOpen", map[string]any{
		"textDocument": map[string]any{
			"uri":        uri,
			"languageId": languageID(path),
			"version":    1,
			"text":       content,
		},
	})
}

// Definition reports where the symbol at a position is defined.
func (c *Client) Definition(ctx context.Context, path, content string, line, col int) ([]Location, error) {
	return c.locations(ctx, "textDocument/definition", path, content, line, col, nil)
}

// References reports everywhere the symbol at a position is used.
func (c *Client) References(ctx context.Context, path, content string, line, col int) ([]Location, error) {
	return c.locations(ctx, "textDocument/references", path, content, line, col,
		map[string]any{"context": map[string]any{"includeDeclaration": true}})
}

func (c *Client) locations(ctx context.Context, method, path, content string, line, col int, extra map[string]any) ([]Location, error) {
	if err := c.open(ctx, path, content); err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, requestTimeout)
	defer cancel()

	params := map[string]any{
		"textDocument": map[string]any{"uri": pathToURI(path)},
		"position":     map[string]any{"line": line - 1, "character": col - 1},
	}
	for k, v := range extra {
		params[k] = v
	}
	raw, err := c.call(ctx, method, params)
	if err != nil {
		return nil, err
	}
	return parseLocations(raw)
}

// parseLocations copes with the three shapes a server may answer with: a
// single Location, an array of them, or an array of LocationLinks. All three
// are legal, and which one arrives depends on the server rather than on
// anything the caller did.
func parseLocations(raw json.RawMessage) ([]Location, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	type rng struct {
		Start struct {
			Line      int `json:"line"`
			Character int `json:"character"`
		} `json:"start"`
	}
	type entry struct {
		URI          string `json:"uri"`
		TargetURI    string `json:"targetUri"`
		Range        rng    `json:"range"`
		TargetRange  rng    `json:"targetSelectionRange"`
		TargetRange2 rng    `json:"targetRange"`
	}

	var many []entry
	if err := json.Unmarshal(raw, &many); err != nil {
		var one entry
		if err := json.Unmarshal(raw, &one); err != nil {
			return nil, fmt.Errorf("lsp: unexpected response shape")
		}
		many = []entry{one}
	}

	out := make([]Location, 0, len(many))
	for _, e := range many {
		uri, r := e.URI, e.Range
		if uri == "" {
			uri, r = e.TargetURI, e.TargetRange
			if r.Start.Line == 0 && r.Start.Character == 0 {
				r = e.TargetRange2
			}
		}
		if uri == "" {
			continue
		}
		out = append(out, Location{
			Path: uriToPath(uri),
			Line: r.Start.Line + 1,
			Col:  r.Start.Character + 1,
		})
	}
	return out, nil
}

// Diagnostics reports what the server thinks is wrong with a file.
func (c *Client) Diagnostics(ctx context.Context, path, content string) ([]Diagnostic, error) {
	uri := pathToURI(path)

	c.diagMu.Lock()
	// The wait is registered before the file is opened. Registering after
	// would leave a window in which a fast server publishes its verdict
	// before anyone is listening, and the caller then waits out the full
	// deadline for a message that already arrived.
	wait := make(chan struct{})
	c.waiters[uri] = append(c.waiters[uri], wait)
	c.diagMu.Unlock()

	if err := c.open(ctx, path, content); err != nil {
		return nil, err
	}

	select {
	case <-wait:
	case <-time.After(diagnosticsWait):
		// Silence is the normal answer for a clean file: many servers publish
		// nothing at all when there is nothing to say.
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-c.done:
		return nil, fmt.Errorf("lsp: %s exited", c.name)
	}

	c.diagMu.Lock()
	defer c.diagMu.Unlock()
	return append([]Diagnostic(nil), c.diags[uri]...), nil
}

// Close shuts the server down.
func (c *Client) Close() {
	c.closeOnce.Do(func() {
		// The polite sequence first, then the process. A server killed
		// mid-write can leave a stale lock or an index it will rebuild from
		// scratch next time.
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		_, _ = c.call(ctx, "shutdown", nil)
		_ = c.notify("exit", nil)
		cancel()

		_ = c.stdin.Close()
		select {
		case <-c.done:
		case <-time.After(2 * time.Second):
		}
		if c.cmd.Process != nil {
			_ = c.cmd.Process.Kill()
		}
		_ = c.cmd.Wait()
	})
}

// pathToURI converts an absolute path to a file URI.
func pathToURI(path string) string {
	path = filepath.ToSlash(path)
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	u := url.URL{Scheme: "file", Path: path}
	return u.String()
}

// uriToPath converts a file URI back to a path, leaving anything else alone.
func uriToPath(uri string) string {
	u, err := url.Parse(uri)
	if err != nil || u.Scheme != "file" {
		return uri
	}
	return filepath.FromSlash(u.Path)
}

// languageID maps a file to the identifier servers expect in didOpen.
func languageID(path string) string {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		return "go"
	case ".ts":
		return "typescript"
	case ".tsx":
		return "typescriptreact"
	case ".js", ".mjs", ".cjs":
		return "javascript"
	case ".jsx":
		return "javascriptreact"
	case ".py":
		return "python"
	case ".rs":
		return "rust"
	}
	return strings.TrimPrefix(strings.ToLower(filepath.Ext(path)), ".")
}
