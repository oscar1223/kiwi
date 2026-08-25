package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Tool wraps one tool exposed by an MCP server so it satisfies the same
// interface every built-in Kiwi tool does.
type Tool struct {
	session     *mcpsdk.ClientSession
	serverName  string
	name        string
	description string
	schema      map[string]any
	perms       *permission.Broker
}

func (t *Tool) Name() string           { return t.name }
func (t *Tool) Description() string    { return t.description }
func (t *Tool) Schema() map[string]any { return t.schema }

func (t *Tool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var args map[string]any
	if len(input) > 0 {
		if err := json.Unmarshal(input, &args); err != nil {
			return "", err
		}
	}

	if err := t.perms.Ask(ctx, permission.Action{
		Name:   permission.MCPPrefix + t.name,
		Detail: fmt.Sprintf("%s (via MCP server %q)", t.name, t.serverName),
	}); err != nil {
		return "", err
	}

	res, err := t.session.CallTool(ctx, &mcpsdk.CallToolParams{Name: t.name, Arguments: args})
	if err != nil {
		return "", fmt.Errorf("mcp tool %q: %w", t.name, err)
	}

	var out strings.Builder
	for _, c := range res.Content {
		if tc, ok := c.(*mcpsdk.TextContent); ok {
			out.WriteString(tc.Text)
		}
	}
	if res.IsError {
		return "", errors.New(out.String())
	}
	return out.String(), nil
}

// Manager owns the live sessions Connect opened, so they can all be closed
// together when Kiwi exits or reloads its MCP configuration.
type Manager struct {
	sessions []*mcpsdk.ClientSession
}

// Close disconnects every server. Safe to call on a Manager with no sessions.
func (m *Manager) Close() {
	for _, s := range m.sessions {
		s.Close()
	}
}

// Connect starts every configured server and gathers their tools.
//
// A server that fails to start, or fails to hand back its tool list, is
// skipped with its error collected rather than aborting the whole connect —
// one misconfigured server (a typo'd command, a missing API key) should
// never take down every other server, or Kiwi itself.
func Connect(ctx context.Context, broker *permission.Broker) (*Manager, []*Tool, []error) {
	cfg, err := LoadConfig()
	if err != nil {
		return &Manager{}, nil, []error{err}
	}

	m := &Manager{}
	var allTools []*Tool
	var errs []error

	for _, name := range sortedNames(cfg) {
		sc := cfg[name]
		if err := sc.Validate(); err != nil {
			errs = append(errs, fmt.Errorf("mcp server %q: %w", name, err))
			continue
		}

		transport := buildTransport(ctx, sc)
		client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "kiwi", Version: "0.1.0"}, nil)
		session, tools, err := connectOne(ctx, client, name, transport, broker)
		if session != nil {
			m.sessions = append(m.sessions, session)
		}
		if err != nil {
			errs = append(errs, err)
			continue
		}
		allTools = append(allTools, tools...)
	}
	return m, allTools, errs
}

// buildTransport picks stdio, Streamable HTTP, or SSE based on the config —
// sc has already passed Validate, so exactly one of Command/URL is set.
func buildTransport(ctx context.Context, sc ServerConfig) mcpsdk.Transport {
	if sc.Command != "" {
		cmd := exec.CommandContext(ctx, sc.Command, sc.Args...)
		cmd.Env = os.Environ()
		for k, v := range sc.Env {
			cmd.Env = append(cmd.Env, k+"="+v)
		}
		return &mcpsdk.CommandTransport{Command: cmd}
	}

	httpClient := &http.Client{Transport: headerTransport{headers: sc.Headers}}
	if sc.Type == TransportSSE {
		return &mcpsdk.SSEClientTransport{Endpoint: sc.URL, HTTPClient: httpClient}
	}
	return &mcpsdk.StreamableClientTransport{Endpoint: sc.URL, HTTPClient: httpClient}
}

// headerTransport injects fixed headers (typically Authorization) into every
// request — neither client transport type exposes a headers option directly,
// only an *http.Client, so this is the seam to add them at.
type headerTransport struct {
	headers map[string]string
}

func (h headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(h.headers) > 0 {
		req = req.Clone(req.Context())
		for k, v := range h.headers {
			req.Header.Set(k, v)
		}
	}
	base := http.DefaultTransport
	return base.RoundTrip(req)
}

// connectOne performs the handshake and tool listing against an already
// constructed transport. Split out from Connect so tests can substitute an
// mcpsdk.InMemoryTransport for a real subprocess and exercise the exact same
// handshake and tool-listing logic against a real (if fake) MCP server.
func connectOne(ctx context.Context, client *mcpsdk.Client, name string, t mcpsdk.Transport, broker *permission.Broker) (*mcpsdk.ClientSession, []*Tool, error) {
	session, err := client.Connect(ctx, t, nil)
	if err != nil {
		return nil, nil, fmt.Errorf("mcp server %q: %w", name, err)
	}

	var out []*Tool
	for tool, err := range session.Tools(ctx, nil) {
		if err != nil {
			return session, out, fmt.Errorf("mcp server %q: listing tools: %w", name, err)
		}
		schema, _ := tool.InputSchema.(map[string]any)
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}
		out = append(out, &Tool{
			session:     session,
			serverName:  name,
			name:        tool.Name,
			description: tool.Description,
			schema:      schema,
			perms:       broker,
		})
	}
	return session, out, nil
}
