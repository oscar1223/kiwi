package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/oscar1223/kiwi/internal/permission"
)

// newHTTPFakeServer starts a real HTTP server (a genuine TCP round trip, not
// an in-memory pipe) speaking the current Streamable HTTP MCP transport, with
// one "echo" tool — so buildTransport's HTTP path is exercised against an
// actual network connection, the same way it will be for a real remote
// server, not just unit-tested in isolation.
func newHTTPFakeServer(t *testing.T) *httptest.Server {
	t.Helper()
	getServer := func(*http.Request) *mcpsdk.Server {
		s := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "fake-http-server", Version: "0.0.1"}, nil)
		mcpsdk.AddTool(s, &mcpsdk.Tool{
			Name:        "echo",
			Description: "Echoes the input back.",
		}, func(ctx context.Context, req *mcpsdk.CallToolRequest, in echoInput) (*mcpsdk.CallToolResult, any, error) {
			return &mcpsdk.CallToolResult{
				Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: in.Text}},
			}, nil, nil
		})
		return s
	}
	handler := mcpsdk.NewStreamableHTTPHandler(getServer, nil)
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

func TestBuildTransportHTTPEndToEnd(t *testing.T) {
	srv := newHTTPFakeServer(t)
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	sc := ServerConfig{URL: srv.URL}
	if err := sc.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "kiwi-test", Version: "0.0.1"}, nil)
	session, tools, err := connectOne(ctx, client, "http-fake", buildTransport(ctx, sc), broker)
	if err != nil {
		t.Fatalf("connectOne: %v", err)
	}
	defer session.Close()

	if len(tools) != 1 || tools[0].Name() != "echo" {
		t.Fatalf("tools = %+v", tools)
	}

	input, _ := json.Marshal(echoInput{Text: "hello over real http"})
	out, err := tools[0].Run(ctx, input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "hello over real http" {
		t.Errorf("out = %q", out)
	}
}

// Authorization (or any fixed header) must actually reach the server — this
// is what a remote MCP server behind a token typically requires.
func TestBuildTransportSendsConfiguredHeaders(t *testing.T) {
	var gotAuth string
	inner := func(*http.Request) *mcpsdk.Server {
		s := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "auth-server", Version: "0.0.1"}, nil)
		return s
	}
	handler := mcpsdk.NewStreamableHTTPHandler(inner, nil)
	wrapped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		handler.ServeHTTP(w, r)
	})
	srv := httptest.NewServer(wrapped)
	t.Cleanup(srv.Close)

	sc := ServerConfig{URL: srv.URL, Headers: map[string]string{"Authorization": "Bearer test-token"}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "kiwi-test", Version: "0.0.1"}, nil)
	session, _, err := connectOne(ctx, client, "auth-fake", buildTransport(ctx, sc), broker)
	if err != nil {
		t.Fatalf("connectOne: %v", err)
	}
	defer session.Close()

	if gotAuth != "Bearer test-token" {
		t.Errorf("Authorization header = %q, want %q", gotAuth, "Bearer test-token")
	}
}

func TestBuildTransportPicksSSEWhenTypeIsSSE(t *testing.T) {
	sc := ServerConfig{URL: "http://example.invalid", Type: TransportSSE}
	transport := buildTransport(context.Background(), sc)
	if _, ok := transport.(*mcpsdk.SSEClientTransport); !ok {
		t.Errorf("transport = %T, want *mcpsdk.SSEClientTransport", transport)
	}
}

func TestBuildTransportDefaultsToStreamableHTTP(t *testing.T) {
	sc := ServerConfig{URL: "http://example.invalid"}
	transport := buildTransport(context.Background(), sc)
	if _, ok := transport.(*mcpsdk.StreamableClientTransport); !ok {
		t.Errorf("transport = %T, want *mcpsdk.StreamableClientTransport", transport)
	}
}

func TestBuildTransportPicksCommandForStdio(t *testing.T) {
	sc := ServerConfig{Command: "echo"}
	transport := buildTransport(context.Background(), sc)
	if _, ok := transport.(*mcpsdk.CommandTransport); !ok {
		t.Errorf("transport = %T, want *mcpsdk.CommandTransport", transport)
	}
}
