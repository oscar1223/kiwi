package mcp

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	mcpsdk "github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/oscar1223/kiwi/internal/permission"
)

// echoInput is the argument shape for the fake server's one tool.
type echoInput struct {
	Text string `json:"text"`
}

// newFakeServer builds an in-process MCP server exposing a single "echo"
// tool, connected to a client-side transport over an in-memory pipe — a real
// handshake and a real tool call, with no subprocess involved.
func newFakeServer(t *testing.T) mcpsdk.Transport {
	t.Helper()
	server := mcpsdk.NewServer(&mcpsdk.Implementation{Name: "fake-server", Version: "0.0.1"}, nil)
	mcpsdk.AddTool(server, &mcpsdk.Tool{
		Name:        "echo",
		Description: "Echoes the input back.",
	}, func(ctx context.Context, req *mcpsdk.CallToolRequest, in echoInput) (*mcpsdk.CallToolResult, any, error) {
		return &mcpsdk.CallToolResult{
			Content: []mcpsdk.Content{&mcpsdk.TextContent{Text: in.Text}},
		}, nil, nil
	})

	serverSide, clientSide := mcpsdk.NewInMemoryTransports()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// The SDK requires the server side to connect first.
	if _, err := server.Connect(ctx, serverSide, nil); err != nil {
		t.Fatalf("server.Connect: %v", err)
	}
	return clientSide
}

func TestConnectOneListsToolsFromARealHandshake(t *testing.T) {
	transport := newFakeServer(t)
	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "kiwi-test", Version: "0.0.1"}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	session, tools, err := connectOne(ctx, client, "fake", transport, broker)
	if err != nil {
		t.Fatalf("connectOne: %v", err)
	}
	defer session.Close()

	if len(tools) != 1 {
		t.Fatalf("got %d tools, want 1", len(tools))
	}
	tool := tools[0]
	if tool.Name() != "echo" {
		t.Errorf("Name() = %q", tool.Name())
	}
	if tool.Description() != "Echoes the input back." {
		t.Errorf("Description() = %q", tool.Description())
	}
}

func TestMCPToolCallRoundTripsThroughARealServer(t *testing.T) {
	transport := newFakeServer(t)
	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "kiwi-test", Version: "0.0.1"}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	session, tools, err := connectOne(ctx, client, "fake", transport, broker)
	if err != nil {
		t.Fatalf("connectOne: %v", err)
	}
	defer session.Close()

	input, _ := json.Marshal(echoInput{Text: "hello from kiwi"})
	out, err := tools[0].Run(ctx, input)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "hello from kiwi" {
		t.Errorf("out = %q, want the echoed text", out)
	}
}

// The permission gate applies to MCP tools exactly like a built-in one — an
// MCP tool call in Plan mode must be refused without ever reaching the
// server.
func TestMCPToolRespectsThePermissionGate(t *testing.T) {
	transport := newFakeServer(t)
	broker := permission.NewBroker(permission.ModeAsk, permission.DenyAll{})
	client := mcpsdk.NewClient(&mcpsdk.Implementation{Name: "kiwi-test", Version: "0.0.1"}, nil)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	session, tools, err := connectOne(ctx, client, "fake", transport, broker)
	if err != nil {
		t.Fatalf("connectOne: %v", err)
	}
	defer session.Close()

	input, _ := json.Marshal(echoInput{Text: "should not run"})
	_, err = tools[0].Run(ctx, input)
	if err != permission.ErrDenied {
		t.Errorf("err = %v, want ErrDenied", err)
	}
}

// A server that fails to start (bad command, or one that exits immediately
// without speaking MCP) must be reported as an error and must not hang or
// take any other configured server down with it. This goes through Connect
// itself — not connectOne with a raw in-memory pipe — because the real
// failure mode here relies on exec.CommandContext actually tearing the
// process down, which is what unblocks the handshake promptly; a bare
// net.Pipe (as connectOne's other tests use) has no such OS-level teardown
// and would hang regardless of ctx, which is not representative of Connect's
// real behaviour.
func TestConnectSkipsAServerThatFailsToStart(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)

	if err := AddServer("broken", ServerConfig{Command: "sh", Args: []string{"-c", "exit 1"}}); err != nil {
		t.Fatalf("AddServer: %v", err)
	}

	broker := permission.NewBroker(permission.ModeWork, permission.AllowAll{})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	done := make(chan struct {
		m     *Manager
		tools []*Tool
		errs  []error
	}, 1)
	go func() {
		m, tools, errs := Connect(ctx, broker)
		done <- struct {
			m     *Manager
			tools []*Tool
			errs  []error
		}{m, tools, errs}
	}()

	select {
	case r := <-done:
		if len(r.errs) != 1 {
			t.Fatalf("got %d errors, want 1: %v", len(r.errs), r.errs)
		}
		if len(r.tools) != 0 {
			t.Errorf("got %d tools from a server that never started", len(r.tools))
		}
		r.m.Close() // must not panic on a manager with no live sessions
	case <-time.After(8 * time.Second):
		t.Fatal("Connect hung on a server that fails to start")
	}
}
