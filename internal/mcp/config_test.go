package mcp

import (
	"errors"
	"testing"
)

func withMCPConfigDir(t *testing.T) {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
}

func TestLoadConfigMissingIsEmpty(t *testing.T) {
	withMCPConfigDir(t)
	cfg, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if len(cfg) != 0 {
		t.Errorf("got %d servers, want 0", len(cfg))
	}
}

func TestAddServerPersists(t *testing.T) {
	withMCPConfigDir(t)
	err := AddServer("filesystem", ServerConfig{
		Command: "npx",
		Args:    []string{"-y", "@modelcontextprotocol/server-filesystem", "/tmp"},
		Env:     map[string]string{"FOO": "bar"},
	})
	if err != nil {
		t.Fatalf("AddServer: %v", err)
	}

	reloaded, err := LoadConfig()
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	sc, ok := reloaded["filesystem"]
	if !ok {
		t.Fatal("server was not persisted")
	}
	if sc.Command != "npx" || len(sc.Args) != 3 || sc.Env["FOO"] != "bar" {
		t.Errorf("got %+v", sc)
	}
}

func TestRemoveServer(t *testing.T) {
	withMCPConfigDir(t)
	AddServer("a", ServerConfig{Command: "x"})

	if err := RemoveServer("a"); err != nil {
		t.Fatalf("RemoveServer: %v", err)
	}
	reloaded, _ := LoadConfig()
	if _, ok := reloaded["a"]; ok {
		t.Error("removal was not persisted")
	}
}

func TestRemoveUnknownServer(t *testing.T) {
	withMCPConfigDir(t)
	if err := RemoveServer("nope"); !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

func TestValidateRequiresCommandOrURL(t *testing.T) {
	if err := (ServerConfig{}).Validate(); err == nil {
		t.Error("expected an error when neither command nor url is set")
	}
}

func TestValidateRejectsBothCommandAndURL(t *testing.T) {
	sc := ServerConfig{Command: "npx", URL: "http://example.com"}
	if err := sc.Validate(); err == nil {
		t.Error("expected an error when both command and url are set")
	}
}

func TestValidateAcceptsStdio(t *testing.T) {
	if err := (ServerConfig{Command: "npx"}).Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

func TestValidateAcceptsHTTPAndSSE(t *testing.T) {
	for _, typ := range []TransportKind{"", TransportHTTP, TransportSSE} {
		sc := ServerConfig{URL: "http://example.com", Type: typ}
		if err := sc.Validate(); err != nil {
			t.Errorf("Validate(%q): %v", typ, err)
		}
	}
}

func TestValidateRejectsUnknownTransportType(t *testing.T) {
	sc := ServerConfig{URL: "http://example.com", Type: "carrier-pigeon"}
	if err := sc.Validate(); err == nil {
		t.Error("expected an error for an unknown transport type")
	}
}

func TestIsRemote(t *testing.T) {
	if (ServerConfig{Command: "npx"}).IsRemote() {
		t.Error("a command-based config should not be remote")
	}
	if !(ServerConfig{URL: "http://example.com"}).IsRemote() {
		t.Error("a url-based config should be remote")
	}
}

// AddServer must reject a malformed config before it ever touches disk.
func TestAddServerValidatesBeforePersisting(t *testing.T) {
	withMCPConfigDir(t)
	if err := AddServer("broken", ServerConfig{}); err == nil {
		t.Fatal("expected AddServer to reject an invalid config")
	}
	cfg, _ := LoadConfig()
	if _, ok := cfg["broken"]; ok {
		t.Error("an invalid server was persisted anyway")
	}
}
