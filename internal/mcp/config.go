// Package mcp connects Kiwi to Model Context Protocol servers and exposes
// their tools through the same tools.Tool interface every built-in tool
// implements — an MCP tool call goes through the exact same permission gate
// as bash or edit_file, the model cannot tell the difference.
package mcp

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/oscar1223/kiwi/internal/config"
)

// TransportKind names the wire protocol a remote ServerConfig speaks. It is
// only meaningful when URL is set — a Command-based server is always stdio.
type TransportKind string

const (
	// TransportHTTP is the current Streamable HTTP transport (the spec since
	// 2025-03-26) and the default for any ServerConfig with a URL: it is
	// what the overwhelming majority of remote MCP servers speak today.
	TransportHTTP TransportKind = "http"
	// TransportSSE is the older transport (spec 2024-11-05) some servers
	// still speak. Set Type: "sse" explicitly to use it.
	TransportSSE TransportKind = "sse"
)

// ServerConfig is one configured MCP server, over stdio (Command set) or a
// remote transport (URL set) — exactly one of the two.
type ServerConfig struct {
	// Command, Args, Env: a local server run as a subprocess over stdio.
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`

	// URL: a remote server. Type selects which transport it speaks; empty
	// defaults to TransportHTTP. Headers are sent with every request — the
	// usual place for an Authorization bearer token.
	URL     string            `json:"url,omitempty"`
	Type    TransportKind     `json:"type,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`
}

// IsRemote reports whether this config connects over HTTP/SSE rather than
// spawning a local subprocess.
func (sc ServerConfig) IsRemote() bool { return sc.URL != "" }

// Validate reports whether the config is well-formed: exactly one of Command
// or URL, and no unrecognized Type.
func (sc ServerConfig) Validate() error {
	switch {
	case sc.Command == "" && sc.URL == "":
		return errors.New("mcp: server config needs either \"command\" (stdio) or \"url\" (remote)")
	case sc.Command != "" && sc.URL != "":
		return errors.New("mcp: server config cannot set both \"command\" and \"url\"")
	}
	if sc.URL != "" {
		switch sc.Type {
		case "", TransportHTTP, TransportSSE:
		default:
			return fmt.Errorf("mcp: unknown transport type %q (want %q or %q)", sc.Type, TransportHTTP, TransportSSE)
		}
	}
	return nil
}

// Config maps server name to its configuration.
type Config map[string]ServerConfig

// ErrNotFound is returned when a named server does not exist.
var ErrNotFound = errors.New("mcp: server not found")

// ConfigPath is where the server list lives: outside any project repo, like
// every other piece of Kiwi's own configuration.
func ConfigPath() (string, error) {
	dir, err := config.Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "mcp.json"), nil
}

// LoadConfig reads the server list, or an empty one if it does not exist yet.
func LoadConfig() (Config, error) {
	path, err := ConfigPath()
	if err != nil {
		return nil, err
	}
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return Config{}, nil
	}
	if err != nil {
		return nil, err
	}
	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("mcp: %s: %w", path, err)
	}
	if cfg == nil {
		cfg = Config{}
	}
	return cfg, nil
}

func SaveConfig(cfg Config) error {
	path, err := ConfigPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

// AddServer validates, saves, and persists a new server config.
func AddServer(name string, sc ServerConfig) error {
	if err := sc.Validate(); err != nil {
		return err
	}
	cfg, err := LoadConfig()
	if err != nil {
		return err
	}
	cfg[name] = sc
	return SaveConfig(cfg)
}

// RemoveServer deletes a server and persists the config.
func RemoveServer(name string) error {
	cfg, err := LoadConfig()
	if err != nil {
		return err
	}
	if _, ok := cfg[name]; !ok {
		return fmt.Errorf("%w: %q", ErrNotFound, name)
	}
	delete(cfg, name)
	return SaveConfig(cfg)
}

func sortedNames(cfg Config) []string {
	names := make([]string, 0, len(cfg))
	for n := range cfg {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}
