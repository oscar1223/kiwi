package lsp

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
)

// EnvServers overrides detection. The format is one server per entry,
// "<ext>[,<ext>…]=<command>", separated by semicolons — for example
// ".go=gopls;.rb=solargraph --stdio".
const EnvServers = "KIWI_LSP"

// Server is one language server Kiwi knows how to start.
type Server struct {
	Name       string
	Command    []string
	Extensions []string
	// Marker is a file whose presence says the project actually uses this
	// language. Without it, having gopls installed would start it inside a
	// Node project the first time anyone asked about a stray .go file.
	Marker []string
}

// known is the built-in catalogue, in the order it is tried.
//
// It is short on purpose. Every entry is a promise that Kiwi knows how to
// launch that server correctly, and a wrong entry fails at the worst moment —
// mid-turn, on somebody's machine, with a timeout rather than an error.
var known = []Server{
	{
		Name:       "gopls",
		Command:    []string{"gopls"},
		Extensions: []string{".go"},
		Marker:     []string{"go.mod"},
	},
	{
		Name:       "typescript-language-server",
		Command:    []string{"typescript-language-server", "--stdio"},
		Extensions: []string{".ts", ".tsx", ".js", ".jsx", ".mts", ".cts"},
		Marker:     []string{"tsconfig.json", "package.json"},
	},
	{
		Name:       "pyright",
		Command:    []string{"pyright-langserver", "--stdio"},
		Extensions: []string{".py"},
		Marker:     []string{"pyproject.toml", "setup.py", "requirements.txt"},
	},
	{
		Name:       "rust-analyzer",
		Command:    []string{"rust-analyzer"},
		Extensions: []string{".rs"},
		Marker:     []string{"Cargo.toml"},
	},
}

// Manager owns the servers for one working directory, starting each on first
// use and reusing it afterwards.
//
// Lazily, because a language server is expensive: gopls indexes the module
// before it can answer anything, and paying that on every kiwi launch would
// tax every session for a feature most of them never use.
type Manager struct {
	workDir string
	servers []Server

	mu      sync.Mutex
	clients map[string]*Client
	// failed remembers servers that would not start, so a broken install
	// costs one timeout per session rather than one per question.
	failed map[string]error
	closed bool
}

// NewManager returns a manager for workDir, or nil when no language server is
// both installed and relevant to this project.
//
// Nil is the feature switch: the caller does not register the tool, and the
// model is never offered a capability that can only fail.
func NewManager(workDir string) *Manager {
	servers := detect(workDir)
	if len(servers) == 0 {
		return nil
	}
	return &Manager{
		workDir: workDir,
		servers: servers,
		clients: map[string]*Client{},
		failed:  map[string]error{},
	}
}

// detect returns the servers worth offering for this project.
func detect(workDir string) []Server {
	if custom := parseEnv(os.Getenv(EnvServers)); len(custom) > 0 {
		var out []Server
		for _, s := range custom {
			if _, err := exec.LookPath(s.Command[0]); err == nil {
				out = append(out, s)
			}
		}
		return out
	}

	var out []Server
	for _, s := range known {
		if _, err := exec.LookPath(s.Command[0]); err != nil {
			continue
		}
		for _, marker := range s.Marker {
			if _, err := os.Stat(filepath.Join(workDir, marker)); err == nil {
				out = append(out, s)
				break
			}
		}
	}
	return out
}

// parseEnv reads the EnvServers override.
func parseEnv(spec string) []Server {
	var out []Server
	for _, entry := range strings.Split(spec, ";") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		exts, command, ok := strings.Cut(entry, "=")
		if !ok {
			continue
		}
		argv := strings.Fields(command)
		if len(argv) == 0 {
			continue
		}
		var list []string
		for _, e := range strings.Split(exts, ",") {
			if e = strings.TrimSpace(e); e != "" {
				if !strings.HasPrefix(e, ".") {
					e = "." + e
				}
				list = append(list, strings.ToLower(e))
			}
		}
		if len(list) == 0 {
			continue
		}
		out = append(out, Server{Name: argv[0], Command: argv, Extensions: list})
	}
	return out
}

// Languages lists the extensions this manager can answer about, for the tool
// description the model reads.
func (m *Manager) Languages() []string {
	if m == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, s := range m.servers {
		for _, e := range s.Extensions {
			if !seen[e] {
				seen[e] = true
				out = append(out, e)
			}
		}
	}
	return out
}

// For returns the client that handles a file, starting it if needed.
func (m *Manager) For(ctx context.Context, path string) (*Client, error) {
	if m == nil {
		return nil, fmt.Errorf("no language server is configured for this project")
	}
	ext := strings.ToLower(filepath.Ext(path))

	var server Server
	for _, s := range m.servers {
		for _, e := range s.Extensions {
			if e == ext {
				server = s
			}
		}
	}
	if server.Name == "" {
		return nil, fmt.Errorf("no language server handles %s files here", ext)
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil, fmt.Errorf("lsp: shutting down")
	}
	if c, ok := m.clients[server.Name]; ok {
		m.mu.Unlock()
		return c, nil
	}
	if err, ok := m.failed[server.Name]; ok {
		m.mu.Unlock()
		return nil, err
	}
	m.mu.Unlock()

	// Started outside the lock: the handshake can take seconds, and holding
	// the lock through it would block every other question in the meantime.
	c, err := Start(ctx, server.Name, server.Command, m.workDir)

	m.mu.Lock()
	defer m.mu.Unlock()
	if err != nil {
		m.failed[server.Name] = err
		return nil, err
	}
	if m.closed {
		go c.Close()
		return nil, fmt.Errorf("lsp: shutting down")
	}
	// Another caller may have won the race while the lock was released; the
	// loser's server is closed rather than leaked.
	if existing, ok := m.clients[server.Name]; ok {
		go c.Close()
		return existing, nil
	}
	m.clients[server.Name] = c
	return c, nil
}

// Close stops every running server.
func (m *Manager) Close() {
	if m == nil {
		return
	}
	m.mu.Lock()
	clients := m.clients
	m.clients = map[string]*Client{}
	m.closed = true
	m.mu.Unlock()

	var wg sync.WaitGroup
	for _, c := range clients {
		wg.Add(1)
		go func(c *Client) { defer wg.Done(); c.Close() }(c)
	}
	wg.Wait()
}
