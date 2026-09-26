// Package reach reports which of the ways Kiwi's agents can reach the
// internet work on this machine, and how to enable the ones that do not.
//
// The idea is borrowed from Agent Reach (github.com/Panniantong/agent-reach):
// the agent calls ordinary command-line tools for each platform, and a doctor
// says which are installed. Nothing here installs anything.
package reach

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/oscar1223/kiwi/internal/tools"
)

// Channel is one way of reaching the internet and whether it works here.
type Channel struct {
	Name    string
	OK      bool
	Backend string // what it uses, when it works
	Fix     string // how to make it work, when it does not; or an upgrade hint
}

// Probe is what Check looks at, so the tests can describe a machine instead
// of depending on the one they run on.
type Probe struct {
	Env      func(string) string
	LookPath func(string) (string, error)
	// Run runs a command and reports whether it succeeded.
	Run func(ctx context.Context, name string, args ...string) bool
	// MCPServers lists the configured MCP servers.
	MCPServers func() ([]string, error)
}

// SystemProbe looks at this machine.
func SystemProbe(mcpServers func() ([]string, error)) Probe {
	return Probe{
		Env:      os.Getenv,
		LookPath: exec.LookPath,
		Run: func(ctx context.Context, name string, args ...string) bool {
			ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
			defer cancel()
			return exec.CommandContext(ctx, name, args...).Run() == nil
		},
		MCPServers: mcpServers,
	}
}

// Check reports on every channel.
func Check(ctx context.Context, p Probe) []Channel {
	has := func(bin string) bool { _, err := p.LookPath(bin); return err == nil }

	var out []Channel

	out = append(out, Channel{Name: "Web pages", OK: true, Backend: "web_fetch; reader: true for JavaScript pages (Jina Reader)"})

	search := Channel{Name: "Web search", OK: true}
	switch tools.SearchProvider(p.Env) {
	case "exa":
		search.Backend = "web_search via Exa"
	case "jina":
		search.Backend = "web_search via Jina Search"
	default:
		search.Backend = "web_search via DuckDuckGo (no key)"
		search.Fix = "set EXA_API_KEY or JINA_API_KEY with /config for better results"
	}
	out = append(out, search)

	yt := Channel{Name: "YouTube", OK: has("yt-dlp")}
	if yt.OK {
		yt.Backend = "yt-dlp"
	} else {
		yt.Fix = "brew install yt-dlp"
	}
	out = append(out, yt)

	gh := Channel{Name: "GitHub"}
	switch {
	case !has("gh"):
		gh.Fix = "brew install gh && gh auth login"
	case !p.Run(ctx, "gh", "auth", "status"):
		gh.OK, gh.Backend = true, "gh (public repos only)"
		gh.Fix = "gh auth login to reach private repos and raise rate limits"
	default:
		gh.OK, gh.Backend = true, "gh (signed in)"
	}
	out = append(out, gh)

	out = append(out,
		Channel{Name: "RSS / Atom", OK: true, Backend: "web_fetch"},
		Channel{Name: "Reddit, X", OK: true, Backend: "web_fetch (.json or reader: true); best effort"},
	)

	if p.MCPServers != nil {
		mcp := Channel{Name: "MCP servers"}
		names, err := p.MCPServers()
		switch {
		case err != nil:
			mcp.Fix = "could not read mcp.json: " + err.Error()
		case len(names) == 0:
			mcp.Fix = "none configured; add one with /mcp"
		default:
			mcp.OK, mcp.Backend = true, strings.Join(names, ", ")
		}
		out = append(out, mcp)
	}
	return out
}

// Render lays the report out as plain text, one channel per line.
func Render(channels []Channel) string {
	width := 0
	for _, c := range channels {
		width = max(width, len(c.Name))
	}
	var b strings.Builder
	for _, c := range channels {
		mark := "✓"
		if !c.OK {
			mark = "✗"
		}
		line := fmt.Sprintf("%s %-*s  %s", mark, width, c.Name, c.Backend)
		if c.Fix != "" {
			if c.Backend != "" {
				line += " — "
			}
			line += c.Fix
		}
		b.WriteString(strings.TrimRight(line, " ") + "\n")
	}
	return b.String()
}
