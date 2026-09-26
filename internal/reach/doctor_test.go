package reach

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func probe(env map[string]string, bins []string, ghAuthed bool, servers []string) Probe {
	return Probe{
		Env: func(k string) string { return env[k] },
		LookPath: func(b string) (string, error) {
			for _, have := range bins {
				if have == b {
					return "/usr/bin/" + b, nil
				}
			}
			return "", errors.New("not found")
		},
		Run: func(_ context.Context, name string, args ...string) bool {
			return ghAuthed && name == "gh"
		},
		MCPServers: func() ([]string, error) { return servers, nil },
	}
}

func find(t *testing.T, cs []Channel, name string) Channel {
	t.Helper()
	for _, c := range cs {
		if c.Name == name {
			return c
		}
	}
	t.Fatalf("no channel %q", name)
	return Channel{}
}

// A bare machine still has the web, and says how to get the rest.
func TestCheckOnABareMachine(t *testing.T) {
	cs := Check(context.Background(), probe(nil, nil, false, nil))

	if c := find(t, cs, "Web search"); !c.OK || !strings.Contains(c.Backend, "DuckDuckGo") {
		t.Errorf("search = %+v", c)
	}
	if c := find(t, cs, "YouTube"); c.OK || c.Fix != "brew install yt-dlp" {
		t.Errorf("youtube = %+v", c)
	}
	if c := find(t, cs, "GitHub"); c.OK || !strings.Contains(c.Fix, "brew install gh") {
		t.Errorf("github = %+v", c)
	}
	if c := find(t, cs, "MCP servers"); c.OK {
		t.Errorf("mcp = %+v", c)
	}
}

func TestCheckOnAnEquippedMachine(t *testing.T) {
	cs := Check(context.Background(), probe(
		map[string]string{"EXA_API_KEY": "k"}, []string{"yt-dlp", "gh"}, true, []string{"exa", "linear"}))

	if c := find(t, cs, "Web search"); !strings.Contains(c.Backend, "Exa") || c.Fix != "" {
		t.Errorf("search = %+v", c)
	}
	if c := find(t, cs, "YouTube"); !c.OK {
		t.Errorf("youtube = %+v", c)
	}
	if c := find(t, cs, "GitHub"); !c.OK || c.Backend != "gh (signed in)" {
		t.Errorf("github = %+v", c)
	}
	if c := find(t, cs, "MCP servers"); !c.OK || c.Backend != "exa, linear" {
		t.Errorf("mcp = %+v", c)
	}
}

// gh installed but signed out still works, for public repositories.
func TestCheckGitHubSignedOut(t *testing.T) {
	cs := Check(context.Background(), probe(nil, []string{"gh"}, false, nil))
	if c := find(t, cs, "GitHub"); !c.OK || !strings.Contains(c.Fix, "gh auth login") {
		t.Errorf("github = %+v", c)
	}
}

func TestRender(t *testing.T) {
	out := Render([]Channel{
		{Name: "Web", OK: true, Backend: "web_fetch"},
		{Name: "YouTube", Fix: "brew install yt-dlp"},
	})
	want := "✓ Web      web_fetch\n✗ YouTube  brew install yt-dlp\n"
	if out != want {
		t.Errorf("Render =\n%q\nwant\n%q", out, want)
	}
}
