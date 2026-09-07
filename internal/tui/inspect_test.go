package tui

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/prompt"
	"github.com/oscar1223/kiwi/internal/tools"
)

// countedTool exists so the registry in a test model is not empty: the tool
// definitions are part of what fills the context window, and a registry with
// nothing in it would not exercise that.
type countedTool struct{}

func (countedTool) Name() string           { return "counted" }
func (countedTool) Description() string    { return "a tool with a description worth some tokens" }
func (countedTool) Schema() map[string]any { return map[string]any{"type": "object"} }
func (countedTool) Run(context.Context, json.RawMessage) (string, error) {
	return "", nil
}

// runnableModel is a test model whose turns can actually start: the bare one
// has no provider, so anything that reaches the agent panics off-goroutine.
func runnableModel(t *testing.T) *Model {
	t.Helper()
	m, _ := newTestModel(t, permission.ModeAsk)
	m.opts.Agent.Provider = &llmtest.Fake{}
	m.opts.Agent.Tools = tools.NewRegistry(countedTool{})
	return m
}

func TestContextBreakdownNamesEveryPartAndIsOrderedBySize(t *testing.T) {
	m := runnableModel(t)
	m.opts.PromptOptions = prompt.Options{
		ProjectInstructions: strings.Repeat("project instructions ", 50),
		ModeInstructions:    "be careful",
		Extra:               []string{strings.Repeat("skill ", 10)},
	}
	m.opts.Agent.System = strings.Repeat("system ", 400)
	m.history = msgs(3)

	parts := m.contextBreakdown()
	if len(parts) < 4 {
		t.Fatalf("breakdown has %d parts, want the system prompt, project, mode and skills at least: %+v", len(parts), parts)
	}
	for i := 1; i < len(parts); i++ {
		if parts[i-1].Tokens < parts[i].Tokens {
			t.Errorf("parts are not ordered largest first: %+v", parts)
			break
		}
	}
	seen := map[string]bool{}
	for _, p := range parts {
		seen[p.Label] = true
		if p.Tokens <= 0 {
			t.Errorf("part %q reported %d tokens", p.Label, p.Tokens)
		}
	}
	for _, want := range []string{"project instructions", "mode instructions", "skills", "system prompt"} {
		if !seen[want] {
			t.Errorf("breakdown is missing %q: %+v", want, parts)
		}
	}
}

// The known pieces are subtracted from the measured system prompt, so a
// project whose instructions dominate cannot push the derived base negative
// and produce a part that makes no sense.
func TestContextBreakdownNeverReportsANegativePart(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.opts.Agent.System = "tiny"
	m.opts.PromptOptions = prompt.Options{ProjectInstructions: strings.Repeat("huge ", 500)}

	for _, p := range m.contextBreakdown() {
		if p.Tokens < 0 {
			t.Errorf("part %q reported %d tokens", p.Label, p.Tokens)
		}
	}
}

func TestContextBreakdownOnAFreshSession(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.opts.Agent.System = ""
	m.opts.PromptOptions = prompt.Options{}
	m.history = nil

	if got := m.contextBreakdown(); len(got) != 0 {
		t.Errorf("breakdown on an empty session = %+v, want nothing", got)
	}
	if _, handled := m.command("/context"); !handled {
		t.Error("/context was not handled on an empty session")
	}
}

func TestContextBarAlwaysShowsATinyPart(t *testing.T) {
	// A part too small to fill a cell still has to be visible: "too small to
	// draw" and "not there" are different answers.
	if bar := plain(contextBar(1, 100_000)); !strings.Contains(bar, "█") {
		t.Errorf("a tiny part drew no bar at all: %q", bar)
	}
	if bar := plain(contextBar(0, 100)); strings.Contains(bar, "█") {
		t.Errorf("an empty part drew a bar: %q", bar)
	}
}

func TestToolSchemaTokensCountTheDefinitions(t *testing.T) {
	m := runnableModel(t)
	if got := m.toolSchemaTokens(); got <= 0 {
		t.Errorf("toolSchemaTokens = %d, want the definitions to cost something", got)
	}
}

func TestInitSendsAPromptWithoutShowingIt(t *testing.T) {
	m := runnableModel(t)

	cmd, handled := m.command("/init")
	if !handled {
		t.Fatal("/init was not handled")
	}
	if cmd == nil {
		t.Fatal("/init started nothing")
	}
	if !m.busy {
		t.Error("/init did not start a turn")
	}

	// The transcript shows the two words the user typed, not the page of
	// instructions actually sent.
	shown := plain(strings.Join(m.transcript.render(200), "\n"))
	if !strings.Contains(shown, "/init") {
		t.Errorf("the transcript does not show the command: %q", shown)
	}
	if strings.Contains(shown, "Look around first") {
		t.Error("the whole init prompt was dumped into the transcript")
	}
}

func TestInitMentionsTheFileItWillWrite(t *testing.T) {
	m := runnableModel(t)
	m.command("/init")
	if got := plain(strings.Join(m.transcript.render(200), "\n")); !strings.Contains(got, "KIWI.md") {
		t.Errorf("the /init notice does not say which file it writes: %q", got)
	}
}

func TestInspectCommandsAreRegistered(t *testing.T) {
	for _, name := range []string{"/init", "/status", "/context", "/tools"} {
		if !isKnownCommand(name) {
			t.Errorf("%s is not in the command registry", name)
		}
		if !strings.Contains(plain(helpText()), name) {
			t.Errorf("%s is missing from /help", name)
		}
		if got := filterCommands(name); len(got) == 0 {
			t.Errorf("%s does not autocomplete", name)
		}
	}
}

func TestStatusAndToolsAreHandled(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	for _, name := range []string{"/status", "/tools"} {
		if _, handled := m.command(name); !handled {
			t.Errorf("%s was not handled and would have gone to the model", name)
		}
		// Each opens a flow; the next one cannot start until it finishes.
		m.flowBusy = false
	}
}

func TestFormatTokenCountStaysShort(t *testing.T) {
	cases := map[int]string{0: "0", 999: "999", 1000: "1.0k", 12345: "12.3k"}
	for in, want := range cases {
		if got := formatTokenCount(in); got != want {
			t.Errorf("formatTokenCount(%d) = %q, want %q", in, got, want)
		}
	}

}
