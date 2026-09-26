package tui

import (
	"strings"
	"testing"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
)

// While the model reasons the spinner row says so, with the latest of its
// thinking; the first words of the answer put it back.
func TestThinkingRowShowsAndClears(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.Update(tea.WindowSizeMsg{Width: 100, Height: 20})
	m.busy = true

	m.Update(reasoningDeltaMsg{gen: m.gen, delta: "first I should check the config file"})
	view := plain(m.View().Content)
	if !strings.Contains(view, "thinking…") || !strings.Contains(view, "check the config file") {
		t.Fatalf("no thinking row:\n%s", view)
	}
	if strings.Contains(plain(strings.Join(m.transcript.render(100), "\n")), "config file") {
		t.Error("reasoning was written into the transcript")
	}

	m.Update(textDeltaMsg{gen: m.gen, delta: "Here"})
	if view := plain(m.View().Content); strings.Contains(view, "thinking…") {
		t.Errorf("thinking row stayed after the answer began:\n%s", view)
	}
}

// Reasoning from a cancelled turn is not shown.
func TestStaleReasoningIsIgnored(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.busy = true
	m.Update(reasoningDeltaMsg{gen: m.gen - 1, delta: "old"})
	if m.thinking {
		t.Error("a stale reasoning delta was shown")
	}
}

type reasoningProvider struct {
	llm.Provider
	level llm.Reasoning
}

func (p reasoningProvider) Reasoning() llm.Reasoning { return p.level }

func TestStatusLineShowsTheLevel(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.opts.Agent.Provider = reasoningProvider{Provider: m.opts.Agent.Provider, level: llm.ReasoningXHigh}
	if line := plain(m.statusLine()); !strings.Contains(line, "think xhigh") {
		t.Errorf("status line = %q", line)
	}
	m.opts.Agent.Provider = reasoningProvider{Provider: m.opts.Agent.Provider, level: llm.ReasoningDefault}
	if line := plain(m.statusLine()); strings.Contains(line, "think") {
		t.Errorf("default level still shown: %q", line)
	}
}

// /reasoning <level> saves the level on the current profile and asks for the
// agent to be rebuilt with it.
func TestReasoningCommandSavesTheLevel(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	cfg := config.Default()
	if err := cfg.Save(); err != nil {
		t.Fatal(err)
	}

	m, _ := newTestModel(t, permission.ModeAsk)
	m.setReasoning(t.Context(), llm.ReasoningLow)

	reloaded, err := config.Load()
	if err != nil {
		t.Fatal(err)
	}
	if got := reloaded.Profiles[reloaded.Current].Reasoning; got != "low" {
		t.Errorf("saved reasoning = %q, want low", got)
	}
}

func TestReasoningCommandRejectsAnUnknownLevel(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.reasoningCommand("/reasoning loads")
	if !strings.Contains(plain(strings.Join(m.transcript.render(100), "\n")), "unknown reasoning level") {
		t.Error("an unknown level was not reported")
	}
}
