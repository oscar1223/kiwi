package tui

import (
	"context"
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/llm"
)

// Showing and choosing how hard the model thinks.
//
// While the model reasons, the spinner row says so and shows the last words
// of its thinking, dimmed. None of it goes into the transcript: it is there
// to show that something is happening, not to be read back.

// reasoningDeltaMsg carries a chunk of the model's reasoning.
type reasoningDeltaMsg struct {
	gen   int
	delta string
}

func (o *observer) OnReasoning(delta string) {
	o.events.send(o.ctx, reasoningDeltaMsg{gen: o.gen, delta: delta})
}

// thinkingTailRunes is how much of the reasoning is kept for display.
const thinkingTailRunes = 240

// think records a chunk of reasoning.
func (m *Model) think(delta string) {
	m.thinking = true
	m.thoughtWords += len(strings.Fields(delta))
	tail := []rune(m.thought + delta)
	if len(tail) > thinkingTailRunes {
		tail = tail[len(tail)-thinkingTailRunes:]
	}
	m.thought = string(tail)
}

// stopThinking clears the thinking display, once the answer or a tool call
// shows the model has moved on.
func (m *Model) stopThinking() {
	m.thinking, m.thought, m.thoughtWords = false, "", 0
}

// thinkingRow is the spinner row while the model reasons: how long, roughly
// how much, and the latest of what it is thinking, cut to the window.
func (m *Model) thinkingRow(width int) string {
	head := sprintf("%s %s", m.spinner.View(),
		styleDim.Render(sprintf("thinking… %s · %d words · esc to cancel", elapsed(m.began), m.thoughtWords)))
	room := width - lipgloss.Width(head) - 3
	last := strings.Join(strings.Fields(m.thought), " ")
	if room < 12 || last == "" {
		return fit(head, width)
	}
	if r := []rune(last); len(r) > room {
		last = "…" + string(r[len(r)-room+1:])
	}
	return fit(head+styleDim.Italic(true).Render("  "+last), width)
}

// reasoningLabel is the status-line tag for the current level, or "" when
// the provider's default is in force.
func (m *Model) reasoningLabel() string {
	if m.opts.Agent == nil {
		return ""
	}
	rr, ok := m.opts.Agent.Provider.(llm.ReasoningReporter)
	if !ok || rr.Reasoning() == llm.ReasoningDefault {
		return ""
	}
	return "think " + rr.Reasoning().Label()
}

// reasoningCommand handles /reasoning: with a level it sets it at once,
// without one it opens a picker.
func (m *Model) reasoningCommand(text string) tea.Cmd {
	args := strings.Fields(text)[1:]
	if len(args) == 0 {
		return m.runFlow(m.reasoningFlow)
	}
	level, err := llm.ParseReasoning(args[0])
	if err != nil {
		return m.println(styleErr.Render("  " + err.Error()))
	}
	return m.runFlow(func(ctx context.Context) { m.setReasoning(ctx, level) })
}

func (m *Model) reasoningFlow(ctx context.Context) {
	cfg, err := config.Load()
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	current, _ := llm.ParseReasoning(cfg.Profiles[cfg.Current].Reasoning)

	describe := map[llm.Reasoning]string{
		llm.ReasoningDefault: "whatever the provider does by default",
		llm.ReasoningOff:     "answer straight away — fastest and cheapest",
		llm.ReasoningLow:     "a little thinking",
		llm.ReasoningMedium:  "a balance of speed and depth",
		llm.ReasoningHigh:    "think it through",
		llm.ReasoningXHigh:   "think hard — good for coding and long tasks",
		llm.ReasoningMax:     "think as much as it can — slowest and dearest",
	}
	var options []pickOption
	for _, l := range append([]llm.Reasoning{llm.ReasoningDefault}, llm.ReasoningLevels...) {
		marker := "  "
		if l == current {
			marker = "→ "
		}
		options = append(options, pickOption{fmt.Sprintf("%s%-8s %s", marker, l.Label(), describe[l]), l.Label()})
	}
	choice, ok := m.events.Pick(ctx, "Reasoning for "+cfg.Current+" — current: "+current.Label(), options)
	if !ok {
		return
	}
	level, _ := llm.ParseReasoning(choice)
	m.setReasoning(ctx, level)
}

// setReasoning saves the level on the current profile and rebuilds the agent
// with it, which is also what makes it apply to the next request.
func (m *Model) setReasoning(ctx context.Context, level llm.Reasoning) {
	cfg, err := config.Load()
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	if err := cfg.SetReasoning(cfg.Current, string(level)); err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	m.events.send(ctx, systemMsg{fmt.Sprintf("Reasoning for %s set to %s.", cfg.Current, level.Label())})
	m.events.send(ctx, requestRebuildMsg{})
}
