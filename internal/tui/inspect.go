package tui

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"sort"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/llm"
)

// --- /context ---

// contextPart is one labelled slice of the context window.
type contextPart struct {
	Label  string
	Tokens int
}

// contextBreakdown splits the next request into the parts it is made of.
//
// The status line's percentage says *when* to compact. This says *what* to
// compact, which is the more useful question: a window that is 70% skills and
// tool definitions does not get better by summarising the conversation.
func (m *Model) contextBreakdown() []contextPart {
	opts := m.opts.PromptOptions
	parts := []contextPart{}

	// The known pieces of the system prompt are measured directly; whatever is
	// left over is Kiwi's own base instructions. Deriving the base by
	// subtraction rather than measuring it keeps the parts summing to the
	// total even when prompt.Build changes shape.
	var accounted int
	add := func(label, body string) {
		if n := llm.EstimateTokens(body); n > 0 {
			parts = append(parts, contextPart{label, n})
			accounted += n
		}
	}
	add("project instructions", opts.ProjectInstructions)
	add("mode instructions", opts.ModeInstructions)
	add("skills", strings.Join(opts.Extra, "\n"))

	if m.opts.Agent != nil {
		if base := llm.EstimateTokens(m.opts.Agent.System) - accounted; base > 0 {
			parts = append(parts, contextPart{"system prompt", base})
		}
		if n := m.toolSchemaTokens(); n > 0 {
			parts = append(parts, contextPart{"tool definitions", n})
		}
	}
	if n := llm.EstimateMessageTokens(m.history); n > 0 {
		parts = append(parts, contextPart{"conversation", n})
	}

	sort.SliceStable(parts, func(i, j int) bool { return parts[i].Tokens > parts[j].Tokens })
	return parts
}

// toolSchemaTokens estimates what the tool definitions cost.
//
// They are easy to forget and rarely small: every MCP server a project
// configures adds its whole catalogue to every single request.
func (m *Model) toolSchemaTokens() int {
	if m.opts.Agent == nil || m.opts.Agent.Tools == nil {
		return 0
	}
	var total int
	for _, s := range m.opts.Agent.Tools.Schemas() {
		total += llm.EstimateTokens(s.Name) + llm.EstimateTokens(s.Description)
		if raw, err := json.Marshal(s.Schema); err == nil {
			total += llm.EstimateTokens(string(raw))
		}
	}
	return total
}

// contextFlow prints the breakdown.
func (m *Model) contextFlow() tea.Cmd {
	parts := m.contextBreakdown()
	window := llm.ContextWindow(m.opts.ModelLabel)

	var total int
	for _, p := range parts {
		total += p.Tokens
	}
	if total == 0 {
		return m.println(styleDim.Render("  nothing in the context window yet"))
	}

	lines := []string{styleDim.Render(fmt.Sprintf("  %s of %s used (%d%% of the window)",
		formatTokenCount(total), formatTokenCount(window), percentOf(total, window)))}
	for _, p := range parts {
		lines = append(lines, fmt.Sprintf("  %s %s %s",
			styleTool.Render(fmt.Sprintf("%-20s", p.Label)),
			contextBar(p.Tokens, total),
			styleDim.Render(fmt.Sprintf("%7s  %3d%%", formatTokenCount(p.Tokens), percentOf(p.Tokens, total)))))
	}
	return m.println(strings.Join(lines, "\n"))
}

// contextBar draws one part's share of the total.
func contextBar(part, total int) string {
	const width = 24
	filled := 0
	if total > 0 {
		filled = part * width / total
	}
	// A part that rounds to nothing still gets a cell: "too small to draw" and
	// "not there" are different answers, and only one of them is true.
	if filled == 0 && part > 0 {
		filled = 1
	}
	return styleKiwi.Render(strings.Repeat("█", filled)) +
		styleDim.Render(strings.Repeat("·", width-filled))
}

// --- /tools ---

// toolsFlow lists the tools and lets one be switched off for the session.
func (m *Model) toolsFlow(ctx context.Context) {
	if m.opts.Agent == nil || m.opts.Agent.Tools == nil {
		m.events.send(ctx, systemMsg{"No agent is configured yet, so there are no tools to show."})
		return
	}
	reg := m.opts.Agent.Tools

	for {
		names := reg.Names()
		if len(names) == 0 {
			m.events.send(ctx, systemMsg{"No tools are registered."})
			return
		}
		options := make([]pickOption, 0, len(names))
		for _, n := range names {
			label := n
			switch {
			case reg.IsMuted(n):
				label = "✗ " + n + "  (off)"
			case strings.HasPrefix(n, "mcp"):
				label = "✓ " + n + "  (mcp)"
			default:
				label = "✓ " + n
			}
			options = append(options, pickOption{label, n})
		}
		options = append(options, pickOption{"Done", ""})

		choice, ok := m.events.Pick(ctx, "Tools — pick one to turn on or off", options)
		if !ok || choice == "" {
			return
		}
		muted := reg.IsMuted(choice)
		reg.Mute(choice, !muted)
		if muted {
			m.events.send(ctx, systemMsg{choice + " is available again."})
		} else {
			m.events.send(ctx, systemMsg{choice + " is off for the rest of this session."})
		}
	}
}

// --- /status ---

// statusFlow prints everything about the session in one place: the answer to
// "what am I actually running right now", which otherwise takes four commands
// to assemble.
func (m *Model) statusFlow(ctx context.Context, st checkpointState) {
	rows := [][2]string{
		{"model", m.opts.ModelLabel},
		{"mode", string(m.opts.Broker.Mode())},
		{"directory", m.opts.WorkDir},
	}

	if branch := gitBranch(ctx, m.opts.WorkDir); branch != "" {
		rows = append(rows, [2]string{"git", branch})
	} else {
		rows = append(rows, [2]string{"git", "not a repository"})
	}

	if file := m.opts.PromptOptions.ProjectFile; file != "" {
		rows = append(rows, [2]string{"instructions", file})
	} else {
		rows = append(rows, [2]string{"instructions", "none — /init writes " + config.ProjectFiles[0]})
	}

	if m.opts.SessionID != "" {
		rows = append(rows, [2]string{"session", m.opts.SessionID})
	} else {
		rows = append(rows, [2]string{"session", "not persisted"})
	}

	if m.opts.Agent != nil && m.opts.Agent.Tools != nil {
		reg := m.opts.Agent.Tools
		active := len(reg.Schemas())
		line := fmt.Sprintf("%d available", active)
		if off := len(reg.Names()) - active; off > 0 {
			line += fmt.Sprintf(", %d turned off", off)
		}
		rows = append(rows, [2]string{"tools", line})
		rows = append(rows, [2]string{"max steps", fmt.Sprintf("%d per turn", m.opts.Agent.MaxSteps)})
	}

	used, window := m.contextUsage()
	rows = append(rows, [2]string{"context", fmt.Sprintf("%s of %s (%d%%)",
		formatTokenCount(used), formatTokenCount(window), percentOf(used, window))})
	rows = append(rows, [2]string{"tokens", fmt.Sprintf("%s in, %s out this session",
		formatTokenCount(m.sessionUsage.InputTokens), formatTokenCount(m.sessionUsage.OutputTokens))})

	switch {
	case st.store == nil:
		rows = append(rows, [2]string{"checkpoints", "unavailable here"})
	case len(st.marks) == 0:
		rows = append(rows, [2]string{"checkpoints", "ready, none taken yet"})
	default:
		line := fmt.Sprintf("%d this session", len(st.marks))
		if len(st.undone) > 0 {
			line += fmt.Sprintf(", %d undone", len(st.undone))
		}
		rows = append(rows, [2]string{"checkpoints", line})
	}

	lines := make([]string, 0, len(rows))
	for _, r := range rows {
		lines = append(lines, fmt.Sprintf("  %s  %s",
			styleTool.Render(fmt.Sprintf("%-14s", r[0])), styleDim.Render(r[1])))
	}
	m.events.send(ctx, printLinesMsg{lines: lines})
}

// gitBranch reports the current branch and whether the tree is dirty, or ""
// when the directory is not a repository.
func gitBranch(ctx context.Context, dir string) string {
	cmd := exec.CommandContext(ctx, "git", "rev-parse", "--abbrev-ref", "HEAD")
	cmd.Dir = dir
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	branch := strings.TrimSpace(string(out))

	status := exec.CommandContext(ctx, "git", "status", "--porcelain")
	status.Dir = dir
	if dirty, err := status.Output(); err == nil && strings.TrimSpace(string(dirty)) != "" {
		n := len(strings.Split(strings.TrimSpace(string(dirty)), "\n"))
		return fmt.Sprintf("%s (%d uncommitted)", branch, n)
	}
	return branch
}

// --- /init ---

// initPrompt is what /init actually sends. It is a turn, not a template: the
// only way to write instructions worth loading into every future prompt is to
// go and read the project first.
const initPrompt = `Write the project instructions file for this repository.

Look around first — use glob, grep and ls to find out what this project is
before writing a word about it. Read the build files, the entry points, the
test layout, and any existing README.

Then write ` + "`KIWI.md`" + ` at the top of the working directory, covering:

- What this project is, in one or two sentences.
- The stack and layout: which directory holds what, and why, where it is not obvious.
- The commands that matter: build, test, run, lint. Exactly as they are invoked here.
- Conventions a newcomer would otherwise get wrong — naming, error handling,
  testing style, anything the code does consistently that is not the default.
- Anything that is genuinely surprising about this codebase.

Keep it short enough that it is worth loading into every prompt: aim for well
under a hundred lines. Write only what you have actually verified by reading
the code. Do not invent commands, do not pad it with generic advice, and do not
restate what the file layout already says.

If KIWI.md or AGENTS.md already exists, read it and improve it in place rather
than starting over.`

// initFlow starts the /init turn.
func (m *Model) initFlow() tea.Cmd {
	if m.opts.Agent == nil {
		return m.println(styleErr.Render("  no model is configured yet — try /model first"))
	}
	existing := m.opts.PromptOptions.ProjectFile
	note := "  writing " + config.ProjectFiles[0] + " — this reads the project first, so give it a moment"
	if existing != "" {
		note = "  updating " + existing + " in place"
	}
	return tea.Batch(
		m.println(styleDim.Render(note)),
		m.startTurn("/init", initPrompt, nil, nil),
	)
}
