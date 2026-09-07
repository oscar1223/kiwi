package tui

import (
	"context"
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/skills"
)

// A skill marked user-invocable becomes a slash command, which is how every
// comparable tool bridges the two: Hermes registers everything in its skills
// directory, Crush surfaces them in its command palette under user:/project:,
// and OpenCode turns any file in commands/ into a command of the same name.
//
// Kiwi already had both halves — /skill manages them, load_skill loads them —
// and only the step between was missing.

// skillCommands returns the user-invocable skills as command entries.
//
// The result is cached: this is called on every keystroke while a "/" is being
// typed, and reading the skills directory that often would put a disk hit
// behind each character. The cache is dropped whenever the agent is rebuilt,
// which is exactly when /skill has changed something.
func (m *Model) skillCommands() []commandSpec {
	if m.opts.Agent == nil {
		return nil
	}
	if m.skillCmds != nil {
		return *m.skillCmds
	}
	loaded, err := skills.Load()
	if err != nil {
		return nil
	}
	var out []commandSpec
	for _, s := range skills.Invocable(loaded) {
		if isNativeCommand("/" + s.Name) {
			// The native command wins. A third-party skill called "clear"
			// hijacking /clear is precisely the failure this guards against,
			// and the skill is still reachable through /skill and load_skill.
			continue
		}
		out = append(out, commandSpec{Name: "/" + s.Name, Desc: skillDesc(s)})
	}
	m.skillCmds = &out
	return out
}

// forgetSkillCommands drops the cache, so the next "/" reflects whatever
// /skill just changed.
func (m *Model) forgetSkillCommands() { m.skillCmds = nil }

// skillDesc trims a skill's description to something that fits a command list.
// The full text is what the model reads; this is what a person skims.
func skillDesc(s skills.Skill) string {
	desc := strings.TrimSpace(s.Description)
	if i := strings.IndexAny(desc, ".\n"); i > 0 {
		desc = desc[:i]
	}
	const max = 64
	if len(desc) > max {
		desc = strings.TrimSpace(desc[:max]) + "…"
	}
	if desc == "" {
		desc = "run the " + s.Name + " skill"
	}
	return desc
}

// skillCommand runs a skill as a turn, with anything typed after the command
// name passed along as the request.
func (m *Model) skillCommand(name, args string) (tea.Cmd, bool) {
	loaded, err := skills.Load()
	if err != nil {
		return nil, false
	}
	sk, ok := loaded[name]
	if !ok || !sk.UserInvocable {
		return nil, false
	}
	if m.opts.Agent == nil {
		return m.println(styleErr.Render("  no model is configured yet — try /model first")), true
	}

	// The body is sent inline rather than through load_skill: the user asked
	// for this skill by name, so spending a tool call for the model to decide
	// whether it wants it would be a round trip that answers a question
	// nobody asked.
	var b strings.Builder
	fmt.Fprintf(&b, "Follow these instructions.\n\n%s\n", sk.Body)
	if args = strings.TrimSpace(args); args != "" {
		fmt.Fprintf(&b, "\nThe user added: %s\n", args)
	}

	display := "/" + name
	if args != "" {
		display += " " + args
	}
	return tea.Batch(
		m.println(styleDim.Render("  running the "+name+" skill")),
		m.startTurn(display, b.String(), nil, nil),
	), true
}

// --- /review ---

// reviewFlow reviews the working changes with a subagent.
//
// A separate agent, not this one. The context that wrote the code is the worst
// context to review it from: it already believes the change is correct, and it
// will read the diff looking for confirmation. Codex, Claude Code and Hermes
// all land on the same answer — the reviewer runs on its own.
func (m *Model) reviewFlow(ctx context.Context, st checkpointState, args string) {
	if m.opts.Agent == nil {
		m.events.send(ctx, systemMsg{"No model is configured yet — try /model first."})
		return
	}

	scope := "everything uncommitted in the working tree"
	if st.store != nil && len(st.marks) > 0 {
		if diff, err := st.store.Diff(ctx, st.marks[0].ID); err == nil && strings.TrimSpace(diff) == "" {
			m.events.send(ctx, systemMsg{"Nothing has changed this session. Reviewing the working tree instead."})
		}
	}

	var b strings.Builder
	b.WriteString("Review the current changes and report what you find.\n\n")
	if sk, err := skills.Load(); err == nil {
		if review, ok := sk["code-review"]; ok {
			b.WriteString(review.Body)
			b.WriteString("\n\n")
		}
	}
	fmt.Fprintf(&b, "Scope: %s. Use the task tool to review it, and report the findings yourself.\n", scope)
	if args = strings.TrimSpace(args); args != "" {
		fmt.Fprintf(&b, "\nThe user asked you to focus on: %s\n", args)
	}

	m.events.send(ctx, reviewRequestMsg{display: strings.TrimSpace("/review " + args), sent: b.String()})
}

// reviewRequestMsg carries a flow's assembled prompt back to Update, which is
// the only goroutine allowed to start a turn.
type reviewRequestMsg struct {
	display string
	sent    string
}

// isNativeCommand reports whether a name is one of Kiwi's own commands.
func isNativeCommand(name string) bool {
	for _, c := range commandRegistry {
		if c.Name == name {
			return true
		}
	}
	return false
}
