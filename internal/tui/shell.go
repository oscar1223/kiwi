package tui

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// shellTimeout bounds a "!" command. Anything longer belongs in another
// terminal: this is for the quick check, not for running a dev server.
const shellTimeout = 2 * time.Minute

// maxShellOutput caps what one command puts in the transcript.
const maxShellOutput = 200

// shellResultMsg carries a finished "!" command back to Update.
type shellResultMsg struct {
	command string
	output  string
	err     error
}

// shellCommand runs a "!"-prefixed line directly, without spending a turn.
//
// `!go test ./...` is a question for the machine, not for the model, and
// routing it through a model turn costs a request, a wait and a chunk of the
// context window to answer something the shell already knows. Claude Code,
// OpenCode, Aider and OpenClaw all have this for the same reason.
func (m *Model) shellCommand(line string) tea.Cmd {
	cmdText := strings.TrimSpace(strings.TrimPrefix(line, "!"))
	if cmdText == "" {
		return m.println(styleDim.Render("  ! runs a shell command, e.g. !go test ./..."))
	}

	// The mode still applies. Plan mode is a promise that nothing will be
	// changed, and a promise that the user can step around by typing "!" is
	// not one worth making.
	if m.opts.Broker.Mode() == permission.ModePlan && !permission.IsReadOnlyCommand(cmdText) {
		return m.println(styleWarn.Render("  plan mode is read-only — that command would change something"))
	}

	m.promptHistory.record(line)
	workDir := m.opts.WorkDir
	base := m.baseContext()

	return tea.Batch(
		m.println(bullet(styleTool.Render("!"), styleTool.Render(cmdText))),
		func() tea.Msg {
			ctx, cancel := context.WithTimeout(base, shellTimeout)
			defer cancel()

			cmd := exec.CommandContext(ctx, "sh", "-c", cmdText)
			cmd.Dir = workDir
			out, err := cmd.CombinedOutput()
			return shellResultMsg{command: cmdText, output: string(out), err: err}
		},
	)
}

// renderShellResult formats a finished command for the transcript.
func renderShellResult(msg shellResultMsg) string {
	body := strings.TrimRight(msg.output, "\n")
	lines := strings.Split(body, "\n")
	if body == "" {
		lines = nil
	}

	var b strings.Builder
	for i, line := range lines {
		if i == maxShellOutput {
			fmt.Fprintf(&b, "%s\n", styleDim.Render(fmt.Sprintf("  … %d more lines", len(lines)-maxShellOutput)))
			break
		}
		b.WriteString("  " + line + "\n")
	}
	if msg.err != nil {
		b.WriteString(styleErr.Render("  "+msg.err.Error()) + "\n")
	} else if body == "" {
		b.WriteString(styleDim.Render("  (no output)") + "\n")
	}
	return strings.TrimRight(b.String(), "\n")
}

// --- queued messages ---

// queue holds what was typed while a turn was running.
//
// Before this, enter during a turn returned nil and the keystroke vanished in
// silence. With work-mode turns now allowed up to 200 steps, that stopped
// being a minor annoyance: the moment you most want to say something is while
// it is working.
func (m *Model) enqueue(text string) tea.Cmd {
	m.queued = append(m.queued, text)
	m.promptHistory.record(text)
	return m.println(bullet(styleDim.Render("»"), styleDim.Render(text+"  (queued)")))
}

// flushQueue sends what was queued, once the turn that was running finishes.
func (m *Model) flushQueue() tea.Cmd {
	if len(m.queued) == 0 || m.busy || m.flowBusy || m.opts.Agent == nil {
		return nil
	}
	// Everything queued goes as one message rather than as several turns.
	// Three thoughts had while watching the same piece of work are one piece
	// of feedback, and sending them separately would make the model answer
	// the first before it has read the third.
	text := strings.Join(m.queued, "\n")
	m.queued = nil
	return m.submit(text)
}

// renderQueued draws the pending messages under the input.
func (m *Model) renderQueued(width int) []string {
	if len(m.queued) == 0 {
		return nil
	}
	label := fmt.Sprintf("  %d message queued", len(m.queued))
	if len(m.queued) != 1 {
		label = fmt.Sprintf("  %d messages queued", len(m.queued))
	}
	return []string{fit(styleDim.Render(label+" — esc to discard"), width)}
}

// --- the task list ---

// toggleTodos shows or hides the checklist the model is keeping.
//
// The summary line in the transcript says which step is in progress; this is
// the whole list on demand, which is what Claude Code binds to ctrl+t. The
// state already exists — this only decides whether to draw it.
func (m *Model) toggleTodos() tea.Cmd {
	if m.opts.Todos == nil {
		return m.println(styleDim.Render("  no task list in this session"))
	}
	items := m.opts.Todos.Items()
	if len(items) == 0 {
		return m.println(styleDim.Render("  the task list is empty"))
	}
	m.showTodos = !m.showTodos
	return nil
}

// renderTodos draws the checklist above the input.
func (m *Model) renderTodos(width int) []string {
	if !m.showTodos || m.opts.Todos == nil {
		return nil
	}
	items := m.opts.Todos.Items()
	if len(items) == 0 {
		return nil
	}

	rows := []string{styleDim.Render("  tasks — ctrl+t to hide")}
	for _, it := range items {
		mark, style := "○", styleDim
		switch it.Status {
		case "done":
			mark, style = "✓", styleTool
		case "in_progress":
			mark, style = "▸", styleKiwi
		}
		rows = append(rows, fit("  "+style.Render(mark+" "+it.Content), width))
	}
	return rows
}
