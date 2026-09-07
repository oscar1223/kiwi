package tui

import (
	"context"
	"errors"
	"fmt"
	"strings"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/checkpoint"
)

// turnMark ties a snapshot of the files to the point in the conversation it
// was taken at.
//
// Both halves are needed. Restoring the files without rewinding the
// conversation would leave the model certain of an edit that no longer exists
// on disk, and it would go on reasoning from that certainty — which is a worse
// state to be in than the one the user was trying to undo.
type turnMark struct {
	ID         string
	HistoryLen int
	Label      string
}

// checkpointReadyMsg carries the result of opening the snapshot store, which
// runs off the UI goroutine because it shells out to git.
type checkpointReadyMsg struct {
	store *checkpoint.Store
	err   error
}

// checkpointTakenMsg reports a snapshot taken just before a turn.
type checkpointTakenMsg struct {
	gen  int
	mark turnMark
}

// checkpointRestoredMsg reports a completed /undo or /redo: what was restored,
// and what the opposite command would restore next.
type checkpointRestoredMsg struct {
	restored turnMark
	opposite turnMark
	// undo distinguishes the two directions, since redo pushes onto the undo
	// stack and undo pushes onto the redo stack.
	undo  bool
	files []string
}

// initCheckpoints opens the snapshot store in the background.
//
// It is deliberately not part of session setup: on a large repository the
// first git call takes long enough to be felt, and there is nothing to
// snapshot until the user actually sends something.
func (m *Model) initCheckpoints() tea.Cmd {
	workDir := m.opts.WorkDir
	base := m.baseContext()
	return func() tea.Msg {
		store, err := checkpoint.New(base, workDir)
		return checkpointReadyMsg{store: store, err: err}
	}
}

// baseContext is the session context every background call inherits.
func (m *Model) baseContext() context.Context {
	if m.opts.BaseContext != nil {
		return m.opts.BaseContext
	}
	return context.Background()
}

// onCheckpointReady records the store, or explains once why there is none.
func (m *Model) onCheckpointReady(msg checkpointReadyMsg) tea.Cmd {
	if msg.err != nil {
		switch {
		case errors.Is(msg.err, checkpoint.ErrNoRepo):
			// Said plainly and once. Kiwi is often launched from a directory
			// that is not a repository, and a warning on every turn would
			// teach people to stop reading the warnings that matter.
			return m.println(styleDim.Render("  not a git repository — /undo and /diff are unavailable here"))
		case errors.Is(msg.err, checkpoint.ErrNoGit):
			return m.println(styleDim.Render("  git is not installed — /undo and /diff are unavailable"))
		}
		return m.println(styleDim.Render("  checkpoints unavailable: " + msg.err.Error()))
	}
	m.checkpoints = msg.store
	return nil
}

// snapshotFn returns the function a turn calls before it starts working.
//
// Everything it needs is captured here, on the UI goroutine, so the returned
// closure touches no Model state — the same discipline runFlow follows. The
// snapshot must also complete before the agent's first edit, or it would
// record a tree that was already half-modified, which is why the caller runs
// it at the head of the turn's own goroutine rather than beside it.
func (m *Model) snapshotFn(gen int, label string) func(context.Context) {
	store, events, histLen := m.checkpoints, m.events, len(m.history)
	if store == nil {
		return func(context.Context) {}
	}
	return func(ctx context.Context) {
		id, err := store.Take(ctx, label)
		if err != nil {
			// A failed snapshot must not stop the turn: the user asked for
			// work, not for a backup. It is reported, and the turn goes ahead
			// without a net.
			events.send(ctx, systemMsg{"could not take a checkpoint: " + err.Error()})
			return
		}
		events.send(ctx, checkpointTakenMsg{gen: gen, mark: turnMark{
			ID:         id,
			HistoryLen: histLen,
			Label:      label,
		}})
	}
}

// --- /undo and /redo ---

// checkpointState is everything a checkpoint flow needs, captured on the UI
// goroutine before the flow starts. Flows never read Model fields directly,
// for the same reason /compact is handed a copy of the history rather than
// reaching for it.
type checkpointState struct {
	store   *checkpoint.Store
	marks   []turnMark
	undone  []turnMark
	histLen int
}

// checkpointState snapshots what the flows need.
func (m *Model) checkpointState() checkpointState {
	return checkpointState{
		store:   m.checkpoints,
		marks:   append([]turnMark(nil), m.marks...),
		undone:  append([]turnMark(nil), m.undone...),
		histLen: len(m.history),
	}
}

// undoFlow restores the files and the conversation to just before the last
// turn that ran.
func (m *Model) undoFlow(ctx context.Context, st checkpointState) {
	if !m.checkpointsReady(ctx, st) {
		return
	}
	if len(st.marks) == 0 {
		m.events.send(ctx, systemMsg{"Nothing to undo — no turn has run in this session yet."})
		return
	}
	m.restore(ctx, st, st.marks[len(st.marks)-1], true)
}

// redoFlow puts back what the last /undo took away.
func (m *Model) redoFlow(ctx context.Context, st checkpointState) {
	if !m.checkpointsReady(ctx, st) {
		return
	}
	if len(st.undone) == 0 {
		m.events.send(ctx, systemMsg{"Nothing to redo."})
		return
	}
	m.restore(ctx, st, st.undone[len(st.undone)-1], false)
}

// restore is the shared body of undo and redo: describe what will change,
// confirm, snapshot the current state so the move is reversible, then restore.
func (m *Model) restore(ctx context.Context, st checkpointState, mark turnMark, undo bool) {
	verb := "Redo"
	if undo {
		verb = "Undo"
	}

	files, err := st.store.Files(ctx, mark.ID)
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	if len(files) == 0 && undo {
		// Worth saying rather than silently doing nothing: the user expected
		// something to change, and the reason it will not is that the turn
		// never wrote anything.
		m.events.send(ctx, systemMsg{"That turn changed no files. Rewinding the conversation only."})
	}

	if len(files) > 0 {
		m.events.send(ctx, printLinesMsg{lines: describeFiles(verb, files)})
		if !m.events.Confirm(ctx, fmt.Sprintf("%s these changes? Anything you edited by hand since then goes too.", verb)) {
			m.events.send(ctx, systemMsg{strings.ToLower(verb) + " cancelled"})
			return
		}
	}

	// The state being left behind becomes the other direction's target, which
	// is what makes undo and redo symmetric rather than one-way.
	opposite := turnMark{HistoryLen: st.histLen, Label: mark.Label}
	if id, err := st.store.Take(ctx, "before "+strings.ToLower(verb)); err == nil {
		opposite.ID = id
	}

	if err := st.store.Restore(ctx, mark.ID); err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	m.events.send(ctx, checkpointRestoredMsg{
		restored: mark,
		opposite: opposite,
		undo:     undo,
		files:    files,
	})
}

// onCheckpointRestored applies a finished undo or redo to the model: the files
// are already back, and this is the half that rewinds the conversation to
// match.
func (m *Model) onCheckpointRestored(msg checkpointRestoredMsg) tea.Cmd {
	if msg.restored.HistoryLen <= len(m.history) {
		m.history = m.history[:msg.restored.HistoryLen]
	}

	verb := "undone"
	if msg.undo {
		m.marks = m.marks[:len(m.marks)-1]
		if msg.opposite.ID != "" {
			m.undone = append(m.undone, msg.opposite)
		}
	} else {
		verb = "redone"
		m.undone = m.undone[:len(m.undone)-1]
		if msg.opposite.ID != "" {
			m.marks = append(m.marks, msg.opposite)
		}
	}

	summary := fmt.Sprintf("%d file", len(msg.files))
	if len(msg.files) != 1 {
		summary += "s"
	}
	if len(msg.files) == 0 {
		summary = "no files"
	}
	return m.println(bullet(
		styleTool.Render("↩"),
		styleTool.Render(verb)+styleDim.Render(fmt.Sprintf(" — %s restored, conversation rewound to %d messages", summary, len(m.history))),
	))
}

// --- /diff ---

// diffFlow shows what changed since the session started, or since the last
// turn with "/diff turn".
//
// This command exists because of work mode. The diff used to be visible in the
// permission prompt for every edit; work mode approves edits without a prompt,
// so the diff stopped being shown at all, and there was no way to ask for it.
func (m *Model) diffFlow(ctx context.Context, st checkpointState, arg string) {
	if !m.checkpointsReady(ctx, st) {
		return
	}
	if len(st.marks) == 0 {
		m.events.send(ctx, systemMsg{"No turn has run yet, so there is nothing to compare against."})
		return
	}

	from := st.marks[0]
	scope := "this session"
	if arg == "turn" || arg == "last" {
		from = st.marks[len(st.marks)-1]
		scope = "the last turn"
	}

	diff, err := st.store.Diff(ctx, from.ID)
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	if strings.TrimSpace(diff) == "" {
		m.events.send(ctx, systemMsg{"No changes since " + scope + " started."})
		return
	}
	m.events.send(ctx, printLinesMsg{lines: []string{
		styleDim.Render("  changes since " + scope + " started"),
		renderDiff(diff, maxDiffLines),
	}})
}

// maxDiffLines caps what /diff prints. A diff longer than this is not read in
// a transcript anyway, and pasting a thousand lines into the window costs the
// scroll position of everything above it.
const maxDiffLines = 200

// checkpointsReady reports whether snapshots are available, explaining why not
// when they are not.
func (m *Model) checkpointsReady(ctx context.Context, st checkpointState) bool {
	if st.store != nil {
		return true
	}
	m.events.send(ctx, systemMsg{"Checkpoints are unavailable here — this needs a git repository and git on PATH."})
	return false
}

// describeFiles renders the file list shown before a restore is confirmed,
// capped so a large turn does not push the confirmation prompt off screen.
func describeFiles(verb string, files []string) []string {
	const max = 12
	lines := []string{styleDim.Render(fmt.Sprintf("  %s will touch %d file(s):", strings.ToLower(verb), len(files)))}
	for i, f := range files {
		if i == max {
			lines = append(lines, styleDim.Render(fmt.Sprintf("    … and %d more", len(files)-max)))
			break
		}
		lines = append(lines, styleDim.Render("    "+f))
	}
	return lines
}
