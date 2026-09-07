package tui

import (
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
)

func msgs(n int) []llm.Message {
	out := make([]llm.Message, n)
	for i := range out {
		out[i] = llm.Message{Role: llm.RoleUser, Content: "m"}
	}
	return out
}

func TestACheckpointFromACancelledTurnIsDropped(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeWork)
	m.gen = 4

	m.Update(checkpointTakenMsg{gen: 4, mark: turnMark{ID: "aaa", HistoryLen: 2}})
	if len(m.marks) != 1 {
		t.Fatalf("the current turn's checkpoint was not recorded: %v", m.marks)
	}

	// A turn that was cancelled and superseded: its snapshot must not become
	// something /undo would restore, or undo and the transcript would disagree
	// about which turn was the last one.
	m.gen = 5
	m.Update(checkpointTakenMsg{gen: 4, mark: turnMark{ID: "bbb", HistoryLen: 2}})
	if len(m.marks) != 1 {
		t.Errorf("a stale turn's checkpoint was recorded: %v", m.marks)
	}
}

func TestANewTurnDiscardsTheRedoHistory(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeWork)
	m.undone = []turnMark{{ID: "undone"}}
	m.gen = 1

	m.Update(checkpointTakenMsg{gen: 1, mark: turnMark{ID: "fresh", HistoryLen: 0}})

	if len(m.undone) != 0 {
		t.Error("redo history survived new work landing on top of it")
	}
}

func TestUndoRewindsTheConversationAndOffersRedo(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeWork)
	m.history = msgs(6)
	m.marks = []turnMark{{ID: "first", HistoryLen: 0}, {ID: "second", HistoryLen: 4}}

	m.Update(checkpointRestoredMsg{
		restored: turnMark{ID: "second", HistoryLen: 4},
		opposite: turnMark{ID: "before-undo", HistoryLen: 6},
		undo:     true,
		files:    []string{"main.go"},
	})

	// Restoring the files without rewinding the conversation would leave the
	// model certain of an edit that is no longer on disk.
	if len(m.history) != 4 {
		t.Errorf("history has %d messages, want it rewound to 4", len(m.history))
	}
	if len(m.marks) != 1 || m.marks[0].ID != "first" {
		t.Errorf("marks = %v, want the undone turn popped", m.marks)
	}
	if len(m.undone) != 1 || m.undone[0].ID != "before-undo" {
		t.Errorf("undone = %v, want the pre-undo state kept for /redo", m.undone)
	}
	if got := plain(strings.Join(m.transcript.render(80), "\n")); !strings.Contains(got, "undone") {
		t.Errorf("the undo was not reported to the user: %q", got)
	}
}

func TestRedoIsTheExactInverseOfUndo(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeWork)
	m.history = msgs(4)
	m.marks = []turnMark{{ID: "first", HistoryLen: 0}}
	m.undone = []turnMark{{ID: "before-undo", HistoryLen: 6}}

	m.Update(checkpointRestoredMsg{
		restored: turnMark{ID: "before-undo", HistoryLen: 6},
		opposite: turnMark{ID: "before-redo", HistoryLen: 4},
		undo:     false,
		files:    []string{"main.go"},
	})

	if len(m.undone) != 0 {
		t.Errorf("undone = %v, want the redone entry consumed", m.undone)
	}
	if len(m.marks) != 2 || m.marks[1].ID != "before-redo" {
		t.Errorf("marks = %v, want the pre-redo state pushed back for /undo", m.marks)
	}
}

// A restore target recorded when the conversation was longer than it is now —
// possible after /clear or a session switch — must not slice past the end.
func TestRestoringPastTheEndOfHistoryIsSafe(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeWork)
	m.history = msgs(2)
	m.marks = []turnMark{{ID: "stale", HistoryLen: 99}}

	m.Update(checkpointRestoredMsg{
		restored: turnMark{ID: "stale", HistoryLen: 99},
		opposite: turnMark{ID: "before-undo", HistoryLen: 2},
		undo:     true,
	})

	if len(m.history) != 2 {
		t.Errorf("history = %d messages, want it left alone", len(m.history))
	}
}

func TestCheckpointCommandsAreRegistered(t *testing.T) {
	for _, name := range []string{"/undo", "/redo", "/diff"} {
		if !isKnownCommand(name) {
			t.Errorf("%s is not in the command registry", name)
		}
		if !strings.Contains(plain(helpText()), name) {
			t.Errorf("%s is missing from /help", name)
		}
	}
}

func TestCheckpointStateIsACopy(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeWork)
	m.marks = []turnMark{{ID: "first"}}
	st := m.checkpointState()

	// The flow runs on its own goroutine; it must not observe later mutations.
	m.marks = append(m.marks, turnMark{ID: "second"})
	if len(st.marks) != 1 {
		t.Errorf("checkpointState shares its backing array with the model: %v", st.marks)
	}
}

func TestCheckpointCommandsReportWhenUnavailable(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeWork)
	if m.checkpoints != nil {
		t.Fatal("the test model unexpectedly has a checkpoint store")
	}
	for _, name := range []string{"/undo", "/redo", "/diff"} {
		if _, handled := m.command(name); !handled {
			t.Errorf("%s was not handled and would have been sent to the model", name)
		}
	}
}
