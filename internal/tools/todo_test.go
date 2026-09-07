package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
)

func writeTodos(t *testing.T, tool TodoWrite, todos ...TodoItem) (string, error) {
	t.Helper()
	input, err := json.Marshal(map[string]any{"todos": todos})
	if err != nil {
		t.Fatal(err)
	}
	return tool.Run(context.Background(), input)
}

func TestTodoWriteReplacesTheWholeList(t *testing.T) {
	list := &TodoList{}
	w := TodoWrite{List: list}

	if _, err := writeTodos(t, w,
		TodoItem{"uno", TodoDone},
		TodoItem{"dos", TodoInProgress},
		TodoItem{"tres", TodoPending},
	); err != nil {
		t.Fatal(err)
	}

	// A second write replaces rather than merges: no ids to keep in sync.
	out, err := writeTodos(t, w, TodoItem{"otra cosa", TodoPending})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out, "uno") {
		t.Errorf("the second write merged instead of replacing:\n%s", out)
	}
	if !strings.Contains(out, "otra cosa") {
		t.Errorf("the new list is missing:\n%s", out)
	}
	if !strings.Contains(out, "0/1 done") {
		t.Errorf("the tally is wrong:\n%s", out)
	}
}

func TestTodoWriteRendersStatuses(t *testing.T) {
	list := &TodoList{}
	out, err := writeTodos(t, TodoWrite{List: list},
		TodoItem{"hecho", TodoDone},
		TodoItem{"en marcha", TodoInProgress},
		TodoItem{"pendiente", TodoPending},
	)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"[x] hecho", "[>] en marcha", "[ ] pendiente", "1/3 done"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
}

// Two things in progress is how a long job loses track of where it was, so it
// is refused rather than quietly accepted.
func TestTodoWriteRejectsTwoInProgress(t *testing.T) {
	list := &TodoList{}
	_, err := writeTodos(t, TodoWrite{List: list},
		TodoItem{"uno", TodoInProgress},
		TodoItem{"dos", TodoInProgress},
	)
	if err == nil {
		t.Fatal("two in_progress steps were accepted")
	}
	if !strings.Contains(err.Error(), "one at a time") {
		t.Errorf("the error does not say what to do instead: %v", err)
	}
}

// A typo in a status must not silently become "not done".
func TestTodoWriteRejectsAnUnknownStatus(t *testing.T) {
	list := &TodoList{}
	if _, err := writeTodos(t, TodoWrite{List: list}, TodoItem{"uno", "doing"}); err == nil {
		t.Error("an unknown status was accepted")
	}
}

func TestTodoWriteRejectsEmptyContent(t *testing.T) {
	list := &TodoList{}
	if _, err := writeTodos(t, TodoWrite{List: list}, TodoItem{"  ", TodoPending}); err == nil {
		t.Error("a step with no content was accepted")
	}
}

// A rejected write must leave the previous list untouched: half-applying it
// would be worse than refusing.
func TestTodoWriteLeavesTheListAloneOnError(t *testing.T) {
	list := &TodoList{}
	w := TodoWrite{List: list}

	if _, err := writeTodos(t, w, TodoItem{"bueno", TodoPending}); err != nil {
		t.Fatal(err)
	}
	if _, err := writeTodos(t, w, TodoItem{"malo", "inventado"}); err == nil {
		t.Fatal("the invalid write was accepted")
	}

	if out := list.render(); !strings.Contains(out, "bueno") || strings.Contains(out, "malo") {
		t.Errorf("a rejected write changed the list:\n%s", out)
	}
}

func TestTodoReadReportsAnEmptyList(t *testing.T) {
	list := &TodoList{}
	out, err := TodoRead{List: list}.Run(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "empty") {
		t.Errorf("an empty list should say so: %q", out)
	}
}

func TestTodoReadSeesWhatWriteStored(t *testing.T) {
	list := &TodoList{}
	if _, err := writeTodos(t, TodoWrite{List: list}, TodoItem{"compartido", TodoInProgress}); err != nil {
		t.Fatal(err)
	}

	out, err := TodoRead{List: list}.Run(context.Background(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "compartido") {
		t.Errorf("read and write are not sharing the same list:\n%s", out)
	}
}
