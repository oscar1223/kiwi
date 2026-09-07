package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// Todo statuses. Anything else is rejected, because a typo that silently
// became "not done" would quietly derail a long task.
const (
	TodoPending    = "pending"
	TodoInProgress = "in_progress"
	TodoDone       = "done"
)

// maxTodos bounds the list. A plan longer than this is not a plan.
const maxTodos = 40

// TodoItem is one step of the current job.
type TodoItem struct {
	Content string `json:"content"`
	Status  string `json:"status"`
}

// TodoList is the working checklist for one session.
//
// It lives for as long as the session and is deliberately not persisted: it
// tracks the job in front of the model right now, not something worth
// remembering across runs — that is what remember is for.
//
// The mutex is not ceremony: tools can be invoked concurrently within a turn.
type TodoList struct {
	mu    sync.Mutex
	items []TodoItem
}

func (l *TodoList) set(items []TodoItem) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.items = items
}

// Items is get() for callers outside this package — the TUI, which draws the
// same list the model is working from rather than keeping a copy that could
// drift out of step with it.
func (l *TodoList) Items() []TodoItem { return l.get() }

func (l *TodoList) get() []TodoItem {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]TodoItem(nil), l.items...)
}

// render draws the list the way it is shown to both the model and the user.
func (l *TodoList) render() string {
	items := l.get()
	if len(items) == 0 {
		return "(the todo list is empty)"
	}

	var b strings.Builder
	done := 0
	for _, it := range items {
		switch it.Status {
		case TodoDone:
			done++
			b.WriteString("[x] ")
		case TodoInProgress:
			b.WriteString("[>] ")
		default:
			b.WriteString("[ ] ")
		}
		b.WriteString(it.Content)
		b.WriteString("\n")
	}
	fmt.Fprintf(&b, "\n%d/%d done", done, len(items))
	return b.String()
}

// --- todo_write ---

// TodoWrite replaces the checklist.
type TodoWrite struct{ List *TodoList }

func (TodoWrite) Name() string { return "todo_write" }

func (TodoWrite) Description() string {
	return "Record the plan for the job you are doing, and keep it current. " +
		"Pass the whole list every time — it replaces the previous one. Mark a step " +
		"in_progress before starting it and done the moment it is finished, one " +
		"in_progress at a time. Use it for anything that takes more than two or " +
		"three steps; skip it for single-step work."
}

func (TodoWrite) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"todos": map[string]any{
				"type":        "array",
				"description": "The complete list, in order. Replaces whatever was there before.",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"content": map[string]any{"type": "string", "description": "What the step is."},
						"status": map[string]any{
							"type":        "string",
							"enum":        []string{TodoPending, TodoInProgress, TodoDone},
							"description": "pending, in_progress or done.",
						},
					},
					"required": []string{"content", "status"},
				},
			},
		},
		"required": []string{"todos"},
	}
}

func (t TodoWrite) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Todos []TodoItem `json:"todos"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if len(in.Todos) > maxTodos {
		return "", fmt.Errorf("%d steps is too many (max %d): break the work up instead", len(in.Todos), maxTodos)
	}

	active := 0
	for i, it := range in.Todos {
		if strings.TrimSpace(it.Content) == "" {
			return "", fmt.Errorf("step %d has no content", i+1)
		}
		switch it.Status {
		case TodoPending, TodoDone:
		case TodoInProgress:
			active++
		default:
			return "", fmt.Errorf("step %d has status %q, want %s, %s or %s",
				i+1, it.Status, TodoPending, TodoInProgress, TodoDone)
		}
	}
	// Two things in progress at once is how a long job quietly loses track of
	// where it was.
	if active > 1 {
		return "", fmt.Errorf("%d steps are in_progress at once: work on one at a time", active)
	}

	t.List.set(in.Todos)
	return t.List.render(), nil
}

// --- todo_read ---

// TodoRead returns the checklist as it stands.
type TodoRead struct{ List *TodoList }

func (TodoRead) Name() string { return "todo_read" }

func (TodoRead) Description() string {
	return "Show the current todo list. Use it to re-orient yourself in a long job."
}

func (TodoRead) Schema() map[string]any {
	return map[string]any{"type": "object", "properties": map[string]any{}}
}

func (t TodoRead) Run(ctx context.Context, input json.RawMessage) (string, error) {
	return t.List.render(), nil
}
