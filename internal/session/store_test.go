package session

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
)

func newStore(t *testing.T) *Store {
	t.Helper()
	s, err := Open(filepath.Join(t.TempDir(), "sessions.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestOpenCreatesParentDir(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "dir", "sessions.db")
	s, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	s.Close()
}

func TestCreateAndGet(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	m, err := s.Create(ctx, "/home/user/project")
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if m.ID == "" {
		t.Fatal("empty id")
	}

	got, err := s.Get(ctx, m.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.ProjectDir != "/home/user/project" {
		t.Errorf("ProjectDir = %q", got.ProjectDir)
	}
}

func TestGetUnknownIDReturnsErrNotFound(t *testing.T) {
	s := newStore(t)
	if _, err := s.Get(context.Background(), "nope"); err != ErrNotFound {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

func TestGetByUniquePrefix(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	m, _ := s.Create(ctx, "/proj")

	got, err := s.Get(ctx, m.ID[:4])
	if err != nil {
		t.Fatalf("Get by prefix: %v", err)
	}
	if got.ID != m.ID {
		t.Errorf("resolved to %q, want %q", got.ID, m.ID)
	}
}

// Two projects must never see each other's "latest" session — that would
// resume the wrong conversation into the wrong repository.
func TestLatestIsScopedPerProject(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	a1, _ := s.Create(ctx, "/proj/a")
	time.Sleep(1100 * time.Millisecond) // updated_at has second resolution
	_, _ = s.Create(ctx, "/proj/b")

	latestA, err := s.Latest(ctx, "/proj/a")
	if err != nil {
		t.Fatalf("Latest: %v", err)
	}
	if latestA.ID != a1.ID {
		t.Errorf("Latest(/proj/a) = %s, want %s", latestA.ID, a1.ID)
	}
}

func TestLatestNoneYet(t *testing.T) {
	s := newStore(t)
	if _, err := s.Latest(context.Background(), "/nowhere"); err != ErrNotFound {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}

// Appending must bump updated_at so the session just used becomes "latest",
// which is the whole point of --continue.
func TestAppendMakesSessionLatest(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	old, _ := s.Create(ctx, "/proj")
	time.Sleep(1100 * time.Millisecond)
	fresh, _ := s.Create(ctx, "/proj")

	if err := s.Append(ctx, old.ID, []llm.Message{{Role: llm.RoleUser, Content: "hi"}}); err != nil {
		t.Fatalf("Append: %v", err)
	}

	latest, err := s.Latest(ctx, "/proj")
	if err != nil {
		t.Fatalf("Latest: %v", err)
	}
	if latest.ID != old.ID {
		t.Errorf("Latest = %s, want %s (the one just appended to, not %s)", latest.ID, old.ID, fresh.ID)
	}
}

func TestAppendAndLoadRoundTrip(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sess, _ := s.Create(ctx, "/proj")

	msgs := []llm.Message{
		{Role: llm.RoleUser, Content: "read main.go"},
		{Role: llm.RoleAssistant, Content: "sure", ToolCalls: []llm.ToolCall{
			{ID: "c1", Name: "read_file", Input: []byte(`{"path":"main.go"}`)},
		}},
		{Role: llm.RoleTool, Content: "package main", ToolCallID: "c1", ToolName: "read_file"},
		{Role: llm.RoleAssistant, Content: "here it is"},
	}
	if err := s.Append(ctx, sess.ID, msgs); err != nil {
		t.Fatalf("Append: %v", err)
	}

	got, err := s.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got) != len(msgs) {
		t.Fatalf("loaded %d messages, want %d", len(got), len(msgs))
	}
	for i, want := range msgs {
		if got[i].Role != want.Role || got[i].Content != want.Content {
			t.Errorf("[%d] = %+v, want %+v", i, got[i], want)
		}
	}
	if len(got[1].ToolCalls) != 1 || got[1].ToolCalls[0].Name != "read_file" {
		t.Errorf("tool calls not round-tripped: %+v", got[1].ToolCalls)
	}
	if string(got[1].ToolCalls[0].Input) != `{"path":"main.go"}` {
		t.Errorf("tool call input = %s", got[1].ToolCalls[0].Input)
	}
	if !got[2].IsError && got[2].ToolCallID != "c1" {
		t.Errorf("tool result linkage lost: %+v", got[2])
	}
}

// Append must be additive across turns, preserving order — this is what
// makes multiple `kiwi ask --continue` calls compose into one conversation.
func TestAppendAcrossMultipleTurnsPreservesOrder(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sess, _ := s.Create(ctx, "/proj")

	s.Append(ctx, sess.ID, []llm.Message{{Role: llm.RoleUser, Content: "first"}})
	s.Append(ctx, sess.ID, []llm.Message{{Role: llm.RoleAssistant, Content: "second"}})
	s.Append(ctx, sess.ID, []llm.Message{{Role: llm.RoleUser, Content: "third"}})

	got, err := s.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	want := []string{"first", "second", "third"}
	if len(got) != 3 {
		t.Fatalf("got %d messages", len(got))
	}
	for i, w := range want {
		if got[i].Content != w {
			t.Errorf("[%d] = %q, want %q", i, got[i].Content, w)
		}
	}
}

func TestReplaceOverwritesMessages(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sess, _ := s.Create(ctx, "/proj")

	s.Append(ctx, sess.ID, []llm.Message{
		{Role: llm.RoleUser, Content: "a"},
		{Role: llm.RoleAssistant, Content: "b"},
		{Role: llm.RoleUser, Content: "c"},
	})

	condensed := []llm.Message{
		{Role: llm.RoleUser, Content: "summary marker"},
		{Role: llm.RoleAssistant, Content: "summary text"},
		{Role: llm.RoleUser, Content: "c"},
	}
	if err := s.Replace(ctx, sess.ID, condensed); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	got, err := s.Load(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(got) != 3 || got[0].Content != "summary marker" {
		t.Errorf("got %+v", got)
	}
}

func TestListOrdersByMostRecentlyUpdated(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()

	a, _ := s.Create(ctx, "/proj")
	time.Sleep(1100 * time.Millisecond)
	b, _ := s.Create(ctx, "/proj")

	list, err := s.List(ctx, "/proj", 10)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 {
		t.Fatalf("got %d sessions", len(list))
	}
	if list[0].ID != b.ID || list[1].ID != a.ID {
		t.Errorf("order = [%s, %s], want most recent first [%s, %s]", list[0].ID, list[1].ID, b.ID, a.ID)
	}
}

func TestListDoesNotLeakOtherProjects(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	s.Create(ctx, "/proj/a")
	s.Create(ctx, "/proj/a")
	s.Create(ctx, "/proj/b")

	list, err := s.List(ctx, "/proj/a", 10)
	if err != nil {
		t.Fatalf("List: %v", err)
	}
	if len(list) != 2 {
		t.Errorf("got %d sessions for /proj/a, want 2", len(list))
	}
}

func TestSetTitle(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sess, _ := s.Create(ctx, "/proj")

	if err := s.SetTitle(ctx, sess.ID, "fix the login bug"); err != nil {
		t.Fatalf("SetTitle: %v", err)
	}
	got, err := s.Get(ctx, sess.ID)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if got.Title != "fix the login bug" {
		t.Errorf("Title = %q", got.Title)
	}
}

func TestProjectHashIsStableAndDistinct(t *testing.T) {
	h1 := ProjectHash("/home/user/project")
	h2 := ProjectHash("/home/user/project")
	if h1 != h2 {
		t.Error("ProjectHash is not deterministic")
	}
	if h1 == ProjectHash("/home/user/other") {
		t.Error("different projects hashed to the same value")
	}
	// Cosmetic differences in spelling the same path must not fragment a
	// project's session history across two hashes.
	if ProjectHash("/home/user/project/") != h1 {
		t.Error("a trailing slash should not change the hash")
	}
}

func TestAppendEmptyIsANoop(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	sess, _ := s.Create(ctx, "/proj")

	if err := s.Append(ctx, sess.ID, nil); err != nil {
		t.Fatalf("Append(nil): %v", err)
	}
	got, _ := s.Load(ctx, sess.ID)
	if len(got) != 0 {
		t.Errorf("got %d messages from an empty append", len(got))
	}
}

func TestGetAmbiguousPrefix(t *testing.T) {
	s := newStore(t)
	ctx := context.Background()
	// Force a collision by sharing a fabricated single-character prefix.
	s.db.ExecContext(ctx, `INSERT INTO sessions (id, project_dir, project_hash, title, created_at, updated_at)
		VALUES ('aaaa01', '/x', 'h', '', 0, 0), ('aaaa02', '/x', 'h', '', 0, 0)`)

	_, err := s.Get(ctx, "aaaa")
	if err == nil {
		t.Fatal("expected an ambiguous-prefix error")
	}
	if !strings.Contains(err.Error(), "matches 2 sessions") {
		t.Errorf("err = %v", err)
	}
}
