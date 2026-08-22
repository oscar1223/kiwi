package main

import (
	"bytes"
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
	sessionstore "github.com/oscar1223/kiwi/internal/session"
)

func newTestStore(t *testing.T) *sessionstore.Store {
	t.Helper()
	s, err := sessionstore.Open(filepath.Join(t.TempDir(), "sessions.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestResolveSessionDefaultCreatesNew(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	g := &globalFlags{}

	meta, history, err := resolveSession(ctx, store, "/proj", g)
	if err != nil {
		t.Fatalf("resolveSession: %v", err)
	}
	if meta.ID == "" {
		t.Error("no session id assigned")
	}
	if history != nil {
		t.Errorf("a brand-new session should start with no history, got %v", history)
	}
}

func TestResolveSessionContinueResumesTheLatest(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	first, _ := store.Create(ctx, "/proj")
	time.Sleep(1100 * time.Millisecond)
	store.Create(ctx, "/proj") // created after first, but never touched again
	time.Sleep(1100 * time.Millisecond)
	// Appending here bumps first's updated_at past the second session's,
	// making first the latest again despite being older.
	store.Append(ctx, first.ID, []llm.Message{{Role: llm.RoleUser, Content: "earlier"}})

	g := &globalFlags{continueLast: true}
	meta, history, err := resolveSession(ctx, store, "/proj", g)
	if err != nil {
		t.Fatalf("resolveSession: %v", err)
	}
	if meta.ID != first.ID {
		t.Errorf("resolved to %s, want the one just appended to (%s)", meta.ID, first.ID)
	}
	if len(history) != 1 || history[0].Content != "earlier" {
		t.Errorf("history = %+v, want the prior turn loaded", history)
	}
}

// --continue in a project with no history yet must not be an error: a habitual
// flag should degrade to "just start", not block the user.
func TestResolveSessionContinueWithNothingToResumeStartsFresh(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	g := &globalFlags{continueLast: true}

	meta, history, err := resolveSession(ctx, store, "/brand/new/project", g)
	if err != nil {
		t.Fatalf("resolveSession: %v", err)
	}
	if meta.ID == "" {
		t.Error("no session was created")
	}
	if history != nil {
		t.Errorf("history = %v, want nil for a freshly started session", history)
	}
}

func TestResolveSessionResumeByID(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	target, _ := store.Create(ctx, "/proj")
	store.Append(ctx, target.ID, []llm.Message{{Role: llm.RoleUser, Content: "hi"}})

	g := &globalFlags{resumeID: target.ID}
	meta, history, err := resolveSession(ctx, store, "/proj", g)
	if err != nil {
		t.Fatalf("resolveSession: %v", err)
	}
	if meta.ID != target.ID {
		t.Errorf("meta.ID = %s, want %s", meta.ID, target.ID)
	}
	if len(history) != 1 {
		t.Errorf("history = %+v", history)
	}
}

func TestResolveSessionResumeUnknownIDFails(t *testing.T) {
	store := newTestStore(t)
	g := &globalFlags{resumeID: "does-not-exist"}

	_, _, err := resolveSession(context.Background(), store, "/proj", g)
	if err == nil {
		t.Fatal("expected an error for an unknown --resume id")
	}
	if !errors.Is(err, sessionstore.ErrNotFound) {
		t.Errorf("err = %v, want it to wrap ErrNotFound", err)
	}
}

// --resume takes priority over --continue when both are somehow set.
func TestResolveSessionResumeTakesPriorityOverContinue(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()

	latest, _ := store.Create(ctx, "/proj")
	store.Append(ctx, latest.ID, nil)
	older, _ := store.Create(ctx, "/proj")

	g := &globalFlags{continueLast: true, resumeID: older.ID}
	meta, _, err := resolveSession(ctx, store, "/proj", g)
	if err != nil {
		t.Fatalf("resolveSession: %v", err)
	}
	if meta.ID != older.ID {
		t.Errorf("meta.ID = %s, want --resume's target %s, not --continue's latest %s", meta.ID, older.ID, latest.ID)
	}
}

func TestListSessionsEmptyProject(t *testing.T) {
	store := newTestStore(t)
	var out bytes.Buffer
	if err := listSessions(context.Background(), store, "/nothing/here", &out); err != nil {
		t.Fatalf("listSessions: %v", err)
	}
	if !strings.Contains(out.String(), "no sessions") {
		t.Errorf("out = %q", out.String())
	}
}

func TestListSessionsShowsIDAndTitle(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	sess, _ := store.Create(ctx, "/proj")
	store.SetTitle(ctx, sess.ID, "fix the login bug")

	var out bytes.Buffer
	if err := listSessions(ctx, store, "/proj", &out); err != nil {
		t.Fatalf("listSessions: %v", err)
	}
	if !strings.Contains(out.String(), sess.ID) {
		t.Errorf("id missing from output:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "fix the login bug") {
		t.Errorf("title missing from output:\n%s", out.String())
	}
}

func TestListSessionsUntitledFallback(t *testing.T) {
	store := newTestStore(t)
	ctx := context.Background()
	store.Create(ctx, "/proj")

	var out bytes.Buffer
	listSessions(ctx, store, "/proj", &out)
	if !strings.Contains(out.String(), "(untitled)") {
		t.Errorf("out = %q", out.String())
	}
}

func TestRelativeTimeBuckets(t *testing.T) {
	now := time.Now()
	cases := []struct {
		ago  time.Duration
		want string
	}{
		{10 * time.Second, "just now"},
		{5 * time.Minute, "5m ago"},
		{3 * time.Hour, "3h ago"},
		{2 * 24 * time.Hour, "2d ago"},
	}
	for _, c := range cases {
		got := relativeTime(now.Add(-c.ago))
		if got != c.want {
			t.Errorf("relativeTime(-%s) = %q, want %q", c.ago, got, c.want)
		}
	}
}
