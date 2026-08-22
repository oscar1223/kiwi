package session

import (
	"context"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
)

func msg(role llm.Role, content string) llm.Message {
	return llm.Message{Role: role, Content: content}
}

func TestCompactNoopUnderBudget(t *testing.T) {
	history := []llm.Message{msg(llm.RoleUser, "hi"), msg(llm.RoleAssistant, "hello")}
	fake := &llmtest.Fake{}

	got, changed, err := Compact(context.Background(), fake, history, DefaultCompactOptions())
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if changed {
		t.Error("changed = true, want false: history is under budget")
	}
	if fake.Calls() != 0 {
		t.Error("the provider should not be consulted when nothing needs compacting")
	}
	if len(got) != len(history) {
		t.Errorf("got %d messages, want the original %d back", len(got), len(history))
	}
}

// The property that matters most: the cut can never separate an assistant's
// tool_calls from the tool results that answer them, because both provider
// wire formats require them adjacent. Build a history where a naive
// "keep last N" would slice straight through a tool round-trip, and confirm
// the boundary was pushed out to the next user message instead.
func TestCompactNeverSplitsAToolRoundTrip(t *testing.T) {
	history := []llm.Message{
		msg(llm.RoleUser, strings.Repeat("filler ", 2000)), // old turn, pushes over budget
		msg(llm.RoleAssistant, "ok"),

		msg(llm.RoleUser, "run the tool"),
		{Role: llm.RoleAssistant, ToolCalls: []llm.ToolCall{{ID: "c1", Name: "bash"}}},
		{Role: llm.RoleTool, Content: "output", ToolCallID: "c1", ToolName: "bash"},
		msg(llm.RoleAssistant, "done"),
	}
	fake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "summary"}}}

	// KeepRecent=3 lands exactly on the tool-result message if taken
	// literally: len(history)=6, cut=3 -> history[3] is the assistant message
	// that owns the tool call, still mid round-trip.
	opts := CompactOptions{CharBudget: 10, KeepRecent: 3}
	got, changed, err := Compact(context.Background(), fake, history, opts)
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !changed {
		t.Fatal("expected compaction to fire")
	}

	// The recent, uncompacted tail must start at "run the tool" (a user
	// message) and contain the full, unbroken round-trip after it.
	idx := indexOfContent(got, "run the tool")
	if idx < 0 {
		t.Fatalf("the safe turn boundary was not preserved:\n%+v", got)
	}
	if got[idx].Role != llm.RoleUser {
		t.Fatalf("cut landed on %s, not a user message", got[idx].Role)
	}
	rest := got[idx:]
	if len(rest) != 4 {
		t.Fatalf("the round-trip after the cut was split: got %d messages, want 4:\n%+v", len(rest), rest)
	}
	if rest[1].Role != llm.RoleAssistant || len(rest[1].ToolCalls) == 0 {
		t.Errorf("tool_calls message missing or moved: %+v", rest[1])
	}
	if rest[2].Role != llm.RoleTool || rest[2].ToolCallID != "c1" {
		t.Errorf("tool result missing or detached from its call: %+v", rest[2])
	}
}

func TestCompactSummarizesTheOldPrefix(t *testing.T) {
	history := []llm.Message{
		msg(llm.RoleUser, strings.Repeat("old ", 3000)),
		msg(llm.RoleAssistant, strings.Repeat("older ", 3000)),
		msg(llm.RoleUser, "recent question"),
		msg(llm.RoleAssistant, "recent answer"),
	}
	fake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "condensed summary"}}}

	got, changed, err := Compact(context.Background(), fake, history, CompactOptions{CharBudget: 10, KeepRecent: 2})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !changed {
		t.Fatal("expected compaction to fire")
	}
	if len(got) != 4 { // 2 summary messages + 2 kept
		t.Fatalf("got %d messages, want 4:\n%+v", len(got), got)
	}
	if got[0].Role != llm.RoleUser || got[1].Role != llm.RoleAssistant {
		t.Errorf("summary pair has wrong roles: %+v, %+v", got[0], got[1])
	}
	if got[1].Content != "condensed summary" {
		t.Errorf("summary content = %q", got[1].Content)
	}
	if got[2].Content != "recent question" || got[3].Content != "recent answer" {
		t.Errorf("recent tail not preserved: %+v", got[2:])
	}

	// The provider must have been asked to summarize, not the tool loop.
	if fake.Calls() != 1 {
		t.Fatalf("provider called %d times, want exactly 1", fake.Calls())
	}
	if len(fake.Requests[0].Tools) != 0 {
		t.Error("the summarization request should carry no tools")
	}
}

// A single oversized turn with nothing safe to cut before it must be left
// alone rather than corrupted.
func TestCompactSkipsWhenNoSafeBoundaryExists(t *testing.T) {
	history := []llm.Message{
		msg(llm.RoleUser, strings.Repeat("huge ", 5000)),
		msg(llm.RoleAssistant, "ok"),
	}
	fake := &llmtest.Fake{}

	got, changed, err := Compact(context.Background(), fake, history, CompactOptions{CharBudget: 10, KeepRecent: 1})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if changed {
		t.Error("changed = true, but there was no safe place to cut")
	}
	if len(got) != len(history) {
		t.Error("history was altered despite there being no safe cut")
	}
	if fake.Calls() != 0 {
		t.Error("the provider should not be called when compaction is skipped")
	}
}

func TestCompactPropagatesSummarizerError(t *testing.T) {
	history := []llm.Message{
		msg(llm.RoleUser, strings.Repeat("x", 5000)),
		msg(llm.RoleAssistant, "y"),
		msg(llm.RoleUser, "z"),
	}
	fake := &llmtest.Fake{Steps: []llmtest.Step{{Err: errBoom}}}

	_, _, err := Compact(context.Background(), fake, history, CompactOptions{CharBudget: 10, KeepRecent: 1})
	if err == nil {
		t.Fatal("expected the summarizer's error to propagate")
	}
}

type boomErr struct{}

func (boomErr) Error() string { return "provider exploded" }

var errBoom = boomErr{}

func indexOfContent(msgs []llm.Message, content string) int {
	for i, m := range msgs {
		if m.Content == content {
			return i
		}
	}
	return -1
}
