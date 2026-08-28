package session

import (
	"context"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
)

// The whole point of sizing from the model: a big-window model must not be
// compacted on a small-window model's schedule.
func TestCompactOptionsScaleWithTheModelWindow(t *testing.T) {
	big := CompactOptionsFor("claude-sonnet-5").TokenBudget
	small := CompactOptionsFor("mixtral-8x7b").TokenBudget

	if big <= small {
		t.Errorf("budget for a 200k model (%d) is not larger than for a 32k one (%d)", big, small)
	}
	if big >= llm.ContextWindow("claude-sonnet-5") {
		t.Errorf("history budget %d leaves no room for the system prompt, tools and reply", big)
	}
}

func TestDefaultCompactOptionsUsesTheFallbackWindow(t *testing.T) {
	if got := DefaultCompactOptions().TokenBudget; got != int(float64(llm.DefaultContextWindow)*historyShare) {
		t.Errorf("DefaultCompactOptions().TokenBudget = %d, want the fallback window's share", got)
	}
	if got := DefaultCompactOptions().KeepRecent; got != DefaultKeepRecent {
		t.Errorf("KeepRecent = %d, want %d", got, DefaultKeepRecent)
	}
}

// /compact asks for a budget of 0: the user deciding the context is cluttered
// outranks any arithmetic about whether it technically had to be compacted.
func TestZeroBudgetCompactsUnconditionally(t *testing.T) {
	history := []llm.Message{
		msg(llm.RoleUser, "old question"),
		msg(llm.RoleAssistant, "old answer"),
		msg(llm.RoleUser, "recent question"),
		msg(llm.RoleAssistant, "recent answer"),
	}
	fake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "the summary"}}}

	got, changed, err := Compact(context.Background(), fake, history, CompactOptions{TokenBudget: 0, KeepRecent: 2})
	if err != nil {
		t.Fatalf("Compact: %v", err)
	}
	if !changed {
		t.Fatal("changed = false: a zero budget must compact even a tiny history")
	}
	if !strings.Contains(got[0].Content, "summary of earlier context") {
		t.Errorf("compacted history does not start with the summary marker: %+v", got[0])
	}
	if got[1].Content != "the summary" {
		t.Errorf("the summarizer's output is missing: %+v", got[1])
	}
}
