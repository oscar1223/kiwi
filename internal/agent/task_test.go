package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/tools"
)

func newTaskTool(t *testing.T, provider llm.Provider, extra ...tools.Tool) TaskTool {
	t.Helper()
	general := tools.NewRegistry(extra...)
	explore := tools.NewRegistry() // no tools needed for these tests
	return TaskTool{
		Provider:     provider,
		System:       "base system prompt",
		ExploreTools: explore,
		GeneralTools: general,
	}
}

func TestTaskToolReturnsSubagentsFinalAnswer(t *testing.T) {
	fake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "the subagent's answer"}}}
	tool := newTaskTool(t, fake)

	raw, _ := json.Marshal(map[string]string{
		"description": "look something up", "prompt": "find X", "agent_type": "explore",
	})
	out, err := tool.Run(context.Background(), raw)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "the subagent's answer" {
		t.Errorf("out = %q", out)
	}
}

// The entire point of a subagent: its internal back-and-forth (tool calls,
// intermediate assistant text) must never reach the parent's history — only
// the final text does, and that happens naturally because TaskTool.Run
// returns just res.Text, never res.Messages.
func TestTaskToolIsolatesSubagentHistoryFromParent(t *testing.T) {
	echo := &echoTool{}
	fake := &llmtest.Fake{Steps: []llmtest.Step{
		{Text: "investigating...", ToolCalls: []llm.ToolCall{
			llmtest.Call("c1", "echo", map[string]string{"text": "secret internal step"}),
		}},
		{Text: "final summary only"},
	}}
	taskTool := newTaskTool(t, fake, echo)

	parentFake := &llmtest.Fake{Steps: []llmtest.Step{
		{ToolCalls: []llm.ToolCall{llmtest.Call("t1", "task", map[string]string{
			"description": "investigate", "prompt": "go look", "agent_type": "general",
		})}},
		{Text: "done"},
	}}
	parent := &Agent{Provider: parentFake, Tools: tools.NewRegistry(taskTool)}

	res, err := parent.Run(context.Background(), "delegate this", nil, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}

	for _, m := range res.Messages {
		if strings.Contains(m.Content, "secret internal step") {
			t.Errorf("the subagent's internal tool call leaked into parent history: %+v", m)
		}
		if strings.Contains(m.Content, "investigating...") {
			t.Errorf("the subagent's intermediate assistant text leaked into parent history: %+v", m)
		}
	}
	// Only the subagent's final text should appear, as the task tool's own
	// result message.
	found := false
	for _, m := range res.Messages {
		if m.Role == llm.RoleTool && m.Content == "final summary only" {
			found = true
		}
	}
	if !found {
		t.Error("the subagent's final summary was not passed back as the tool's result")
	}
}

// No recursion: a subagent must not receive the task tool itself, regardless
// of agent_type — this is what rules out a subagent delegating to another
// subagent and forking without bound.
func TestSubagentHasNoTaskTool(t *testing.T) {
	for _, agentType := range []string{"explore", "general"} {
		var seenTools []string
		fake := &llmtest.Fake{Steps: []llmtest.Step{{Hook: func() {}, Text: "ok"}}}
		taskTool := newTaskTool(t, fake, &echoTool{})

		raw, _ := json.Marshal(map[string]string{
			"description": "x", "prompt": "x", "agent_type": agentType,
		})
		if _, err := taskTool.Run(context.Background(), raw); err != nil {
			t.Fatalf("[%s] Run: %v", agentType, err)
		}

		toolset := taskTool.ExploreTools
		if agentType == "general" {
			toolset = taskTool.GeneralTools
		}
		for _, s := range toolset.Schemas() {
			seenTools = append(seenTools, s.Name)
		}
		for _, name := range seenTools {
			if name == TaskName {
				t.Errorf("[%s] subagent toolset includes %q; recursion must be impossible", agentType, TaskName)
			}
		}
	}
}

func TestExploreAgentTypeUsesExploreTools(t *testing.T) {
	fake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "ok"}}}
	tool := newTaskTool(t, fake)
	tool.ExploreTools = tools.NewRegistry(&echoTool{}) // distinct marker tool
	tool.GeneralTools = tools.NewRegistry()            // empty — proves explore didn't fall through to this

	raw, _ := json.Marshal(map[string]string{
		"description": "x", "prompt": "x", "agent_type": "explore",
	})
	if _, err := tool.Run(context.Background(), raw); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(fake.Requests) != 1 {
		t.Fatalf("provider called %d times", len(fake.Requests))
	}
	if len(fake.Requests[0].Tools) != 1 || fake.Requests[0].Tools[0].Name != "echo" {
		t.Errorf("subagent did not receive ExploreTools: %+v", fake.Requests[0].Tools)
	}
}

func TestUnknownAgentTypeIsRejected(t *testing.T) {
	tool := newTaskTool(t, &llmtest.Fake{})
	raw, _ := json.Marshal(map[string]string{
		"description": "x", "prompt": "x", "agent_type": "sudo",
	})
	if _, err := tool.Run(context.Background(), raw); err == nil {
		t.Error("expected an error for an unknown agent_type")
	}
}

func TestTaskToolRequiresAPrompt(t *testing.T) {
	tool := newTaskTool(t, &llmtest.Fake{})
	raw, _ := json.Marshal(map[string]string{"description": "x", "agent_type": "explore"})
	if _, err := tool.Run(context.Background(), raw); err == nil {
		t.Error("expected an error for a missing prompt")
	}
}

// Multiple task calls in one batch must actually overlap in time, not just
// "not deadlock" — proven by two subagents that each block until both have
// started, which only resolves if they are running concurrently.
func TestParallelTaskCallsActuallyOverlap(t *testing.T) {
	var wg sync.WaitGroup
	wg.Add(2)
	barrier := func() {
		wg.Done()
		wg.Wait() // blocks forever if the two calls run one at a time
	}

	fakeA := &llmtest.Fake{Steps: []llmtest.Step{{Hook: barrier, Text: "a done"}}}
	fakeB := &llmtest.Fake{Steps: []llmtest.Step{{Hook: barrier, Text: "b done"}}}

	toolA := TaskTool{Provider: fakeA, GeneralTools: tools.NewRegistry(), ExploreTools: tools.NewRegistry()}
	toolB := TaskTool{Provider: fakeB, GeneralTools: tools.NewRegistry(), ExploreTools: tools.NewRegistry()}

	// The parent dispatches by tool name, so use a small registry that
	// routes "task" to a router picking A or B by call ID — simplest is two
	// distinctly named wrapper tools instead, since agent.Run only special-
	// cases the literal name "task" for parallel dispatch. Wrap both under
	// that name via a tiny adapter.
	parentFake := &llmtest.Fake{Steps: []llmtest.Step{
		{ToolCalls: []llm.ToolCall{
			llmtest.Call("c1", "task", map[string]string{"description": "a", "prompt": "a", "agent_type": "general"}),
			llmtest.Call("c2", "task", map[string]string{"description": "b", "prompt": "b", "agent_type": "general"}),
		}},
		{Text: "both done"},
	}}
	parent := &Agent{Provider: parentFake, Tools: tools.NewRegistry(&dispatchTask{a: toolA, b: toolB})}

	done := make(chan error, 1)
	go func() {
		_, err := parent.Run(context.Background(), "go", nil, nil)
		done <- err
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("the two task calls did not run concurrently — the barrier never released")
	}
}

// dispatchTask fans a single "task" tool call out to two underlying
// TaskTools by call id, so TestParallelTaskCallsActuallyOverlap can prove two
// *different* provider fakes actually run at once.
type dispatchTask struct {
	a, b TaskTool
	n    atomic.Int32
}

func (*dispatchTask) Name() string           { return TaskName }
func (*dispatchTask) Description() string    { return "test dispatcher" }
func (*dispatchTask) Schema() map[string]any { return map[string]any{"type": "object"} }
func (d *dispatchTask) Run(ctx context.Context, input json.RawMessage) (string, error) {
	if d.n.Add(1) == 1 {
		return d.a.Run(ctx, input)
	}
	return d.b.Run(ctx, input)
}
