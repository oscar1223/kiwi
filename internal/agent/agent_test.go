package agent

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/tools"
)

// echoTool returns whatever it is given, recording invocations.
type echoTool struct {
	calls atomic.Int32
	block chan struct{} // if non-nil, Run waits on it or ctx
}

func (e *echoTool) Name() string        { return "echo" }
func (e *echoTool) Description() string { return "Echo the input back." }
func (e *echoTool) Schema() map[string]any {
	return map[string]any{
		"type":       "object",
		"properties": map[string]any{"text": map[string]any{"type": "string"}},
		"required":   []string{"text"},
	}
}

func (e *echoTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	e.calls.Add(1)
	if e.block != nil {
		select {
		case <-e.block:
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	var in struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	return in.Text, nil
}

type failingTool struct{}

func (failingTool) Name() string           { return "boom" }
func (failingTool) Description() string    { return "Always fails." }
func (failingTool) Schema() map[string]any { return map[string]any{"type": "object"} }
func (failingTool) Run(context.Context, json.RawMessage) (string, error) {
	return "", errors.New("detonated")
}

type recorder struct {
	text     strings.Builder
	calls    []llm.ToolCall
	results  []string
	errFlags []bool
}

func (r *recorder) OnText(d string)           { r.text.WriteString(d) }
func (r *recorder) OnToolCall(c llm.ToolCall) { r.calls = append(r.calls, c) }
func (r *recorder) OnUsage(llm.Usage)         {}
func (r *recorder) OnToolResult(_ llm.ToolCall, out string, isErr bool) {
	r.results = append(r.results, out)
	r.errFlags = append(r.errFlags, isErr)
}

func TestRunPlainAnswer(t *testing.T) {
	fake := &llmtest.Fake{Steps: []llmtest.Step{
		{Chunks: []string{"Hola", ", ", "mundo"}},
	}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry()}
	rec := &recorder{}

	res, err := a.Run(context.Background(), "saluda", nil, rec)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if res.Text != "Hola, mundo" {
		t.Errorf("Text = %q, want %q", res.Text, "Hola, mundo")
	}
	if rec.text.String() != "Hola, mundo" {
		t.Errorf("streamed = %q, want deltas to assemble to the same text", rec.text.String())
	}
	if res.Steps != 1 {
		t.Errorf("Steps = %d, want 1", res.Steps)
	}
	// One user + one assistant message.
	if len(res.Messages) != 2 {
		t.Fatalf("len(Messages) = %d, want 2", len(res.Messages))
	}
	if res.Usage.OutputTokens != 5 {
		t.Errorf("Usage.OutputTokens = %d, want 5", res.Usage.OutputTokens)
	}
}

func TestRunToolCallRoundTrip(t *testing.T) {
	echo := &echoTool{}
	fake := &llmtest.Fake{Steps: []llmtest.Step{
		{Text: "voy a llamar", ToolCalls: []llm.ToolCall{
			llmtest.Call("c1", "echo", map[string]string{"text": "ping"}),
		}},
		{Text: "listo: ping"},
	}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry(echo)}
	rec := &recorder{}

	res, err := a.Run(context.Background(), "haz eco de ping", nil, rec)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if got := echo.calls.Load(); got != 1 {
		t.Errorf("tool invoked %d times, want 1", got)
	}
	if res.Text != "listo: ping" {
		t.Errorf("Text = %q", res.Text)
	}
	if res.Steps != 2 {
		t.Errorf("Steps = %d, want 2", res.Steps)
	}
	// user, assistant+call, tool result, assistant
	if len(res.Messages) != 4 {
		t.Fatalf("len(Messages) = %d, want 4", len(res.Messages))
	}
	if res.Messages[2].Role != llm.RoleTool || res.Messages[2].Content != "ping" {
		t.Errorf("tool message = %+v", res.Messages[2])
	}
	if len(rec.results) != 1 || rec.results[0] != "ping" {
		t.Errorf("observer results = %v", rec.results)
	}

	// The second request must carry the full transcript so far.
	if len(fake.Requests) != 2 {
		t.Fatalf("provider called %d times", len(fake.Requests))
	}
	if n := len(fake.Requests[1].Messages); n != 3 {
		t.Errorf("second request had %d messages, want 3", n)
	}
}

// A failing tool must not abort the turn: the model gets an error observation
// and a chance to recover.
func TestToolErrorBecomesObservation(t *testing.T) {
	fake := &llmtest.Fake{Steps: []llmtest.Step{
		{ToolCalls: []llm.ToolCall{llmtest.Call("c1", "boom", map[string]string{})}},
		{Text: "me recupero"},
	}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry(failingTool{})}
	rec := &recorder{}

	res, err := a.Run(context.Background(), "explota", nil, rec)
	if err != nil {
		t.Fatalf("Run should not fail on a tool error, got %v", err)
	}
	if res.Text != "me recupero" {
		t.Errorf("Text = %q", res.Text)
	}
	if len(rec.errFlags) != 1 || !rec.errFlags[0] {
		t.Errorf("expected the observation to be flagged as an error, got %v", rec.errFlags)
	}
	if !strings.Contains(res.Messages[2].Content, "detonated") {
		t.Errorf("error text not passed to the model: %q", res.Messages[2].Content)
	}
}

// An unknown tool is recoverable too — models hallucinate tool names.
func TestUnknownToolIsRecoverable(t *testing.T) {
	fake := &llmtest.Fake{Steps: []llmtest.Step{
		{ToolCalls: []llm.ToolCall{llmtest.Call("c1", "nope", map[string]string{})}},
		{Text: "vale"},
	}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry()}

	res, err := a.Run(context.Background(), "x", nil, nil)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(res.Messages[2].Content, "unknown tool") {
		t.Errorf("observation = %q", res.Messages[2].Content)
	}
}

// Cancelling the turn must surface as an error, not a truncated answer.
func TestCancelDuringToolStopsTheTurn(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	echo := &echoTool{block: make(chan struct{})} // never released
	fake := &llmtest.Fake{Steps: []llmtest.Step{
		{ToolCalls: []llm.ToolCall{llmtest.Call("c1", "echo", map[string]string{"text": "x"})}},
		{Text: "no debería llegar aquí"},
	}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry(echo)}

	go func() {
		for echo.calls.Load() == 0 {
		}
		cancel()
	}()

	_, err := a.Run(ctx, "x", nil, nil)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled", err)
	}
	if fake.Calls() != 1 {
		t.Errorf("provider called %d times after cancel, want 1", fake.Calls())
	}
}

func TestHistoryIsNotMutated(t *testing.T) {
	history := []llm.Message{{Role: llm.RoleUser, Content: "previo"}}
	fake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "ok"}}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry()}

	if _, err := a.Run(context.Background(), "nuevo", history, nil); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(history) != 1 || history[0].Content != "previo" {
		t.Errorf("history was mutated: %+v", history)
	}
	// But the request must have included it.
	if n := len(fake.Requests[0].Messages); n != 2 {
		t.Errorf("request had %d messages, want 2 (history + input)", n)
	}
}

func TestMaxStepsGuard(t *testing.T) {
	steps := make([]llmtest.Step, 10)
	for i := range steps {
		steps[i] = llmtest.Step{ToolCalls: []llm.ToolCall{
			llmtest.Call("c", "echo", map[string]string{"text": "loop"}),
		}}
	}
	fake := &llmtest.Fake{Steps: steps}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry(&echoTool{}), MaxSteps: 3}

	_, err := a.Run(context.Background(), "loop", nil, nil)
	if !errors.Is(err, ErrMaxSteps) {
		t.Fatalf("err = %v, want ErrMaxSteps", err)
	}
	if fake.Calls() != 3 {
		t.Errorf("provider called %d times, want 3", fake.Calls())
	}
}

func TestSchemasAreStablyOrdered(t *testing.T) {
	r := tools.NewRegistry(&echoTool{}, failingTool{})
	first := r.Schemas()
	for range 20 {
		got := r.Schemas()
		for i := range got {
			if got[i].Name != first[i].Name {
				t.Fatalf("tool order is not stable: %v vs %v", got, first)
			}
		}
	}
	if first[0].Name != "boom" || first[1].Name != "echo" {
		t.Errorf("want alphabetical order, got %s,%s", first[0].Name, first[1].Name)
	}
}

func TestRegistrySubset(t *testing.T) {
	r := tools.NewRegistry(&echoTool{}, failingTool{})
	sub := r.Subset("echo", "does-not-exist")
	if _, ok := sub.Get("echo"); !ok {
		t.Error("echo missing from subset")
	}
	if _, ok := sub.Get("boom"); ok {
		t.Error("boom should not be in the subset")
	}
	if n := len(sub.Schemas()); n != 1 {
		t.Errorf("subset has %d tools, want 1", n)
	}
}
