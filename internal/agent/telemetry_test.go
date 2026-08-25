package agent

import (
	"context"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/tools"
)

func withSpanRecorder(t *testing.T) *tracetest.InMemoryExporter {
	t.Helper()
	exp := tracetest.NewInMemoryExporter()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp))
	otel.SetTracerProvider(tp)
	t.Cleanup(func() { tp.Shutdown(context.Background()) })
	return exp
}

// One turn with one tool call must produce a turn span with a model-call
// child and a tool child nested under it, all on the same trace — this is
// what makes a turn render as one connected trace rather than orphan spans.
func TestRunProducesNestedTelemetrySpans(t *testing.T) {
	exp := withSpanRecorder(t)

	echo := &echoTool{}
	fake := &llmtest.Fake{Steps: []llmtest.Step{
		{ToolCalls: []llm.ToolCall{llmtest.Call("c1", "echo", map[string]string{"text": "hi"})}},
		{Text: "done"},
	}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry(echo)}

	if _, err := a.Run(context.Background(), "go", nil, nil); err != nil {
		t.Fatalf("Run: %v", err)
	}

	spans := exp.GetSpans()
	byName := map[string]tracetest.SpanStub{}
	for _, s := range spans {
		byName[s.Name] = s
	}

	turn, ok := byName["kiwi.turn"]
	if !ok {
		t.Fatalf("no kiwi.turn span among %d spans: %+v", len(spans), spans)
	}
	modelCalls := 0
	for _, s := range spans {
		if s.Name == "chat fake-1" {
			modelCalls++
			if s.Parent.SpanID() != turn.SpanContext.SpanID() {
				t.Error("model-call span is not a child of the turn span")
			}
		}
	}
	if modelCalls != 2 {
		t.Errorf("got %d model-call spans, want 2 (one per provider round trip)", modelCalls)
	}

	tool, ok := byName["execute_tool echo"]
	if !ok {
		t.Fatalf("no execute_tool echo span among: %+v", spans)
	}
	if tool.Parent.SpanID() != turn.SpanContext.SpanID() {
		t.Error("tool span is not a child of the turn span")
	}
	if tool.SpanContext.TraceID() != turn.SpanContext.TraceID() {
		t.Error("tool span is on a different trace than its turn")
	}
}

// A turn that errors must still close its span, marked as failed — a span
// that never ends would leak memory in the real exporter's batching buffer,
// and Langfuse would show the trace as perpetually "in progress".
func TestRunRecordsErrorOnTurnSpan(t *testing.T) {
	exp := withSpanRecorder(t)

	fake := &llmtest.Fake{Steps: []llmtest.Step{{Err: errBoom}}}
	a := &Agent{Provider: fake, Tools: tools.NewRegistry()}

	if _, err := a.Run(context.Background(), "go", nil, nil); err == nil {
		t.Fatal("expected an error")
	}

	// One span for the failed model call, one for the turn it failed inside —
	// both must end (nothing left open) and both must carry the error.
	spans := exp.GetSpans()
	if len(spans) != 2 {
		t.Fatalf("got %d spans, want 2 (model call + turn): %+v", len(spans), spans)
	}
	for _, s := range spans {
		if s.Status.Code != codes.Error {
			t.Errorf("span %q status = %v, want Error", s.Name, s.Status.Code)
		}
	}
}

var errBoom = boomErr2{}

type boomErr2 struct{}

func (boomErr2) Error() string { return "provider exploded" }
