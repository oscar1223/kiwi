package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
)

// thinkingStream is a Messages stream that thinks, then answers.
var thinkingStream = []string{
	`event: message_start
data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","model":"claude","content":[],"stop_reason":null,"usage":{"input_tokens":5,"output_tokens":1}}}`,
	`event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`,
	`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Weighing it."}}`,
	`event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIG"}}`,
	`event: content_block_stop
data: {"type":"content_block_stop","index":0}`,
	`event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}`,
	`event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Done."}}`,
	`event: content_block_stop
data: {"type":"content_block_stop","index":1}`,
	`event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}`,
	`event: message_stop
data: {"type":"message_stop"}`,
}

func server(t *testing.T) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		json.Unmarshal(raw, &body)
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range thinkingStream {
			fmt.Fprintf(w, "%s\n\n", e)
		}
	}))
	t.Cleanup(srv.Close)
	return srv, &bodies
}

func run(t *testing.T, p *Provider, history []llm.Message) (reasoning string, msg *llm.Message) {
	t.Helper()
	for ev, err := range p.Stream(context.Background(), llm.Request{Messages: history}) {
		if err != nil {
			t.Fatal(err)
		}
		switch ev.Type {
		case llm.EventReasoningDelta:
			reasoning += ev.Text
		case llm.EventDone:
			msg = ev.Message
		}
	}
	return
}

func js(v any) string { b, _ := json.Marshal(v); return string(b) }

// Each model family gets the shape of request it accepts.
func TestThinkingConfigPerModel(t *testing.T) {
	cases := []struct {
		model    string
		level    llm.Reasoning
		thinking string
		effort   string
	}{
		{"claude-sonnet-5", llm.ReasoningXHigh, `{"display":"summarized","type":"adaptive"}`, "xhigh"},
		{"claude-sonnet-4-6", llm.ReasoningXHigh, `{"display":"summarized","type":"adaptive"}`, "high"},
		{"claude-opus-5", llm.ReasoningOff, `{"type":"disabled"}`, ""},
		{"claude-opus-5-5", llm.ReasoningOff, "", "low"},
		{"claude-haiku-4-5", llm.ReasoningMedium, `{"budget_tokens":4096,"type":"enabled"}`, ""},
		{"claude-sonnet-5", llm.ReasoningDefault, "", ""},
	}
	for _, c := range cases {
		srv, bodies := server(t)
		p := New(Options{APIKey: "k", BaseURL: srv.URL, Model: c.model, Reasoning: c.level})
		run(t, p, []llm.Message{{Role: llm.RoleUser, Content: "hi"}})
		body := (*bodies)[0]
		thinking := ""
		if v, ok := body["thinking"]; ok {
			thinking = js(v)
		}
		effort := ""
		if oc, ok := body["output_config"].(map[string]any); ok {
			effort, _ = oc["effort"].(string)
		}
		if thinking != c.thinking || effort != c.effort {
			t.Errorf("%s/%s: thinking %s effort %q, want %s %q", c.model, c.level, thinking, effort, c.thinking, c.effort)
		}
		if c.level == llm.ReasoningMedium && body["max_tokens"].(float64) <= 4096 {
			t.Errorf("%s: max_tokens %v does not leave room past the budget", c.model, body["max_tokens"])
		}
	}
}

// Thinking streams as reasoning, and its signed block goes back ahead of the
// answer on the next request, as the API requires with tools in play.
func TestThinkingIsStreamedAndReplayed(t *testing.T) {
	srv, bodies := server(t)
	p := New(Options{APIKey: "k", BaseURL: srv.URL, Model: "claude-sonnet-5", Reasoning: llm.ReasoningHigh})
	history := []llm.Message{{Role: llm.RoleUser, Content: "hi"}}
	reasoning, msg := run(t, p, history)
	if reasoning != "Weighing it." || msg.Content != "Done." || msg.Reasoning == nil {
		t.Fatalf("reasoning %q content %q trace %v", reasoning, msg.Content, msg.Reasoning)
	}

	run(t, p, append(history, *msg, llm.Message{Role: llm.RoleUser, Content: "more"}))
	assistant := (*bodies)[1]["messages"].([]any)[1].(map[string]any)
	first := assistant["content"].([]any)[0].(map[string]any)
	if first["type"] != "thinking" || first["signature"] != "SIG" || first["thinking"] != "Weighing it." {
		t.Errorf("thinking block not replayed first: %s", js(assistant))
	}

	// A different model does not get it.
	other := New(Options{APIKey: "k", BaseURL: srv.URL, Model: "claude-opus-5", Reasoning: llm.ReasoningHigh})
	run(t, other, append(history, *msg, llm.Message{Role: llm.RoleUser, Content: "more"}))
	first = (*bodies)[2]["messages"].([]any)[1].(map[string]any)["content"].([]any)[0].(map[string]any)
	if first["type"] == "thinking" {
		t.Error("thinking was replayed to a different model")
	}
}
