package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
)

// server answers every request with the given SSE chunks and records the
// request bodies it was sent.
func server(t *testing.T, chunks ...string) (*httptest.Server, *[]map[string]any) {
	t.Helper()
	var bodies []map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		if err := json.Unmarshal(raw, &body); err != nil {
			t.Errorf("request body: %v", err)
		}
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "text/event-stream")
		for _, c := range chunks {
			fmt.Fprintf(w, "data: %s\n\n", c)
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(srv.Close)
	return srv, &bodies
}

func chunk(delta string, finish string) string {
	fin := "null"
	if finish != "" {
		fin = `"` + finish + `"`
	}
	return `{"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[{"index":0,"delta":` + delta + `,"finish_reason":` + fin + `}]}`
}

func run(t *testing.T, p *Provider, history []llm.Message) (reasoning, text string, msg *llm.Message) {
	t.Helper()
	for ev, err := range p.Stream(context.Background(), llm.Request{Messages: history}) {
		if err != nil {
			t.Fatal(err)
		}
		switch ev.Type {
		case llm.EventReasoningDelta:
			reasoning += ev.Text
		case llm.EventTextDelta:
			text += ev.Text
		case llm.EventDone:
			msg = ev.Message
		}
	}
	return
}

var openRouterChunks = []string{
	chunk(`{"role":"assistant","reasoning":"Let me ","reasoning_details":[{"type":"reasoning.text","text":"Let me ","index":0,"format":"f"}]}`, ""),
	chunk(`{"reasoning":"think.","reasoning_details":[{"type":"reasoning.text","text":"think.","index":0,"signature":"sig"}]}`, ""),
	chunk(`{"content":"Answer."}`, ""),
	chunk(`{}`, "stop"),
}

func TestOpenRouterSendsTheLevel(t *testing.T) {
	for level, want := range map[llm.Reasoning]string{
		llm.ReasoningHigh: `{"effort":"high"}`,
		llm.ReasoningOff:  `{"enabled":false}`,
	} {
		srv, bodies := server(t, openRouterChunks...)
		p := New(Options{BaseURL: srv.URL + "/openrouter.ai/api/v1", APIKey: "k", Model: "z-ai/glm-5.2", Reasoning: level})
		p.flavor = flavorOpenRouter // the test server's URL cannot say openrouter.ai on its own
		run(t, p, []llm.Message{{Role: llm.RoleUser, Content: "hi"}})
		got, _ := json.Marshal((*bodies)[0]["reasoning"])
		if string(got) != want {
			t.Errorf("%s: reasoning = %s, want %s", level, got, want)
		}
	}
}

func TestDefaultSendsNothing(t *testing.T) {
	srv, bodies := server(t, openRouterChunks...)
	p := New(Options{BaseURL: srv.URL, APIKey: "k", Model: "m"})
	p.flavor = flavorOpenRouter
	run(t, p, []llm.Message{{Role: llm.RoleUser, Content: "hi"}})
	for _, k := range []string{"reasoning", "reasoning_effort", "thinking"} {
		if _, ok := (*bodies)[0][k]; ok {
			t.Errorf("default level sent %q", k)
		}
	}
}

// Reasoning streams as its own events, and OpenRouter's reasoning_details are
// stitched back together and returned on the next request.
func TestOpenRouterStreamsAndReplaysReasoning(t *testing.T) {
	srv, bodies := server(t, openRouterChunks...)
	p := New(Options{BaseURL: srv.URL, APIKey: "k", Model: "m", Reasoning: llm.ReasoningHigh})
	p.flavor = flavorOpenRouter

	history := []llm.Message{{Role: llm.RoleUser, Content: "hi"}}
	reasoning, text, msg := run(t, p, history)
	if reasoning != "Let me think." || text != "Answer." {
		t.Fatalf("reasoning %q, text %q", reasoning, text)
	}
	if msg.Reasoning == nil {
		t.Fatal("no reasoning trace kept")
	}
	var details []map[string]any
	json.Unmarshal(msg.Reasoning.Raw, &details)
	if len(details) != 1 || details[0]["text"] != "Let me think." || details[0]["signature"] != "sig" {
		t.Fatalf("details not stitched: %s", msg.Reasoning.Raw)
	}

	run(t, p, append(history, *msg, llm.Message{Role: llm.RoleUser, Content: "again"}))
	sent := (*bodies)[1]["messages"].([]any)[1].(map[string]any)
	if !strings.Contains(fmt.Sprint(sent["reasoning_details"]), "Let me think.") {
		t.Errorf("reasoning_details not replayed: %v", sent)
	}

	// Another model's trace is not handed over.
	other := New(Options{BaseURL: srv.URL, APIKey: "k", Model: "other", Reasoning: llm.ReasoningHigh})
	other.flavor = flavorOpenRouter
	run(t, other, append(history, *msg, llm.Message{Role: llm.RoleUser, Content: "again"}))
	if _, ok := (*bodies)[2]["messages"].([]any)[1].(map[string]any)["reasoning_details"]; ok {
		t.Error("a trace was replayed to a different model")
	}
}

func TestZAIUsesThinkingAndReasoningContent(t *testing.T) {
	srv, bodies := server(t,
		chunk(`{"role":"assistant","reasoning_content":"hmm"}`, ""),
		chunk(`{"content":"ok"}`, "stop"))
	p := New(Options{BaseURL: srv.URL, APIKey: "k", Model: "glm-5.2", Reasoning: llm.ReasoningOff})
	p.flavor = flavorZAI
	reasoning, _, msg := run(t, p, []llm.Message{{Role: llm.RoleUser, Content: "hi"}})
	got, _ := json.Marshal((*bodies)[0]["thinking"])
	if string(got) != `{"type":"disabled"}` {
		t.Errorf("thinking = %s", got)
	}
	if reasoning != "hmm" || msg.Reasoning == nil || msg.Reasoning.Text != "hmm" {
		t.Errorf("reasoning_content not read: %q %+v", reasoning, msg.Reasoning)
	}
}

func TestGenericUsesReasoningEffort(t *testing.T) {
	for level, want := range map[llm.Reasoning]string{llm.ReasoningLow: "low", llm.ReasoningOff: "none"} {
		srv, bodies := server(t, chunk(`{"role":"assistant","content":"ok"}`, "stop"))
		p := New(Options{BaseURL: srv.URL, APIKey: "k", Model: "gpt", Reasoning: level})
		run(t, p, []llm.Message{{Role: llm.RoleUser, Content: "hi"}})
		if got := (*bodies)[0]["reasoning_effort"]; got != want {
			t.Errorf("%s: reasoning_effort = %v, want %s", level, got, want)
		}
	}
}

func TestFlavorOf(t *testing.T) {
	for url, want := range map[string]flavor{
		"https://openrouter.ai/api/v1":         flavorOpenRouter,
		"https://open.bigmodel.cn/api/paas/v4": flavorZAI,
		"https://api.z.ai/api/paas/v4":         flavorZAI,
		"http://localhost:11434/v1":            flavorGeneric,
		"":                                     flavorGeneric,
	} {
		if got := flavorOf(url); got != want {
			t.Errorf("flavorOf(%q) = %d, want %d", url, got, want)
		}
	}
}
