package llm

import "testing"

func TestContextWindowMatchesKnownFamilies(t *testing.T) {
	cases := map[string]int{
		"claude-sonnet-5":         200_000,
		"claude-opus-5":           200_000,
		"gpt-5.5":                 400_000,
		"gpt-4o-mini":             128_000,
		"gemini-2.5-pro":          1_000_000,
		"qwen3-coder":             256_000,
		"llama-3.3-70b-versatile": 128_000,
	}
	for model, want := range cases {
		if got := ContextWindow(model); got != want {
			t.Errorf("ContextWindow(%q) = %d, want %d", model, got, want)
		}
	}
}

// Profiles carry provider-prefixed names ("openai/gpt-5.5" on OpenRouter) and
// arbitrary casing, so matching has to survive both.
func TestContextWindowIgnoresPrefixesAndCase(t *testing.T) {
	if got := ContextWindow("OpenAI/GPT-5.5"); got != 400_000 {
		t.Errorf("ContextWindow with a prefix and mixed case = %d, want 400000", got)
	}
}

func TestContextWindowFallsBackForUnknownModels(t *testing.T) {
	if got := ContextWindow("something-nobody-has-heard-of"); got != DefaultContextWindow {
		t.Errorf("ContextWindow(unknown) = %d, want the %d default", got, DefaultContextWindow)
	}
	if got := ContextWindow(""); got != DefaultContextWindow {
		t.Errorf("ContextWindow(\"\") = %d, want the %d default", got, DefaultContextWindow)
	}
}

// A turn that read three files is mostly tool traffic; counting only prose
// would under-count exactly the histories most in need of compacting.
func TestEstimateMessageTokensCountsToolCallArguments(t *testing.T) {
	prose := []Message{{Role: RoleAssistant, Content: "ok"}}
	withCall := []Message{{
		Role:      RoleAssistant,
		Content:   "ok",
		ToolCalls: []ToolCall{{ID: "1", Name: "read_file", Input: []byte(`{"path":"a/very/long/path/to/some/file.go"}`)}},
	}}

	if EstimateMessageTokens(withCall) <= EstimateMessageTokens(prose) {
		t.Error("a message carrying a tool call was estimated no larger than one without")
	}
}

func TestEstimateTokensRoundsUp(t *testing.T) {
	if got := EstimateTokens(""); got != 0 {
		t.Errorf("EstimateTokens(\"\") = %d, want 0", got)
	}
	if got := EstimateTokens("ab"); got != 1 {
		t.Errorf("EstimateTokens(%q) = %d, want 1 — a partial token still costs one", "ab", got)
	}
}
