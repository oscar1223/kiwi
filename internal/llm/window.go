package llm

import "strings"

// DefaultContextWindow is what an unrecognised model is assumed to have. It is
// deliberately on the low side: under-estimating a window makes Kiwi compact
// earlier than it had to, while over-estimating it makes a request bounce off
// the provider with a context-length error the user cannot act on.
const DefaultContextWindow = 128_000

// contextWindows maps a model-name fragment to its context window in tokens,
// longest fragment first so "gpt-4.1" is matched before "gpt-4".
//
// This is a heuristic, not a specification. It exists only to size a
// compaction threshold, so being a little wrong costs one summarization more
// or less — nothing depends on it being exact, and no provider is asked for
// it, which keeps this a pure function with no network in the path.
var contextWindows = []struct {
	fragment string
	window   int
}{
	{"claude-3-haiku", 200_000},
	{"claude", 200_000},
	{"gpt-4.1", 1_000_000},
	{"gpt-4o", 128_000},
	{"gpt-4", 128_000},
	{"gpt-5", 400_000},
	{"o1", 200_000},
	{"o3", 200_000},
	{"o4", 200_000},
	{"gemini", 1_000_000},
	{"grok", 256_000},
	{"qwen", 256_000},
	{"kimi", 256_000},
	{"moonshot", 256_000},
	{"deepseek", 128_000},
	{"llama", 128_000},
	{"mistral", 128_000},
	{"mixtral", 32_000},
	{"glm", 128_000},
}

// ContextWindow returns the token window Kiwi should assume for a model.
func ContextWindow(model string) int {
	m := strings.ToLower(model)
	for _, e := range contextWindows {
		if strings.Contains(m, e.fragment) {
			return e.window
		}
	}
	return DefaultContextWindow
}

// charsPerToken is the rough ratio used to size history without pulling in a
// tokenizer per provider. Real tokenizers disagree with each other anyway, and
// four characters per token is close enough for a threshold — code and JSON
// tend to run denser, which errs toward compacting early.
const charsPerToken = 4

// EstimateTokens approximates how many tokens a run of text costs.
func EstimateTokens(s string) int {
	return (len(s) + charsPerToken - 1) / charsPerToken
}

// EstimateMessageTokens approximates the cost of a conversation, counting the
// JSON arguments of tool calls as well as prose: a turn that read three files
// is mostly tool traffic, and ignoring it would badly under-count exactly the
// histories most in need of compacting.
func EstimateMessageTokens(msgs []Message) int {
	total := 0
	for _, m := range msgs {
		total += EstimateTokens(m.Content)
		for _, tc := range m.ToolCalls {
			total += EstimateTokens(tc.Name) + EstimateTokens(string(tc.Input))
		}
	}
	return total
}
