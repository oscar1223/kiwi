package anthropic

import (
	"encoding/json"
	"strings"

	sdk "github.com/anthropics/anthropic-sdk-go"
	"github.com/oscar1223/kiwi/internal/llm"
)

// Reasoning on Claude.
//
// Current models think adaptively and take the level as output_config.effort.
// Older ones (Haiku 4.5, and the 4.5-and-earlier Opus and Sonnet) only know a
// fixed thinking budget. A few recent models cannot stop thinking at all, so
// "off" is the lowest effort there. The model is recognised by its ID, which
// on Bedrock and Vertex carries a prefix or a date but the same name.

// thinkingMaxTokens replaces the default response cap while the model is
// thinking, since the thinking comes out of the same budget.
const thinkingMaxTokens = 32000

// budgets are the thinking budgets for models that predate effort.
var budgets = map[llm.Reasoning]int64{
	llm.ReasoningLow:    2048,
	llm.ReasoningMedium: 4096,
	llm.ReasoningHigh:   8192,
	llm.ReasoningXHigh:  16384,
	llm.ReasoningMax:    24576,
}

func hasAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if strings.Contains(s, sub) {
			return true
		}
	}
	return false
}

// budgetOnly: models from before adaptive thinking and effort.
func budgetOnly(model string) bool {
	m := strings.ToLower(model)
	return hasAny(m, "claude-3", "haiku-4-5", "sonnet-4-5", "opus-4-5", "opus-4-1",
		"sonnet-4-2", "opus-4-2", "sonnet-4@", "opus-4@")
}

// alwaysThinks: models that reject any attempt to turn thinking off.
func alwaysThinks(model string) bool {
	return hasAny(strings.ToLower(model), "fable", "mythos", "opus-5-5")
}

// thinks reports whether this provider asks for thinking at all.
func (p *Provider) thinks() bool {
	return p.reasoning != llm.ReasoningDefault && p.reasoning != llm.ReasoningOff
}

func (p *Provider) applyReasoning(params *sdk.MessageNewParams) {
	r := p.reasoning
	switch {
	case r == llm.ReasoningDefault:
		return

	case r == llm.ReasoningOff && alwaysThinks(p.model):
		// Thinking cannot be switched off here; the least of it is the
		// closest there is.
		params.OutputConfig.Effort = sdk.OutputConfigEffortLow

	case r == llm.ReasoningOff:
		params.Thinking = sdk.ThinkingConfigParamUnion{OfDisabled: &sdk.ThinkingConfigDisabledParam{}}

	case budgetOnly(p.model):
		budget := budgets[r]
		params.Thinking = sdk.ThinkingConfigParamOfEnabled(budget)
		if params.MaxTokens <= budget {
			params.MaxTokens = budget + defaultMaxTokens
		}

	default:
		// Summarized, so there is something to show while it thinks: the
		// default on recent models is an empty thinking block.
		adaptive := sdk.ThinkingConfigAdaptiveParam{Display: sdk.ThinkingConfigAdaptiveDisplaySummarized}
		params.Thinking = sdk.ThinkingConfigParamUnion{OfAdaptive: &adaptive}
		effort := sdk.OutputConfigEffort(r)
		if r == llm.ReasoningXHigh && strings.Contains(strings.ToLower(p.model), "4-6") {
			effort = sdk.OutputConfigEffortHigh // xhigh arrived with 4.7
		}
		params.OutputConfig.Effort = effort
	}
}

// thinkingBlock is one thinking or redacted_thinking block, kept verbatim so
// it can be sent back unchanged.
type thinkingBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

func (p *Provider) source() string { return "anthropic:" + p.name + "/" + p.model }

func (p *Provider) trace(blocks []thinkingBlock) *llm.ReasoningTrace {
	if len(blocks) == 0 {
		return nil
	}
	raw, err := json.Marshal(blocks)
	if err != nil {
		return nil
	}
	return &llm.ReasoningTrace{Source: p.source(), Raw: raw}
}

// replayThinking turns a trace back into the blocks it came from. A trace
// from another provider or model is dropped: thinking is bound to the model
// that produced it.
func replayThinking(t *llm.ReasoningTrace, source string) []sdk.ContentBlockParamUnion {
	if t == nil || t.Source != source || len(t.Raw) == 0 {
		return nil
	}
	var blocks []thinkingBlock
	if json.Unmarshal(t.Raw, &blocks) != nil {
		return nil
	}
	out := make([]sdk.ContentBlockParamUnion, 0, len(blocks))
	for _, b := range blocks {
		switch b.Type {
		case "thinking":
			out = append(out, sdk.NewThinkingBlock(b.Signature, b.Thinking))
		case "redacted_thinking":
			out = append(out, sdk.NewRedactedThinkingBlock(b.Data))
		}
	}
	return out
}
