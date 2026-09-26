package openai

import (
	"encoding/json"
	"sort"
	"strings"

	sdk "github.com/openai/openai-go/v3"
	"github.com/oscar1223/kiwi/internal/llm"
)

// Reasoning over chat-completions.
//
// There is no standard field for it, so the server is recognised by its base
// URL and spoken to in its own dialect:
//
//   - OpenRouter takes a reasoning object ({"effort": …} or {"enabled":
//     false}), streams the thinking as delta.reasoning, and asks for its
//     reasoning_details back on the next request so a model can carry its
//     thinking across tool calls.
//   - Z.ai (GLM directly) has an on/off switch, thinking.type, streams
//     reasoning_content, and reads it back the same way.
//   - Everything else gets OpenAI's reasoning_effort. Servers that do not
//     know it either ignore it or say so, and nothing is sent at all unless a
//     level was chosen.

type flavor int

const (
	flavorGeneric flavor = iota
	flavorOpenRouter
	flavorZAI
)

func flavorOf(baseURL string) flavor {
	u := strings.ToLower(baseURL)
	switch {
	case strings.Contains(u, "openrouter.ai"):
		return flavorOpenRouter
	case strings.Contains(u, "bigmodel.cn"), strings.Contains(u, "api.z.ai"):
		return flavorZAI
	default:
		return flavorGeneric
	}
}

// applyReasoning adds the level to a request, in the server's dialect.
func (p *Provider) applyReasoning(params *sdk.ChatCompletionNewParams) {
	r := p.reasoning
	if r == llm.ReasoningDefault {
		return
	}
	switch p.flavor {
	case flavorOpenRouter:
		if r == llm.ReasoningOff {
			params.SetExtraFields(map[string]any{"reasoning": map[string]any{"enabled": false}})
		} else {
			params.SetExtraFields(map[string]any{"reasoning": map[string]any{"effort": string(r)}})
		}
	case flavorZAI:
		// Z.ai thinks or does not; every level above off means "on".
		kind := "enabled"
		if r == llm.ReasoningOff {
			kind = "disabled"
		}
		params.SetExtraFields(map[string]any{"thinking": map[string]any{"type": kind}})
	default:
		if r == llm.ReasoningOff {
			params.ReasoningEffort = "none"
		} else {
			params.ReasoningEffort = sdk.ReasoningEffort(r)
		}
	}
}

// source identifies this provider and model on a ReasoningTrace.
func (p *Provider) source() string { return "openai:" + p.name + "/" + p.model }

// trace keeps what the server will want back next time, if anything.
func (p *Provider) trace(a *reasoningAccumulator) *llm.ReasoningTrace {
	switch p.flavor {
	case flavorOpenRouter:
		if raw := a.details(); raw != nil {
			return &llm.ReasoningTrace{Source: p.source(), Raw: raw}
		}
	case flavorZAI:
		if text := a.text.String(); text != "" {
			return &llm.ReasoningTrace{Source: p.source(), Text: text}
		}
	}
	return nil
}

// replay hands a trace back on the assistant message that produced it. A
// trace from another provider or model is dropped: it means nothing there.
func (p *Provider) replay(am *sdk.ChatCompletionAssistantMessageParam, t *llm.ReasoningTrace) {
	if t == nil || t.Source != p.source() {
		return
	}
	switch p.flavor {
	case flavorOpenRouter:
		if len(t.Raw) > 0 {
			am.SetExtraFields(map[string]any{"reasoning_details": t.Raw})
		}
	case flavorZAI:
		if t.Text != "" {
			am.SetExtraFields(map[string]any{"reasoning_content": t.Text})
		}
	}
}

// reasoningAccumulator gathers the reasoning a stream carries alongside the
// answer: its text, for display, and OpenRouter's reasoning_details, which
// arrive in pieces keyed by index and are stitched back together here.
type reasoningAccumulator struct {
	text    strings.Builder
	pieces  map[int]map[string]any
	ordered []int
}

// add reads one streamed delta and returns the reasoning text it carried.
func (a *reasoningAccumulator) add(raw string) string {
	if raw == "" || !strings.Contains(raw, "reason") {
		return ""
	}
	var d struct {
		Reasoning        *string          `json:"reasoning"`
		ReasoningContent *string          `json:"reasoning_content"`
		Details          []map[string]any `json:"reasoning_details"`
	}
	if json.Unmarshal([]byte(raw), &d) != nil {
		return ""
	}
	var text string
	switch {
	case d.Reasoning != nil:
		text = *d.Reasoning
	case d.ReasoningContent != nil:
		text = *d.ReasoningContent
	}
	a.text.WriteString(text)

	for _, piece := range d.Details {
		idx := 0
		if f, ok := piece["index"].(float64); ok {
			idx = int(f)
		}
		if a.pieces == nil {
			a.pieces = map[int]map[string]any{}
		}
		have, ok := a.pieces[idx]
		if !ok {
			a.pieces[idx] = piece
			a.ordered = append(a.ordered, idx)
			continue
		}
		for k, v := range piece {
			// Text arrives a chunk at a time; everything else (type, id,
			// signature, format) is the same on every piece or set once.
			if s, isStr := v.(string); isStr && (k == "text" || k == "summary" || k == "data") {
				prev, _ := have[k].(string)
				have[k] = prev + s
				continue
			}
			have[k] = v
		}
	}
	return text
}

// details is the stitched reasoning_details array, or nil if there was none.
func (a *reasoningAccumulator) details() json.RawMessage {
	if len(a.pieces) == 0 {
		return nil
	}
	sort.Ints(a.ordered)
	out := make([]map[string]any, 0, len(a.ordered))
	for _, idx := range a.ordered {
		out = append(out, a.pieces[idx])
	}
	raw, err := json.Marshal(out)
	if err != nil {
		return nil
	}
	return raw
}
