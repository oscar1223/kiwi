package llm

import (
	"encoding/json"
	"fmt"
	"strings"
)

// Reasoning is how hard a model thinks before it answers.
//
// It is one scale for every provider: each adapter maps it onto whatever its
// API calls the same idea (OpenRouter's reasoning object, OpenAI's
// reasoning_effort, Z.ai's thinking switch, Claude's effort). The zero value
// sends nothing, so the provider's own default applies.
type Reasoning string

const (
	ReasoningDefault Reasoning = ""
	ReasoningOff     Reasoning = "off"
	ReasoningLow     Reasoning = "low"
	ReasoningMedium  Reasoning = "medium"
	ReasoningHigh    Reasoning = "high"
	ReasoningXHigh   Reasoning = "xhigh"
	ReasoningMax     Reasoning = "max"
)

// ReasoningLevels lists the settable levels, least to most thinking.
var ReasoningLevels = []Reasoning{
	ReasoningOff, ReasoningLow, ReasoningMedium, ReasoningHigh, ReasoningXHigh, ReasoningMax,
}

// ParseReasoning reads a level as written in kiwi.json or typed after
// /reasoning. "default", "auto" and "" all mean the provider's default.
func ParseReasoning(s string) (Reasoning, error) {
	switch v := strings.ToLower(strings.TrimSpace(s)); v {
	case "", "default", "auto":
		return ReasoningDefault, nil
	case "none", "disabled":
		return ReasoningOff, nil
	default:
		for _, l := range ReasoningLevels {
			if Reasoning(v) == l {
				return l, nil
			}
		}
		return "", fmt.Errorf("unknown reasoning level %q (want off, low, medium, high, xhigh, max or default)", s)
	}
}

// Label is the level as shown to the user.
func (r Reasoning) Label() string {
	if r == ReasoningDefault {
		return "default"
	}
	return string(r)
}

// ReasoningTrace is what a model thought before one assistant message.
//
// Some APIs need it handed back on the next request of the same turn —
// Claude rejects a tool-use turn with thinking on unless the signed thinking
// blocks come back, and OpenRouter asks for reasoning_details the same way.
// Raw is opaque outside the adapter that wrote it, and Source says which
// adapter and model that was, so a trace is never replayed to another.
type ReasoningTrace struct {
	Source string          `json:"source"`
	Text   string          `json:"text,omitempty"`
	Raw    json.RawMessage `json:"raw,omitempty"`
}

// ReasoningReporter is implemented by providers that know which reasoning
// level they were built with, so the interface can show it.
type ReasoningReporter interface {
	Reasoning() Reasoning
}
