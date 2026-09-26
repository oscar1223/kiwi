package llm

import "testing"

func TestParseReasoning(t *testing.T) {
	for in, want := range map[string]Reasoning{
		"": ReasoningDefault, "default": ReasoningDefault, "Auto": ReasoningDefault,
		"off": ReasoningOff, "none": ReasoningOff, "HIGH": ReasoningHigh, " xhigh ": ReasoningXHigh, "max": ReasoningMax,
	} {
		got, err := ParseReasoning(in)
		if err != nil || got != want {
			t.Errorf("ParseReasoning(%q) = %q, %v; want %q", in, got, err, want)
		}
	}
	if _, err := ParseReasoning("extreme"); err == nil {
		t.Error("an unknown level was accepted")
	}
}
