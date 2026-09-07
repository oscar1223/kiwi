package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
)

// fakeTool is a registered tool that does nothing, for testing the registry
// itself rather than any tool's behaviour.
type fakeTool struct {
	name   string
	result string
}

func (f fakeTool) Name() string           { return f.name }
func (f fakeTool) Description() string    { return "a tool that exists" }
func (f fakeTool) Schema() map[string]any { return map[string]any{"type": "object"} }
func (f fakeTool) Run(context.Context, json.RawMessage) (string, error) {
	return f.result, nil
}

// A tool that exists but was never registered is invisible to the model, and
// nothing else in the suite would notice.
func TestDefaultRegistryCarriesTheCoreToolset(t *testing.T) {
	r := Default(t.TempDir(), permission.NewBroker(permission.ModeAsk, nil))

	have := map[string]bool{}
	for _, s := range r.Schemas() {
		have[s.Name] = true
	}
	for _, want := range []string{
		"read_file", "write_file", "edit_file", "multi_edit",
		"grep", "glob", "ls", "bash",
	} {
		if !have[want] {
			t.Errorf("%s is not in the default registry", want)
		}
	}
}

// Every schema the model is shown has to be a usable JSON Schema object, or
// the provider rejects the whole request and the turn dies on arrival.
func TestEverySchemaIsAnObjectWithProperties(t *testing.T) {
	r := Default(t.TempDir(), permission.NewBroker(permission.ModeAsk, nil))

	for _, s := range r.Schemas() {
		if s.Description == "" {
			t.Errorf("%s has no description", s.Name)
		}
		if s.Schema["type"] != "object" {
			t.Errorf("%s: schema type = %v, want object", s.Name, s.Schema["type"])
		}
		if _, ok := s.Schema["properties"]; !ok {
			t.Errorf("%s: schema has no properties", s.Name)
		}
	}
}

func TestMuteHidesAToolFromTheModelButKeepsIt(t *testing.T) {
	r := NewRegistry(fakeTool{name: "keep"}, fakeTool{name: "hide"})

	if !r.Mute("hide", true) {
		t.Fatal("Mute reported an unknown tool")
	}
	for _, s := range r.Schemas() {
		if s.Name == "hide" {
			t.Error("a muted tool is still offered to the model")
		}
	}
	// Registered, not torn down: turning it back on has to be symmetric.
	if _, ok := r.Get("hide"); !ok {
		t.Error("muting unregistered the tool")
	}
	if names := r.Names(); len(names) != 2 {
		t.Errorf("Names() = %v, want both tools listed", names)
	}

	r.Mute("hide", false)
	if r.IsMuted("hide") {
		t.Error("the tool stayed muted after being turned back on")
	}
	if len(r.Schemas()) != 2 {
		t.Error("the tool did not come back into the schemas")
	}
}

// The model works from schemas it was sent before the mute and does not
// re-read them mid-turn, so the call has to be refused where it lands.
func TestAMutedToolRefusesToRun(t *testing.T) {
	r := NewRegistry(fakeTool{name: "dangerous", result: "ran"})
	r.Mute("dangerous", true)

	out, err := r.Run(context.Background(), llm.ToolCall{Name: "dangerous"})
	if err == nil {
		t.Fatalf("a muted tool ran anyway and returned %q", out)
	}
	if !strings.Contains(err.Error(), "/tools") {
		t.Errorf("the error does not say how to undo it: %v", err)
	}
}

func TestMuteReportsAnUnknownTool(t *testing.T) {
	r := NewRegistry()
	if r.Mute("nope", true) {
		t.Error("Mute accepted a tool that was never registered")
	}
}
