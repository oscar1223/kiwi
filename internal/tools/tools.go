// Package tools defines the Tool contract and the registry the agent draws on.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Tool is one capability the model can invoke.
//
// Run returns the observation handed back to the model. A returned error means
// the tool itself failed; the agent turns it into an error observation rather
// than aborting the turn, so the model gets a chance to recover.
type Tool interface {
	Name() string
	Description() string
	// Schema is the JSON Schema of Run's input.
	Schema() map[string]any
	Run(ctx context.Context, input json.RawMessage) (string, error)
}

// Registry holds the tools available to one agent.
type Registry struct {
	mu    sync.RWMutex
	tools map[string]Tool
}

func NewRegistry(ts ...Tool) *Registry {
	r := &Registry{tools: make(map[string]Tool, len(ts))}
	for _, t := range ts {
		r.Register(t)
	}
	return r
}

func (r *Registry) Register(t Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.tools[t.Name()] = t
}

func (r *Registry) Get(name string) (Tool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// Subset returns a new registry containing only the named tools that exist.
// Subagents use this to receive a restricted toolset.
func (r *Registry) Subset(names ...string) *Registry {
	r.mu.RLock()
	defer r.mu.RUnlock()
	sub := &Registry{tools: make(map[string]Tool, len(names))}
	for _, n := range names {
		if t, ok := r.tools[n]; ok {
			sub.tools[n] = t
		}
	}
	return sub
}

// Schemas returns tool descriptions for the model, in stable name order so
// prompt caching is not defeated by map iteration.
func (r *Registry) Schemas() []llm.ToolSchema {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.tools))
	for n := range r.tools {
		names = append(names, n)
	}
	sort.Strings(names)

	out := make([]llm.ToolSchema, 0, len(names))
	for _, n := range names {
		t := r.tools[n]
		out = append(out, llm.ToolSchema{
			Name:        t.Name(),
			Description: t.Description(),
			Schema:      t.Schema(),
		})
	}
	return out
}

// Run executes a tool call, reporting an unknown tool as a recoverable error
// observation rather than a hard failure.
func (r *Registry) Run(ctx context.Context, call llm.ToolCall) (string, error) {
	t, ok := r.Get(call.Name)
	if !ok {
		return "", fmt.Errorf("unknown tool %q", call.Name)
	}
	return t.Run(ctx, call.Input)
}

// Default returns the tools every Kiwi agent starts with.
func Default(workDir string, perms *permission.Broker) *Registry {
	fs := &FS{WorkDir: workDir, Perms: perms}
	return NewRegistry(
		ReadFile{fs},
		WriteFile{fs},
		EditFile{fs},
		Bash{WorkDir: workDir, Perms: perms},
	)
}
