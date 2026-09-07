// Package tools defines the Tool contract and the registry the agent draws on.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"sync"

	"github.com/oscar1223/kiwi/internal/diagnostics"
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
	// muted are tools hidden from the model for this session. They stay
	// registered so unmuting is symmetric — /tools is a switch, not a
	// teardown.
	muted map[string]bool
}

func NewRegistry(ts ...Tool) *Registry {
	r := &Registry{tools: make(map[string]Tool, len(ts)), muted: map[string]bool{}}
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
	sub := &Registry{tools: make(map[string]Tool, len(names)), muted: map[string]bool{}}
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
		if r.muted[n] {
			continue
		}
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
	// A muted tool can still be called: the schemas the model is working from
	// were sent before the mute, and a model does not re-read them mid-turn.
	// Refusing here rather than silently running is the difference between
	// /tools meaning something and /tools looking like it does.
	if r.IsMuted(call.Name) {
		return "", fmt.Errorf("tool %q is turned off for this session (/tools to turn it back on)", call.Name)
	}
	return t.Run(ctx, call.Input)
}

// Names lists every registered tool, muted or not, in stable order.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.tools))
	for n := range r.tools {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// Mute hides a tool from the model without unregistering it.
//
// This is the escape hatch work mode needs. Work mode auto-approves MCP calls,
// and an MCP tool is opaque — Kiwi cannot tell one that reads from one that
// writes to production. Being able to switch a single tool off for the session,
// without editing config or restarting, is what makes that trade recoverable.
func (r *Registry) Mute(name string, muted bool) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.tools[name]; !ok {
		return false
	}
	if r.muted == nil {
		r.muted = map[string]bool{}
	}
	if muted {
		r.muted[name] = true
	} else {
		delete(r.muted, name)
	}
	return true
}

// IsMuted reports whether a tool is hidden from the model.
func (r *Registry) IsMuted(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.muted[name]
}

// Default returns the tools every Kiwi agent starts with, plus any extra
// tools the caller supplies — skills.LoadSkill when skills exist, MCP-server
// tools when any are configured.
func Default(workDir string, perms *permission.Broker, extra ...Tool) *Registry {
	fs := &FS{WorkDir: workDir, Perms: perms, Diag: diagnostics.New(workDir)}
	r := NewRegistry(
		ReadFile{fs},
		WriteFile{fs},
		EditFile{fs},
		MultiEdit{fs},
		// Search costs no permission plumbing: like ReadFile, these never
		// call Perms.Ask, so they are available in every mode including Plan.
		Glob{fs},
		Grep{fs},
		List{fs},
		Bash{WorkDir: workDir, Perms: perms},
	)
	for _, t := range extra {
		r.Register(t)
	}
	return r
}
