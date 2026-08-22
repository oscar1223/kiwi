// Package llm defines Kiwi's provider-agnostic model interface.
//
// Everything above this package speaks in terms of Message, ToolCall and
// Event; nothing above it imports a vendor SDK. That boundary is what makes
// the agent loop testable without network access and what keeps adding a new
// provider a matter of writing one adapter.
package llm

import (
	"context"
	"encoding/json"
	"iter"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ToolCall is a model's request to invoke a tool.
type ToolCall struct {
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

// Message is one entry of conversation history.
//
// A single assistant message may carry both Content and ToolCalls: models
// routinely narrate what they are about to do before calling a tool.
type Message struct {
	Role      Role       `json:"role"`
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`

	// Set on RoleTool messages only.
	ToolCallID string `json:"tool_call_id,omitempty"`
	ToolName   string `json:"tool_name,omitempty"`
	IsError    bool   `json:"is_error,omitempty"`
}

// ToolSchema describes a tool to the model. Schema is a JSON Schema object.
type ToolSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Schema      map[string]any `json:"schema"`
}

type Request struct {
	System    string
	Messages  []Message
	Tools     []ToolSchema
	MaxTokens int
	// Temperature is a pointer so "unset" is distinguishable from 0.
	Temperature *float64
}

type EventType int

const (
	// EventTextDelta carries an incremental chunk of assistant prose.
	EventTextDelta EventType = iota
	// EventToolCall carries one fully assembled tool call.
	EventToolCall
	// EventDone is emitted once, last, with the final assembled message.
	EventDone
)

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type Event struct {
	Type     EventType
	Text     string
	ToolCall *ToolCall
	// Message is set on EventDone: the complete assistant message, ready to
	// append to history.
	Message *Message
	Usage   *Usage
}

// Provider is a streaming chat-completions backend.
//
// Stream returns a pull-based iterator. Nothing runs until it is ranged over,
// and abandoning the range mid-way must release the underlying connection —
// which is also how cancellation propagates: callers pass a ctx they can
// cancel, and the iterator stops yielding.
type Provider interface {
	Name() string
	Model() string
	Stream(ctx context.Context, req Request) iter.Seq2[Event, error]
}
