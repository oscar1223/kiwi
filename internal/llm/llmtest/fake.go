// Package llmtest provides a scripted llm.Provider for tests.
//
// Having our own provider interface is what makes this possible: the whole
// agent loop, including tool calling and cancellation, can be exercised with
// no network and no API key.
package llmtest

import (
	"context"
	"encoding/json"
	"iter"
	"sync"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
)

// Step is one scripted model response.
type Step struct {
	// Text is emitted as deltas (split per rune group by Chunks, or whole).
	Text string
	// Chunks, if set, overrides Text and is emitted delta by delta.
	Chunks []string
	// ToolCalls are emitted after the text and included in the final message.
	ToolCalls []llm.ToolCall
	// Err, if set, is yielded instead of any events.
	Err error
	// Hook runs when this step begins streaming; useful for triggering
	// cancellation mid-turn.
	Hook func()
	// Delay is waited before each chunk, so a test can catch a turn while it
	// is still streaming. It respects ctx, like a real provider would.
	Delay time.Duration
}

// Fake is a Provider that replays Steps in order.
type Fake struct {
	Steps []Step

	mu       sync.Mutex
	calls    int
	Requests []llm.Request
}

func (f *Fake) Name() string  { return "fake" }
func (f *Fake) Model() string { return "fake-1" }

// Calls reports how many times Stream was consumed.
func (f *Fake) Calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls
}

func (f *Fake) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.Event, error] {
	f.mu.Lock()
	idx := f.calls
	f.calls++
	f.Requests = append(f.Requests, req)
	f.mu.Unlock()

	return func(yield func(llm.Event, error) bool) {
		if idx >= len(f.Steps) {
			yield(llm.Event{}, errUnexpectedCall{idx})
			return
		}
		step := f.Steps[idx]
		if step.Hook != nil {
			step.Hook()
		}
		if step.Err != nil {
			yield(llm.Event{}, step.Err)
			return
		}

		chunks := step.Chunks
		if chunks == nil && step.Text != "" {
			chunks = []string{step.Text}
		}
		var text string
		for _, c := range chunks {
			if step.Delay > 0 {
				select {
				case <-time.After(step.Delay):
				case <-ctx.Done():
					yield(llm.Event{}, ctx.Err())
					return
				}
			}
			if err := ctx.Err(); err != nil {
				yield(llm.Event{}, err)
				return
			}
			text += c
			if !yield(llm.Event{Type: llm.EventTextDelta, Text: c}, nil) {
				return
			}
		}
		for _, tc := range step.ToolCalls {
			if !yield(llm.Event{Type: llm.EventToolCall, ToolCall: &tc}, nil) {
				return
			}
		}
		if err := ctx.Err(); err != nil {
			yield(llm.Event{}, err)
			return
		}
		yield(llm.Event{
			Type: llm.EventDone,
			Message: &llm.Message{
				Role:      llm.RoleAssistant,
				Content:   text,
				ToolCalls: step.ToolCalls,
			},
			Usage: &llm.Usage{InputTokens: 10, OutputTokens: 5},
		}, nil)
	}
}

type errUnexpectedCall struct{ n int }

func (e errUnexpectedCall) Error() string {
	return "llmtest: unscripted provider call #" + itoa(e.n)
}

func itoa(n int) string {
	b, _ := json.Marshal(n)
	return string(b)
}

// Call builds a tool call with JSON-encoded input.
func Call(id, name string, input any) llm.ToolCall {
	raw, err := json.Marshal(input)
	if err != nil {
		panic(err)
	}
	return llm.ToolCall{ID: id, Name: name, Input: raw}
}
