// Package agent implements Kiwi's turn loop.
//
// The loop is deliberately small and owns no I/O of its own: it streams from a
// llm.Provider, executes tool calls against a tools.Registry, and reports
// progress through an Observer. Everything cancellable takes the ctx it was
// given, so cancelling a turn tears down the model stream and every child
// process a tool started.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/tools"
)

// DefaultMaxSteps bounds how many model↔tool round trips one turn may take.
// It is a runaway guard, not a feature: hitting it means something went wrong.
const DefaultMaxSteps = 50

// Observer receives turn progress. Every method must tolerate being called
// from the agent's goroutine and must not block for long. A nil Observer is
// valid; use NopObserver to avoid nil checks.
type Observer interface {
	OnText(delta string)
	OnToolCall(call llm.ToolCall)
	OnToolResult(call llm.ToolCall, output string, isErr bool)
	OnUsage(u llm.Usage)
}

type NopObserver struct{}

func (NopObserver) OnText(string)                           {}
func (NopObserver) OnToolCall(llm.ToolCall)                 {}
func (NopObserver) OnToolResult(llm.ToolCall, string, bool) {}
func (NopObserver) OnUsage(llm.Usage)                       {}

type Agent struct {
	Provider llm.Provider
	Tools    *tools.Registry
	System   string
	MaxSteps int
	// MaxTokens caps a single model response.
	MaxTokens int
}

// Result is the outcome of one completed turn.
type Result struct {
	// Text is the assistant's final prose.
	Text string
	// Messages are the messages produced this turn, ready to append to history.
	Messages []llm.Message
	Usage    llm.Usage
	Steps    int
}

// ErrMaxSteps is returned when a turn exceeds MaxSteps round trips.
var ErrMaxSteps = errors.New("agent: max steps exceeded")

// Run executes one turn: it appends input to history and drives the
// model↔tool loop until the model answers with prose and no tool calls.
//
// history is not mutated; the messages generated are returned in Result so the
// caller decides what to persist.
func (a *Agent) Run(ctx context.Context, input string, history []llm.Message, obs Observer) (*Result, error) {
	if obs == nil {
		obs = NopObserver{}
	}
	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = DefaultMaxSteps
	}

	// Working copy: history stays untouched for the caller.
	convo := make([]llm.Message, 0, len(history)+4)
	convo = append(convo, history...)
	turn := []llm.Message{{Role: llm.RoleUser, Content: input}}
	convo = append(convo, turn[0])

	res := &Result{}

	for step := 1; step <= maxSteps; step++ {
		res.Steps = step

		assistant, usage, err := a.stream(ctx, convo, obs)
		if err != nil {
			return nil, err
		}
		if usage != nil {
			res.Usage.InputTokens += usage.InputTokens
			res.Usage.OutputTokens += usage.OutputTokens
			obs.OnUsage(*usage)
		}

		convo = append(convo, *assistant)
		turn = append(turn, *assistant)

		// No tool calls means the model is done talking.
		if len(assistant.ToolCalls) == 0 {
			res.Text = assistant.Content
			res.Messages = turn
			return res, nil
		}

		for _, call := range assistant.ToolCalls {
			// Check cancellation between tools: a cancelled turn should not
			// keep firing the remaining calls of a batch.
			if err := ctx.Err(); err != nil {
				return nil, err
			}

			obs.OnToolCall(call)
			output, runErr := a.Tools.Run(ctx, call)

			// A cancelled tool cancels the turn; any other failure becomes an
			// observation so the model can correct itself.
			if runErr != nil && (errors.Is(runErr, context.Canceled) || errors.Is(runErr, context.DeadlineExceeded)) {
				return nil, runErr
			}

			isErr := runErr != nil
			if isErr {
				output = "Error: " + runErr.Error()
			}
			obs.OnToolResult(call, output, isErr)

			msg := llm.Message{
				Role:       llm.RoleTool,
				Content:    output,
				ToolCallID: call.ID,
				ToolName:   call.Name,
				IsError:    isErr,
			}
			convo = append(convo, msg)
			turn = append(turn, msg)
		}
	}

	return nil, fmt.Errorf("%w (%d)", ErrMaxSteps, maxSteps)
}

// stream consumes one model response, forwarding text deltas as they arrive
// and assembling the complete assistant message.
func (a *Agent) stream(ctx context.Context, convo []llm.Message, obs Observer) (*llm.Message, *llm.Usage, error) {
	req := llm.Request{
		System:    a.System,
		Messages:  convo,
		Tools:     a.Tools.Schemas(),
		MaxTokens: a.MaxTokens,
	}

	var (
		final *llm.Message
		usage *llm.Usage
	)
	for ev, err := range a.Provider.Stream(ctx, req) {
		if err != nil {
			return nil, nil, err
		}
		switch ev.Type {
		case llm.EventTextDelta:
			obs.OnText(ev.Text)
		case llm.EventDone:
			final = ev.Message
			usage = ev.Usage
		}
	}
	if err := ctx.Err(); err != nil {
		return nil, nil, err
	}
	if final == nil {
		return nil, nil, errors.New("agent: provider stream ended without a final message")
	}
	return final, usage, nil
}

// DecodeInput is a helper for tools: unmarshal a call's input into v.
func DecodeInput(raw json.RawMessage, v any) error {
	if len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, v)
}
