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
	"github.com/oscar1223/kiwi/internal/telemetry"
	"github.com/oscar1223/kiwi/internal/tools"
	"golang.org/x/sync/errgroup"
)

// DefaultMaxSteps bounds how many model↔tool round trips one turn may take.
// It is a runaway guard, not a feature: hitting it means something went wrong.
const DefaultMaxSteps = 50

// Observer receives turn progress. Every method must tolerate being called
// from the agent's goroutine and must not block for long. A nil Observer is
// valid; use NopObserver to avoid nil checks.
//
// Implementations must also be safe for concurrent calls: multiple task
// (subagent) calls in the same batch run in their own goroutines, and each
// reports through the same Observer.
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

	ctx, turnSpan := telemetry.StartTurn(ctx, a.Provider.Name(), a.Provider.Model())
	// turnErr is a different variable from any tool's own error below: the
	// deferred EndTurn must record how the *turn* ended, not the outcome of
	// whichever tool happened to run last.
	var turnErr error
	defer func() { telemetry.EndTurn(turnSpan, turnErr) }()

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
			turnErr = err
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

		// task calls are dispatched together into their own goroutines before
		// the sequential loop below even starts, so several delegated
		// subagents run concurrently with each other — and with whatever
		// non-task calls follow — rather than one at a time. Everything else
		// stays strictly sequential: a model that asks for two file edits or
		// two bash commands in one step should not have them race.
		results := make([]toolCallResult, len(assistant.ToolCalls))
		var group errgroup.Group
		for i, call := range assistant.ToolCalls {
			if call.Name != TaskName {
				continue
			}
			i, call := i, call
			group.Go(func() error {
				results[i] = a.runToolCall(ctx, call, obs)
				return nil // errors travel in the result, not the group
			})
		}

		for i, call := range assistant.ToolCalls {
			if call.Name == TaskName {
				continue // already dispatched above
			}
			if err := ctx.Err(); err != nil {
				group.Wait()
				turnErr = err
				return nil, err
			}
			results[i] = a.runToolCall(ctx, call, obs)
			if results[i].cancelErr != nil {
				group.Wait()
				turnErr = results[i].cancelErr
				return nil, results[i].cancelErr
			}
		}
		group.Wait()

		// Reassembled in the model's original order regardless of which
		// pass computed each result, so a cancelled call is reported
		// deterministically no matter which goroutine hit it first.
		for _, r := range results {
			if r.cancelErr != nil {
				turnErr = r.cancelErr
				return nil, r.cancelErr
			}
			msg := llm.Message{
				Role:       llm.RoleTool,
				Content:    r.output,
				ToolCallID: r.call.ID,
				ToolName:   r.call.Name,
				IsError:    r.isErr,
			}
			convo = append(convo, msg)
			turn = append(turn, msg)
		}
	}

	turnErr = fmt.Errorf("%w (%d)", ErrMaxSteps, maxSteps)
	return nil, turnErr
}

// stream consumes one model response, forwarding text deltas as they arrive
// and assembling the complete assistant message.
func (a *Agent) stream(ctx context.Context, convo []llm.Message, obs Observer) (final *llm.Message, usage *llm.Usage, err error) {
	ctx, span := telemetry.StartModelCall(ctx, a.Provider.Model())
	defer func() {
		var in, out int
		if usage != nil {
			in, out = usage.InputTokens, usage.OutputTokens
		}
		telemetry.EndModelCall(span, in, out, err)
	}()

	req := llm.Request{
		System:    a.System,
		Messages:  convo,
		Tools:     a.Tools.Schemas(),
		MaxTokens: a.MaxTokens,
	}

	for ev, streamErr := range a.Provider.Stream(ctx, req) {
		if streamErr != nil {
			err = streamErr
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
	if ctxErr := ctx.Err(); ctxErr != nil {
		err = ctxErr
		return nil, nil, err
	}
	if final == nil {
		err = errors.New("agent: provider stream ended without a final message")
		return nil, nil, err
	}
	return final, usage, nil
}

// toolCallResult is one tool call's outcome, collected regardless of whether
// it ran as part of the sequential pass or a parallel task batch, so both
// can be merged back into the model's original call order afterward.
type toolCallResult struct {
	call   llm.ToolCall
	output string
	isErr  bool
	// cancelErr is set only for context.Canceled/DeadlineExceeded — the one
	// case that must abort the whole turn rather than become an observation.
	cancelErr error
}

// runToolCall executes one tool call and reports it through obs. It is safe
// to call from multiple goroutines at once, each with its own call — see
// Observer's own concurrency note.
func (a *Agent) runToolCall(ctx context.Context, call llm.ToolCall, obs Observer) toolCallResult {
	obs.OnToolCall(call)
	toolCtx, toolSpan := telemetry.StartTool(ctx, call.Name)
	output, callErr := a.Tools.Run(toolCtx, call)
	telemetry.EndTool(toolSpan, callErr != nil)

	if callErr != nil && (errors.Is(callErr, context.Canceled) || errors.Is(callErr, context.DeadlineExceeded)) {
		return toolCallResult{call: call, cancelErr: callErr}
	}

	isErr := callErr != nil
	if isErr {
		output = "Error: " + callErr.Error()
	}
	obs.OnToolResult(call, output, isErr)
	return toolCallResult{call: call, output: output, isErr: isErr}
}

// DecodeInput is a helper for tools: unmarshal a call's input into v.
func DecodeInput(raw json.RawMessage, v any) error {
	if len(raw) == 0 {
		return nil
	}
	return json.Unmarshal(raw, v)
}
