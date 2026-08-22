// Package tui is Kiwi's terminal interface.
package tui

import (
	"context"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Agent event messages. They all carry the generation they belong to so a
// cancelled turn's late events can be discarded instead of painted.
type (
	textDeltaMsg struct {
		gen   int
		delta string
	}
	toolCallMsg struct {
		gen  int
		call llm.ToolCall
	}
	toolResultMsg struct {
		gen    int
		call   llm.ToolCall
		output string
		isErr  bool
	}
	turnDoneMsg struct {
		gen      int
		messages []llm.Message
		usage    llm.Usage
	}
	turnErrMsg struct {
		gen int
		err error
	}

	// historyPersistedMsg reports the outcome of a background persistTurn
	// call: the history to adopt (possibly compacted), or an error if saving
	// failed.
	historyPersistedMsg struct {
		gen     int
		history []llm.Message
		err     error
	}

	// permissionMsg asks the user to approve an action.
	permissionMsg struct{ req *permission.Request }
	// autoDecisionMsg records a decision the mode policy took on its own, so
	// there is a trace of why something was allowed or blocked silently.
	autoDecisionMsg struct {
		req     *permission.Request
		allowed bool
	}
)

// Events is the single stream the Bubble Tea loop consumes.
//
// Agent progress and permission questions both land here, which is what keeps
// Update the only place that touches model state. It also implements
// permission.Decider, so the same object can be handed to the broker before
// the model exists — the broker needs a decider at construction time, and the
// model needs the broker.
type Events struct {
	ch chan tea.Msg
}

func NewEvents() *Events { return &Events{ch: make(chan tea.Msg, 128)} }

// send delivers a message unless the turn is being torn down.
func (e *Events) send(ctx context.Context, msg tea.Msg) {
	select {
	case e.ch <- msg:
	case <-ctx.Done():
	}
}

// next is the Cmd that pulls one message and re-arms itself.
func (e *Events) next() tea.Cmd {
	return func() tea.Msg { return <-e.ch }
}

// Decide routes a permission question to the Bubble Tea loop and blocks the
// asking goroutine until Update answers.
//
// Several tools — and, later, several subagents — can be blocked here at once;
// each waits on its own request, so answers never cross.
func (e *Events) Decide(ctx context.Context, req *permission.Request) (bool, error) {
	select {
	case e.ch <- permissionMsg{req: req}:
	case <-ctx.Done():
		return false, ctx.Err()
	}
	return req.Wait(ctx)
}

// LogAutoDecision records a decision the mode policy took on its own. It is
// wired to the broker so silent allows and blocks still leave a trace.
func (e *Events) LogAutoDecision(req *permission.Request, allowed bool) {
	select {
	case e.ch <- autoDecisionMsg{req: req, allowed: allowed}:
	default:
		// Never block the agent to write a log line.
	}
}

// observer forwards agent progress into the event stream.
type observer struct {
	events *Events
	ctx    context.Context
	gen    int
}

func (o *observer) OnText(delta string) {
	o.events.send(o.ctx, textDeltaMsg{gen: o.gen, delta: delta})
}

func (o *observer) OnToolCall(call llm.ToolCall) {
	o.events.send(o.ctx, toolCallMsg{gen: o.gen, call: call})
}

func (o *observer) OnToolResult(call llm.ToolCall, output string, isErr bool) {
	o.events.send(o.ctx, toolResultMsg{gen: o.gen, call: call, output: output, isErr: isErr})
}

func (o *observer) OnUsage(llm.Usage) {}

// runTurn drives one agent turn in the background, reporting through events.
func runTurn(ctx context.Context, a *agent.Agent, gen int, input string, history []llm.Message, ev *Events) {
	obs := &observer{events: ev, ctx: ctx, gen: gen}
	res, err := a.Run(ctx, input, history, obs)
	if err != nil {
		ev.send(context.WithoutCancel(ctx), turnErrMsg{gen: gen, err: err})
		return
	}
	ev.send(context.WithoutCancel(ctx), turnDoneMsg{
		gen:      gen,
		messages: res.Messages,
		usage:    res.Usage,
	})
}
