package permission

import (
	"context"
	"errors"
	"sync"
)

// ErrDenied is returned to a tool whose action the user refused.
var ErrDenied = errors.New("denied by the user")

// Request is one pending question for the user.
type Request struct {
	Action
	// Mode is the mode in effect when the question was raised.
	Mode Mode

	resp chan bool
	once sync.Once
}

// Allow resolves the request. Calling it more than once is safe, so a UI can
// wire the same request to several controls without bookkeeping.
func (r *Request) Allow() { r.answer(true) }

// Deny resolves the request negatively.
func (r *Request) Deny() { r.answer(false) }

func (r *Request) answer(v bool) {
	r.once.Do(func() {
		r.resp <- v
		close(r.resp)
	})
}

// Decider answers permission questions the policy could not settle.
//
// Implementations must be safe for concurrent use: with subagents, several
// goroutines ask at once.
type Decider interface {
	Decide(ctx context.Context, req *Request) (bool, error)
}

// Broker gates tool actions behind the current mode's policy and, when the
// policy abstains, behind a Decider.
//
// This replaces the prototype's single global event: the queue has one
// consumer and N concurrent producers, so two subagents asking at the same
// time queue up instead of clobbering each other.
type Broker struct {
	mu      sync.RWMutex
	mode    Mode
	decider Decider

	// onAuto is notified of automatic decisions so the UI can log why
	// something was allowed or blocked without a prompt appearing.
	onAuto func(req *Request, allowed bool)
}

func NewBroker(mode Mode, d Decider) *Broker {
	if !mode.Valid() {
		mode = ModeAsk
	}
	return &Broker{mode: mode, decider: d}
}

func (b *Broker) Mode() Mode {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.mode
}

func (b *Broker) SetMode(m Mode) {
	if !m.Valid() {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.mode = m
}

// OnAutoDecision registers a callback for decisions taken without asking.
func (b *Broker) OnAutoDecision(f func(req *Request, allowed bool)) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.onAuto = f
}

// Ask gates one action. It returns nil when the action may proceed, ErrDenied
// when it was refused, or ctx.Err() when the turn was cancelled while waiting.
func (b *Broker) Ask(ctx context.Context, a Action) error {
	b.mu.RLock()
	mode, decider, onAuto := b.mode, b.decider, b.onAuto
	b.mu.RUnlock()

	req := &Request{Action: a, Mode: mode, resp: make(chan bool, 1)}

	if allow, decided := Resolve(mode, a); decided {
		if onAuto != nil {
			onAuto(req, allow)
		}
		if allow {
			return nil
		}
		return ErrDenied
	}

	if decider == nil {
		// No UI attached: refuse rather than act unsupervised.
		return ErrDenied
	}

	allow, err := decider.Decide(ctx, req)
	if err != nil {
		return err
	}
	if !allow {
		return ErrDenied
	}
	return nil
}

// AllowAll approves everything. Used by `--yolo` and by tests.
type AllowAll struct{}

func (AllowAll) Decide(context.Context, *Request) (bool, error) { return true, nil }

// DenyAll refuses everything the policy did not already settle. This is the
// headless default: a non-interactive run has nobody to ask.
type DenyAll struct{}

func (DenyAll) Decide(context.Context, *Request) (bool, error) { return false, nil }

// Queue is a Decider that hands requests to a single consumer, typically the
// TUI event loop.
type Queue struct {
	ch chan *Request
}

func NewQueue(buffer int) *Queue {
	return &Queue{ch: make(chan *Request, buffer)}
}

// Requests is the channel the UI ranges over.
func (q *Queue) Requests() <-chan *Request { return q.ch }

func (q *Queue) Decide(ctx context.Context, req *Request) (bool, error) {
	select {
	case q.ch <- req:
	case <-ctx.Done():
		return false, ctx.Err()
	}

	select {
	case allow := <-req.resp:
		return allow, nil
	case <-ctx.Done():
		// Unblock any UI still holding the request.
		req.Deny()
		return false, ctx.Err()
	}
}

// ErrNoUI explains a refusal in a non-interactive run. It is phrased for the
// model, which would otherwise be told a human denied something no human saw.
var ErrNoUI = errors.New(
	"this action needs confirmation, but kiwi is running non-interactively so " +
		"there is nobody to ask. Re-run with --yolo to allow it, or work within " +
		"the current mode's limits")

// NonInteractive refuses anything the policy did not settle, explaining why.
type NonInteractive struct{}

func (NonInteractive) Decide(context.Context, *Request) (bool, error) { return false, ErrNoUI }

// Wait blocks until the request is answered or ctx is done.
//
// A Decider that hands requests to a UI uses this to park the calling tool's
// goroutine: on cancellation it denies the request itself, so a UI still
// holding it is not left waiting on a reader that has gone away.
func (r *Request) Wait(ctx context.Context) (bool, error) {
	select {
	case allow := <-r.resp:
		return allow, nil
	case <-ctx.Done():
		r.Deny()
		return false, ctx.Err()
	}
}
