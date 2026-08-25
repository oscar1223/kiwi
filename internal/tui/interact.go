package tui

import (
	"context"

	"charm.land/bubbles/v2/textinput"
)

// pickOption is one choice in a picker.
type pickOption struct {
	Label string
	Value string
}

// pickRequest is a tea.Msg asking Update to show an arrow-navigable list and
// report back which option the user chose.
type pickRequest struct {
	title   string
	options []pickOption
	resp    chan pickResult
	// onHighlight, if set, is called with the highlighted option's value on
	// every up/down move — before the user confirms anything. It runs
	// directly from onPickKey, which is Update's own goroutine, so it is
	// safe to mutate shared UI state from it (see the theme picker's live
	// preview). Left nil, an ordinary picker does nothing special while
	// browsing.
	onHighlight func(value string)
	// onCancel, if set, is called when the user cancels with esc — also
	// directly from onPickKey, for the same reason onHighlight is: reverting
	// a live preview touches the same shared UI state, so it must happen on
	// Update's goroutine too, not in the flow's own goroutine after Pick
	// returns.
	onCancel func()
}

type pickResult struct {
	value string
	ok    bool
}

// textRequest is a tea.Msg asking Update to show a single-line text prompt.
type textRequest struct {
	title       string
	placeholder string
	defaultVal  string
	// secret masks input as it is typed (API keys, tokens) and masks the
	// value in the one-line record printed to scrollback once answered —
	// masking only the live typing and then echoing the real value back
	// would defeat the point.
	secret bool
	resp   chan textResult
}

type textResult struct {
	value string
	ok    bool
}

// Pick shows an arrow-navigable list and blocks the calling goroutine until
// the user chooses one, cancels (esc), or ctx is done.
//
// This is the same pattern permission.Broker uses for tool confirmations:
// a command flow (started as its own goroutine, see runCommandFlow) reads
// like a straight-line script — list this, ask that, act — while the actual
// prompting happens on Update's goroutine, the only one allowed to touch
// Model state.
func (e *Events) Pick(ctx context.Context, title string, options []pickOption) (string, bool) {
	return e.pick(ctx, title, options, nil, nil)
}

// PickWithPreview is Pick plus a live-preview hook: onHighlight is called
// with the highlighted option's value on every up/down move, before the
// user confirms anything, and onCancel is called if the user cancels with
// esc so the preview can be reverted. See pickRequest.onHighlight and
// pickRequest.onCancel for why both must run this way instead of being
// handled after Pick returns.
func (e *Events) PickWithPreview(ctx context.Context, title string, options []pickOption, onHighlight func(value string), onCancel func()) (string, bool) {
	return e.pick(ctx, title, options, onHighlight, onCancel)
}

func (e *Events) pick(ctx context.Context, title string, options []pickOption, onHighlight func(value string), onCancel func()) (string, bool) {
	req := &pickRequest{title: title, options: options, onHighlight: onHighlight, onCancel: onCancel, resp: make(chan pickResult, 1)}
	select {
	case e.ch <- req:
	case <-ctx.Done():
		return "", false
	}
	select {
	case r := <-req.resp:
		return r.value, r.ok
	case <-ctx.Done():
		return "", false
	}
}

// Text shows a single-line prompt and blocks until the user submits, cancels,
// or ctx is done. An empty submission is reported as ok with an empty value —
// callers that require non-empty input check for that themselves, the same
// way the prototype's flows did.
func (e *Events) Text(ctx context.Context, title, placeholder, defaultVal string) (string, bool) {
	req := &textRequest{title: title, placeholder: placeholder, defaultVal: defaultVal, resp: make(chan textResult, 1)}
	select {
	case e.ch <- req:
	case <-ctx.Done():
		return "", false
	}
	select {
	case r := <-req.resp:
		return r.value, r.ok
	case <-ctx.Done():
		return "", false
	}
}

// SecretText is Text with the input masked as it is typed and in the record
// printed once answered — for API keys and other values that should never
// sit in plaintext on screen or in scrollback.
func (e *Events) SecretText(ctx context.Context, title, placeholder string) (string, bool) {
	req := &textRequest{title: title, placeholder: placeholder, secret: true, resp: make(chan textResult, 1)}
	select {
	case e.ch <- req:
	case <-ctx.Done():
		return "", false
	}
	select {
	case r := <-req.resp:
		return r.value, r.ok
	case <-ctx.Done():
		return "", false
	}
}

// Confirm is Pick specialised to a yes/no question.
func (e *Events) Confirm(ctx context.Context, title string) bool {
	v, ok := e.Pick(ctx, title, []pickOption{{"Yes", "yes"}, {"No", "no"}})
	return ok && v == "yes"
}

// pickState is the live, re-rendered state of an open picker.
type pickState struct {
	req   *pickRequest
	index int
}

func (p *pickState) up() {
	if p.index > 0 {
		p.index--
	}
}

func (p *pickState) down() {
	if p.index < len(p.req.options)-1 {
		p.index++
	}
}

func (p *pickState) selected() pickOption {
	return p.req.options[p.index]
}

// textState is the live, re-rendered state of an open text prompt.
type textState struct {
	req   *textRequest
	input textinput.Model
}
