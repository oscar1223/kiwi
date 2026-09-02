package tui

import (
	"context"

	"charm.land/bubbles/v2/textinput"
	"github.com/oscar1223/kiwi/internal/tools"
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
	// value in the one-line record filed in the transcript once answered —
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

// questionRequest is a tea.Msg asking Update to show one clarifying question
// — the model's ask_questions tool, driven through AskQuestions below — with
// single- or multi-select options plus a free-text "Other", and report back
// the chosen values.
type questionRequest struct {
	q    tools.Question
	resp chan questionResult
}

type questionResult struct {
	values []string
	ok     bool
}

// AskQuestions drives the model's ask_questions tool: it shows the questions
// one after another and blocks the calling goroutine (the tool's Run) until
// they are all answered or one is cancelled. It implements tools.Asker.
//
// Cancelling any question — esc — abandons the whole batch: a partial answer
// set is not something the model asked for and is not worth guessing at.
func (e *Events) AskQuestions(ctx context.Context, qs []tools.Question) ([]tools.Answer, bool) {
	answers := make([]tools.Answer, 0, len(qs))
	for _, q := range qs {
		req := &questionRequest{q: q, resp: make(chan questionResult, 1)}
		select {
		case e.ch <- req:
		case <-ctx.Done():
			return nil, false
		}
		select {
		case r := <-req.resp:
			if !r.ok {
				return nil, false
			}
			answers = append(answers, tools.Answer{Question: q.Question, Values: r.values})
		case <-ctx.Done():
			return nil, false
		}
	}
	return answers, true
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

// questionState is the live, re-rendered state of an open ask_questions
// prompt for one question. Rows are the question's options plus one
// synthetic trailing "Other" row at index otherIndex(), which opens
// otherInput instead of resolving directly.
type questionState struct {
	req   *questionRequest
	index int

	// selected holds toggled rows for a multi-select question, keyed by row
	// index (otherIndex() included). Unused for single-select, which
	// resolves as soon as a row is chosen.
	selected map[int]bool

	// otherActive is true while the free-text "Other" answer is being typed.
	// otherText holds the last value submitted through it.
	otherActive bool
	otherText   string
	otherInput  textinput.Model
}

func (q *questionState) otherIndex() int { return len(q.req.q.Options) }

func (q *questionState) up() {
	if q.index > 0 {
		q.index--
	}
}

func (q *questionState) down() {
	if q.index < q.otherIndex() {
		q.index++
	}
}

func (q *questionState) setSelected(i int, v bool) {
	if q.selected == nil {
		q.selected = map[int]bool{}
	}
	q.selected[i] = v
}

// selectedLabels returns the labels of toggled options, in option order. It
// never includes the synthetic "Other" row — callers add otherText for that
// themselves, since it has no Label.
func (q *questionState) selectedLabels() []string {
	labels := make([]string, 0, len(q.selected))
	for i, opt := range q.req.q.Options {
		if q.selected[i] {
			labels = append(labels, opt.Label)
		}
	}
	return labels
}

// openOtherInput switches the prompt into free-text mode, prefilled with
// whatever "Other" text was last submitted (empty the first time).
func (q *questionState) openOtherInput(width int) {
	ti := textinput.New()
	ti.SetVirtualCursor(false)
	ti.Placeholder = "type your own answer"
	ti.SetValue(q.otherText)
	ti.Focus()
	ti.SetWidth(max(20, width-4))
	q.otherInput = ti
	q.otherActive = true
}
