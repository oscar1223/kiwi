package remote

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
)

// DefaultApprovalTimeout is how long an approval waits for a button press
// before refusing.
const DefaultApprovalTimeout = 5 * time.Minute

// ErrApprovalTimeout tells the model why an action was refused when nobody
// answered. It is not "denied by the user": nobody decided anything.
var ErrApprovalTimeout = errors.New(
	"this action needed approval from the user on Telegram, who did not answer in time, " +
		"so it was refused. Carry on without it, or explain what you needed it for")

// maxDiff bounds the diff shown with an approval, so the whole question fits
// in one message with room to spare.
const maxDiff = 2500

// Approver is a permission.Decider that asks on Telegram, with Allow and Deny
// buttons under the question.
//
// In work mode the policy settles almost everything by itself and only
// dangerous commands reach a Decider, so this fires rarely: autonomy, with a
// circuit breaker in your pocket.
//
// It asks in the chat of the turn in progress, which Session attaches for the
// length of the turn. It is safe for concurrent use: subagents can ask at the
// same time, and each question gets its own message and buttons.
type Approver struct {
	// Timeout is how long to wait for an answer. Zero means
	// DefaultApprovalTimeout.
	Timeout time.Duration

	mu      sync.Mutex
	conv    Conversation
	next    int
	pending map[int]*permission.Request
}

// attach makes conv the chat questions go to, until detach.
func (a *Approver) attach(conv Conversation) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.conv = conv
}

func (a *Approver) detach() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.conv = nil
}

// Decide implements permission.Decider.
func (a *Approver) Decide(ctx context.Context, req *permission.Request) (bool, error) {
	a.mu.Lock()
	conv := a.conv
	if conv == nil {
		a.mu.Unlock()
		// No turn from Telegram in progress, so nobody to ask.
		return false, permission.ErrNoUI
	}
	if a.pending == nil {
		a.pending = map[int]*permission.Request{}
	}
	a.next++
	id := a.next
	a.pending[id] = req
	a.mu.Unlock()

	defer func() {
		a.mu.Lock()
		delete(a.pending, id)
		a.mu.Unlock()
	}()

	question := renderQuestion(req)
	msgID, err := conv.Send(ctx, question, []Button{
		{Text: "✅ Permitir", Data: callbackData(id, true)},
		{Text: "🚫 Denegar", Data: callbackData(id, false)},
	})
	if err != nil {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		return false, fmt.Errorf("could not ask for approval on Telegram: %w", err)
	}

	timeout := a.Timeout
	if timeout <= 0 {
		timeout = DefaultApprovalTimeout
	}
	waitCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	allow, err := req.Wait(waitCtx)

	// The outcome replaces the buttons, so the question cannot be answered
	// twice and the chat keeps a record of what was decided.
	var outcome string
	switch {
	case err == nil && allow:
		outcome = "✅ Permitido"
	case err == nil:
		outcome = "🚫 Denegado"
	case ctx.Err() != nil:
		outcome = "✗ Turno interrumpido"
	default:
		outcome = fmt.Sprintf("⌛ Sin respuesta en %s: denegado", timeout.Round(time.Second))
		err = ErrApprovalTimeout
	}
	editCtx := ctx
	if ctx.Err() != nil {
		// Still worth leaving the message in a final state on shutdown.
		var cancel context.CancelFunc
		editCtx, cancel = context.WithTimeout(context.WithoutCancel(ctx), 5*time.Second)
		defer cancel()
	}
	_ = conv.Edit(editCtx, msgID, question+"\n\n"+outcome)

	if err != nil {
		if ctx.Err() != nil {
			return false, ctx.Err()
		}
		return false, err
	}
	return allow, nil
}

// HandleCallback is the Bot's CallbackHandler for approval buttons. The Bot
// has already checked that whoever pressed is on the allowed list.
func (a *Approver) HandleCallback(_ context.Context, q CallbackQuery) string {
	id, allow, ok := parseCallbackData(q.Data)
	if !ok {
		return ""
	}
	a.mu.Lock()
	req := a.pending[id]
	a.mu.Unlock()
	if req == nil {
		return "Esta pregunta ya no está pendiente."
	}
	if allow {
		req.Allow()
		return "Permitido"
	}
	req.Deny()
	return "Denegado"
}

func renderQuestion(req *permission.Request) string {
	var b strings.Builder
	if req.Dangerous {
		b.WriteString("⚠️ ")
	}
	fmt.Fprintf(&b, "Kiwi pide permiso (%s):\n\n%s", req.Name, req.Detail)
	if diff := strings.TrimSpace(req.Diff); diff != "" {
		r := []rune(diff)
		if len(r) > maxDiff {
			diff = string(r[:maxDiff]) + "\n… (diff recortado)"
		}
		b.WriteString("\n\n")
		b.WriteString(diff)
	}
	return b.String()
}

const callbackPrefix = "perm:"

func callbackData(id int, allow bool) string {
	verb := "deny"
	if allow {
		verb = "allow"
	}
	return fmt.Sprintf("%s%d:%s", callbackPrefix, id, verb)
}

func parseCallbackData(data string) (id int, allow, ok bool) {
	rest, found := strings.CutPrefix(data, callbackPrefix)
	if !found {
		return 0, false, false
	}
	num, verb, found := strings.Cut(rest, ":")
	if !found {
		return 0, false, false
	}
	id, err := strconv.Atoi(num)
	if err != nil {
		return 0, false, false
	}
	switch verb {
	case "allow":
		return id, true, true
	case "deny":
		return id, false, true
	}
	return 0, false, false
}
