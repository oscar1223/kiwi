package remote

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
)

var rmrf = permission.Action{Name: permission.ActionBash, Detail: "rm -rf /tmp/kiwi-test", Dangerous: true}

// ask runs a dangerous action through a work-mode broker, the way a tool
// would, and returns a channel with the broker's verdict.
func ask(ctx context.Context, a *Approver, action permission.Action) <-chan error {
	broker := permission.NewBroker(permission.ModeWork, a)
	done := make(chan error, 1)
	go func() { done <- broker.Ask(ctx, action) }()
	return done
}

// question waits for the approval message and returns it.
func question(t *testing.T, conv *fakeConv) convOp {
	t.Helper()
	var q convOp
	waitFor(t, "the approval question", func() bool {
		for _, op := range conv.snapshot() {
			if !op.edit && len(op.buttons) > 0 {
				q = op
				return true
			}
		}
		return false
	})
	return q
}

func press(a *Approver, q convOp, allow bool) string {
	b := q.buttons[0][1] // deny
	if allow {
		b = q.buttons[0][0]
	}
	return a.HandleCallback(context.Background(), CallbackQuery{ID: "cb", From: User{ID: 42}, Data: b.Data})
}

func verdict(t *testing.T, done <-chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(2 * time.Second):
		t.Fatal("the broker is still waiting for an answer")
		return nil
	}
}

func lastEdit(conv *fakeConv) string {
	ops := conv.snapshot()
	for i := len(ops) - 1; i >= 0; i-- {
		if ops[i].edit {
			return ops[i].text
		}
	}
	return ""
}

func TestApproveAllows(t *testing.T) {
	a, conv := &Approver{}, &fakeConv{}
	a.attach(conv)
	done := ask(context.Background(), a, rmrf)

	q := question(t, conv)
	if !strings.Contains(q.text, "rm -rf /tmp/kiwi-test") || !strings.HasPrefix(q.text, "⚠️") {
		t.Errorf("question should show the command and flag it as dangerous:\n%s", q.text)
	}
	if notice := press(a, q, true); notice != "Permitido" {
		t.Errorf("notice = %q", notice)
	}
	if err := verdict(t, done); err != nil {
		t.Fatalf("Ask = %v, want nil after Allow", err)
	}
	if e := lastEdit(conv); !strings.Contains(e, "✅ Permitido") {
		t.Errorf("the question should be edited to its outcome (which drops the buttons), got:\n%s", e)
	}
}

func TestApproveDenies(t *testing.T) {
	a, conv := &Approver{}, &fakeConv{}
	a.attach(conv)
	done := ask(context.Background(), a, rmrf)

	press(a, question(t, conv), false)
	if err := verdict(t, done); !errors.Is(err, permission.ErrDenied) {
		t.Fatalf("Ask = %v, want ErrDenied", err)
	}
	if e := lastEdit(conv); !strings.Contains(e, "🚫 Denegado") {
		t.Errorf("final message:\n%s", e)
	}
}

func TestApprovalTimesOut(t *testing.T) {
	a, conv := &Approver{Timeout: 50 * time.Millisecond}, &fakeConv{}
	a.attach(conv)
	done := ask(context.Background(), a, rmrf)

	if err := verdict(t, done); !errors.Is(err, ErrApprovalTimeout) {
		t.Fatalf("Ask = %v, want ErrApprovalTimeout", err)
	}
	if e := lastEdit(conv); !strings.Contains(e, "Sin respuesta") || !strings.Contains(e, "denegado") {
		t.Errorf("final message:\n%s", e)
	}
	// A late press finds nothing to resolve.
	if notice := press(a, question(t, conv), true); !strings.Contains(notice, "ya no está pendiente") {
		t.Errorf("a press after the timeout returned %q", notice)
	}
}

func TestApprovalCancelledWithTheTurn(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	a, conv := &Approver{}, &fakeConv{}
	a.attach(conv)
	done := ask(ctx, a, rmrf)

	question(t, conv)
	cancel()
	if err := verdict(t, done); !errors.Is(err, context.Canceled) {
		t.Fatalf("Ask = %v, want context.Canceled", err)
	}
	if e := lastEdit(conv); !strings.Contains(e, "interrumpido") {
		t.Errorf("the question should be closed even on shutdown:\n%s", e)
	}
}

func TestNoTurnNoQuestion(t *testing.T) {
	done := ask(context.Background(), &Approver{}, rmrf)
	if err := verdict(t, done); !errors.Is(err, permission.ErrNoUI) {
		t.Fatalf("Ask with no chat attached = %v, want ErrNoUI", err)
	}
}

func TestSafeCommandsDoNotAsk(t *testing.T) {
	a, conv := &Approver{}, &fakeConv{}
	a.attach(conv)
	done := ask(context.Background(), a, permission.Action{Name: permission.ActionBash, Detail: "go test ./..."})
	if err := verdict(t, done); err != nil {
		t.Fatalf("a safe command in work mode = %v, want it allowed without asking", err)
	}
	if ops := conv.snapshot(); len(ops) != 0 {
		t.Errorf("asked about a safe command: %+v", ops)
	}
}

func TestConcurrentQuestionsResolveIndependently(t *testing.T) {
	a, conv := &Approver{}, &fakeConv{}
	a.attach(conv)
	first := ask(context.Background(), a, rmrf)
	q1 := question(t, conv)
	second := ask(context.Background(), a, permission.Action{Name: permission.ActionBash, Detail: "sudo reboot", Dangerous: true})

	var q2 convOp
	waitFor(t, "the second question", func() bool {
		for _, op := range conv.snapshot() {
			if !op.edit && len(op.buttons) > 0 && op.id != q1.id {
				q2 = op
				return true
			}
		}
		return false
	})

	press(a, q2, false)
	press(a, q1, true)
	if err := verdict(t, first); err != nil {
		t.Errorf("first = %v, want allowed", err)
	}
	if err := verdict(t, second); !errors.Is(err, permission.ErrDenied) {
		t.Errorf("second = %v, want denied", err)
	}
}

func TestDiffIsShownTrimmed(t *testing.T) {
	a, conv := &Approver{}, &fakeConv{}
	a.attach(conv)
	action := permission.Action{Name: permission.ActionWrite, Detail: "main.go", Diff: strings.Repeat("+línea\n", 2000)}
	// write_file only asks in ask mode.
	broker := permission.NewBroker(permission.ModeAsk, a)
	done := make(chan error, 1)
	go func() { done <- broker.Ask(context.Background(), action) }()

	q := question(t, conv)
	if !strings.Contains(q.text, "+línea") || !strings.Contains(q.text, "diff recortado") {
		t.Errorf("want the diff, trimmed:\n%.200s…", q.text)
	}
	if n := utf16Len(q.text); n > maxMessage {
		t.Errorf("question is %d units, over the limit", n)
	}
	press(a, q, false)
	verdict(t, done)
}

func TestParseCallbackData(t *testing.T) {
	for _, tt := range []struct {
		data  string
		id    int
		allow bool
		ok    bool
	}{
		{callbackData(7, true), 7, true, true},
		{callbackData(7, false), 7, false, true},
		{"perm:7:maybe", 0, false, false},
		{"perm:x:allow", 0, false, false},
		{"other:7:allow", 0, false, false},
		{"", 0, false, false},
	} {
		id, allow, ok := parseCallbackData(tt.data)
		if id != tt.id || allow != tt.allow || ok != tt.ok {
			t.Errorf("parseCallbackData(%q) = %d, %v, %v", tt.data, id, allow, ok)
		}
	}
	if n := len(callbackData(1<<30, false)); n > 64 {
		t.Errorf("callback data is %d bytes; Telegram allows 64", n)
	}
}

// End to end through the Bot: only a press from an allowed user counts.
func TestStrangerCannotPressTheButton(t *testing.T) {
	api := newFakeAPI(t)
	a := &Approver{}
	b := testBot(api, 42)
	b.OnCallback = a.HandleCallback
	a.attach(botChat{b, 42})

	stop := runBot(t, b)
	defer stop()
	done := ask(context.Background(), a, rmrf)

	var allowData string
	waitFor(t, "the question to be sent", func() bool { return len(api.sentMessages()) == 1 })
	allowData = callbackData(1, true)

	api.queue(Update{UpdateID: 1, CallbackQuery: &CallbackQuery{ID: "evil", From: User{ID: 666}, Data: allowData}})
	waitFor(t, "the stranger's press to be consumed", func() bool {
		api.mu.Lock()
		defer api.mu.Unlock()
		return len(api.offsets) > 0 && api.offsets[len(api.offsets)-1] == 2
	})
	select {
	case err := <-done:
		t.Fatalf("a stranger's press resolved the question: %v", err)
	default:
	}

	api.queue(Update{UpdateID: 2, CallbackQuery: &CallbackQuery{ID: "mine", From: User{ID: 42}, Data: allowData}})
	if err := verdict(t, done); err != nil {
		t.Fatalf("the allowed user's press = %v, want allowed", err)
	}

	api.mu.Lock()
	defer api.mu.Unlock()
	for _, id := range api.answered {
		if id == "evil" {
			t.Error("the stranger's press was answered; it should get nothing")
		}
	}
}
