package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

const testToken = "123456:SECRET-token"

// Echo replies with the message it got: a handler with no agent behind it.
func Echo(_ context.Context, msg Message) string { return msg.Text }

// fakeAPI plays the Telegram Bot API: it serves queued updates to getUpdates
// and records every sendMessage.
type fakeAPI struct {
	t *testing.T

	mu        sync.Mutex
	updates   []Update
	sent      []sentMessage
	failNext  int // getUpdates calls to answer with HTTP 502 first
	limitNext int // sendMessage calls to answer with 429 first
	polls     int
	offsets   []int64

	srv *httptest.Server
}

type sentMessage struct {
	ChatID int64  `json:"chat_id"`
	Text   string `json:"text"`
}

func newFakeAPI(t *testing.T) *fakeAPI {
	f := &fakeAPI{t: t}
	f.srv = httptest.NewServer(http.HandlerFunc(f.serve))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeAPI) client() *Client {
	return newClient(f.srv.URL+"/bot"+testToken, f.srv.Client())
}

func (f *fakeAPI) queue(us ...Update) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.updates = append(f.updates, us...)
}

func (f *fakeAPI) sentMessages() []sentMessage {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]sentMessage(nil), f.sent...)
}

func (f *fakeAPI) reply(w http.ResponseWriter, result any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"ok": true, "result": result})
}

func (f *fakeAPI) serve(w http.ResponseWriter, r *http.Request) {
	prefix := "/bot" + testToken + "/"
	if !strings.HasPrefix(r.URL.Path, prefix) {
		w.WriteHeader(http.StatusUnauthorized)
		json.NewEncoder(w).Encode(map[string]any{"ok": false, "error_code": 401, "description": "Unauthorized"})
		return
	}

	switch method := strings.TrimPrefix(r.URL.Path, prefix); method {
	case "getMe":
		f.reply(w, User{ID: 1, Username: "kiwi_test_bot"})

	case "getUpdates":
		var p struct {
			Offset int64 `json:"offset"`
		}
		json.NewDecoder(r.Body).Decode(&p)

		f.mu.Lock()
		f.polls++
		f.offsets = append(f.offsets, p.Offset)
		if f.failNext > 0 {
			f.failNext--
			f.mu.Unlock()
			w.WriteHeader(http.StatusBadGateway)
			w.Write([]byte("<html>bad gateway</html>"))
			return
		}
		var out []Update
		for _, u := range f.updates {
			if u.UpdateID >= p.Offset {
				out = append(out, u)
			}
		}
		f.mu.Unlock()

		if len(out) == 0 {
			// A long poll with nothing to deliver: hold the request open
			// until the client gives up, like Telegram does.
			select {
			case <-r.Context().Done():
				return
			case <-time.After(50 * time.Millisecond):
			}
		}
		f.reply(w, out)

	case "sendMessage":
		var m sentMessage
		json.NewDecoder(r.Body).Decode(&m)
		f.mu.Lock()
		if f.limitNext > 0 {
			f.limitNext--
			f.mu.Unlock()
			w.WriteHeader(http.StatusTooManyRequests)
			json.NewEncoder(w).Encode(map[string]any{
				"ok": false, "error_code": 429, "description": "Too Many Requests: retry after 1",
				"parameters": map[string]any{"retry_after": 1},
			})
			return
		}
		f.sent = append(f.sent, m)
		f.mu.Unlock()
		f.reply(w, map[string]any{"message_id": 1})

	default:
		f.t.Errorf("unexpected Bot API method %q", method)
		w.WriteHeader(http.StatusNotFound)
	}
}

func textUpdate(id, userID int64, chatType, text string) Update {
	return Update{UpdateID: id, Message: &Message{
		MessageID: id,
		From:      &User{ID: userID},
		Chat:      Chat{ID: userID, Type: chatType},
		Text:      text,
	}}
}

// runBot starts b in the background and returns a function that stops it and
// waits for Run to return.
func runBot(t *testing.T, b *Bot) (stop func() error) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- b.Run(ctx) }()
	return func() error {
		cancel()
		select {
		case err := <-done:
			return err
		case <-time.After(2 * time.Second):
			t.Fatal("Run did not return after its context was cancelled")
			return nil
		}
	}
}

// waitFor polls cond until it holds or a second passes.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(5 * time.Millisecond)
	}
}

func testBot(api *fakeAPI, allowed ...int64) *Bot {
	b := NewBot(api.client(), allowed, Echo)
	b.minBackoff = time.Millisecond
	b.maxBackoff = 5 * time.Millisecond
	return b
}

func TestEchoesAllowedUser(t *testing.T) {
	api := newFakeAPI(t)
	api.queue(textUpdate(10, 42, "private", "hola kiwi"))

	stop := runBot(t, testBot(api, 42))
	waitFor(t, "the echo", func() bool { return len(api.sentMessages()) == 1 })
	if err := stop(); err != nil {
		t.Fatalf("Run returned %v, want nil on a clean stop", err)
	}

	got := api.sentMessages()[0]
	if got.ChatID != 42 || got.Text != "hola kiwi" {
		t.Errorf("sent %+v, want the text echoed back to chat 42", got)
	}
}

func TestIgnoresStrangersSilently(t *testing.T) {
	api := newFakeAPI(t)
	api.queue(
		textUpdate(1, 666, "private", "rm -rf /"), // not on the list
		textUpdate(2, 42, "group", "hola grupo"),  // allowed user, but not a private chat
		Update{UpdateID: 3},                       // not a message at all
		textUpdate(4, 42, "private", "solo a mí"),
	)

	var logs []string
	var logMu sync.Mutex
	b := testBot(api, 42)
	b.Log = func(s string) { logMu.Lock(); logs = append(logs, s); logMu.Unlock() }

	stop := runBot(t, b)
	waitFor(t, "the reply to the allowed user", func() bool { return len(api.sentMessages()) >= 1 })
	// The poll that acknowledges update 4 proves every earlier update was
	// already handled, so nothing else can still be on its way.
	waitFor(t, "the queue to be acknowledged", func() bool {
		api.mu.Lock()
		defer api.mu.Unlock()
		return len(api.offsets) > 0 && api.offsets[len(api.offsets)-1] == 5
	})
	stop()

	sent := api.sentMessages()
	if len(sent) != 1 || sent[0].ChatID != 42 || sent[0].Text != "solo a mí" {
		t.Fatalf("sent %+v, want only the reply to user 42's private message", sent)
	}

	logMu.Lock()
	defer logMu.Unlock()
	if joined := strings.Join(logs, "\n"); !strings.Contains(joined, "666") {
		t.Errorf("the stranger's ID should be logged locally so the operator can see it; logs:\n%s", joined)
	}
}

func TestAdvancesOffset(t *testing.T) {
	api := newFakeAPI(t)
	api.queue(textUpdate(7, 42, "private", "uno"), textUpdate(8, 42, "private", "dos"))

	stop := runBot(t, testBot(api, 42))
	waitFor(t, "both echoes", func() bool { return len(api.sentMessages()) == 2 })
	waitFor(t, "a second poll", func() bool {
		api.mu.Lock()
		defer api.mu.Unlock()
		return len(api.offsets) >= 2
	})
	stop()

	api.mu.Lock()
	defer api.mu.Unlock()
	if api.offsets[0] != 0 || api.offsets[1] != 9 {
		t.Errorf("offsets = %v, want 0 first and then 9 (last update_id + 1)", api.offsets)
	}
	// Each update is answered once, never redelivered.
	if n := len(api.sent); n != 2 {
		t.Errorf("sent %d messages, want 2", n)
	}
}

func TestRetriesAfterNetworkErrors(t *testing.T) {
	api := newFakeAPI(t)
	api.failNext = 3
	api.queue(textUpdate(1, 42, "private", "sigues ahí?"))

	stop := runBot(t, testBot(api, 42))
	waitFor(t, "the echo after the errors", func() bool { return len(api.sentMessages()) == 1 })
	if err := stop(); err != nil {
		t.Fatalf("Run returned %v; transient errors should be retried, not returned", err)
	}
}

func TestStopsOnRejectedToken(t *testing.T) {
	api := newFakeAPI(t)
	c := newClient(api.srv.URL+"/botWRONG", api.srv.Client())
	b := NewBot(c, []int64{42}, Echo)

	done := make(chan error, 1)
	go func() { done <- b.Run(context.Background()) }()

	select {
	case err := <-done:
		if err == nil || !strings.Contains(err.Error(), "rejected") {
			t.Fatalf("Run returned %v, want a rejected-token error", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run kept retrying a token Telegram rejected")
	}
}

func TestCancelDuringLongPoll(t *testing.T) {
	// A server that never answers, like a long poll with nothing to say.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// The server only notices a client hanging up once the body is read.
		io.Copy(io.Discard, r.Body)
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}))
	t.Cleanup(srv.Close)

	b := NewBot(newClient(srv.URL+"/bot"+testToken, srv.Client()), []int64{42}, Echo)
	stop := runBot(t, b)
	time.Sleep(20 * time.Millisecond) // let the poll start
	if err := stop(); err != nil {
		t.Fatalf("Run returned %v, want nil when cancelled mid-poll", err)
	}
}

func TestErrorsNeverContainTheToken(t *testing.T) {
	// Nothing listens here, so the request fails at the transport level,
	// which is where net/http would print the full URL.
	c := newClient("http://127.0.0.1:1/bot"+testToken, nil)
	_, err := c.GetMe(context.Background())
	if err == nil {
		t.Fatal("expected a connection error")
	}
	if strings.Contains(err.Error(), "SECRET") {
		t.Errorf("error leaks the bot token: %v", err)
	}
}

func TestGetMe(t *testing.T) {
	api := newFakeAPI(t)
	me, err := api.client().GetMe(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if me.Username != "kiwi_test_bot" {
		t.Errorf("username = %q", me.Username)
	}
}

func TestParseUserIDs(t *testing.T) {
	tests := []struct {
		in      string
		want    []int64
		wantErr bool
	}{
		{"42", []int64{42}, false},
		{"42,7", []int64{42, 7}, false},
		{" 42 , 7\n", []int64{42, 7}, false},
		{"", []int64{}, false},
		{"@oscar", nil, true},
		{"-5", nil, true},
	}
	for _, tt := range tests {
		got, err := ParseUserIDs(tt.in)
		if (err != nil) != tt.wantErr {
			t.Errorf("ParseUserIDs(%q) error = %v, wantErr %v", tt.in, err, tt.wantErr)
			continue
		}
		if tt.wantErr {
			continue
		}
		if len(got) != len(tt.want) {
			t.Errorf("ParseUserIDs(%q) = %v, want %v", tt.in, got, tt.want)
			continue
		}
		for i := range got {
			if got[i] != tt.want[i] {
				t.Errorf("ParseUserIDs(%q) = %v, want %v", tt.in, got, tt.want)
			}
		}
	}
}

func TestSlowHandlerDoesNotBlockPolling(t *testing.T) {
	api := newFakeAPI(t)
	release := make(chan struct{})
	var once sync.Once
	handler := func(ctx context.Context, m Message) string {
		if m.Text == "lento" {
			select {
			case <-release:
			case <-ctx.Done():
			}
			return "terminé"
		}
		once.Do(func() { close(release) })
		return "rápido"
	}

	b := NewBot(api.client(), []int64{42}, handler)
	api.queue(textUpdate(1, 42, "private", "lento"))
	stop := runBot(t, b)

	// Arrives while the first message is still being handled.
	time.Sleep(20 * time.Millisecond)
	api.queue(textUpdate(2, 42, "private", "otro"))

	waitFor(t, "both replies", func() bool { return len(api.sentMessages()) == 2 })
	stop()

	// "lento" can only finish because "otro" was handled while it was still
	// running; with sequential handling the test times out above instead.
	got := map[string]bool{}
	for _, m := range api.sentMessages() {
		got[m.Text] = true
	}
	if !got["rápido"] || !got["terminé"] {
		t.Errorf("replies = %+v, want both", api.sentMessages())
	}
}

func TestLongReplyArrivesInOrder(t *testing.T) {
	api := newFakeAPI(t)
	var long strings.Builder
	for i := range 300 {
		fmt.Fprintf(&long, "Línea %03d de una respuesta muy larga del agente.\n", i)
	}
	reply := long.String() // ~15.000 caracteres
	b := NewBot(api.client(), []int64{42}, func(context.Context, Message) string { return reply })
	api.queue(textUpdate(1, 42, "private", "cuéntamelo todo"))

	stop := runBot(t, b)
	waitFor(t, "every chunk", func() bool { return len(api.sentMessages()) >= 4 })
	stop()

	sent := api.sentMessages()
	var got []string
	for _, m := range sent {
		if n := utf16Len(m.Text); n > maxMessage {
			t.Errorf("sent a message of %d units, over Telegram's limit", n)
		}
		got = append(got, m.Text)
	}
	if strings.Join(got, "\n") != strings.TrimRight(reply, "\n") {
		t.Error("the chunks, joined in the order sent, are not the original answer")
	}
}

func TestWaitsOutRateLimit(t *testing.T) {
	api := newFakeAPI(t)
	api.limitNext = 1
	api.queue(textUpdate(1, 42, "private", "hola"))

	start := time.Now()
	stop := runBot(t, testBot(api, 42))
	deadline := time.Now().Add(3 * time.Second)
	for len(api.sentMessages()) == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	stop()

	if len(api.sentMessages()) != 1 {
		t.Fatal("the reply was dropped after a 429 instead of retried")
	}
	if waited := time.Since(start); waited < time.Second {
		t.Errorf("retried after %s; Telegram asked for 1s", waited)
	}
}
