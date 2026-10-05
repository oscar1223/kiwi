package remote

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Handler answers one message from an allowed user. An empty reply sends
// nothing. conv lets it post and edit messages in that chat while it works,
// before the reply.
type Handler func(ctx context.Context, msg Message, conv Conversation) string

// Conversation is the chat a Handler is answering.
type Conversation interface {
	// Send posts a message and returns its ID, for Edit. buttons, if any,
	// are shown under it, one row each.
	Send(ctx context.Context, text string, buttons ...[]Button) (int64, error)
	// Edit replaces the text of a message Send returned.
	Edit(ctx context.Context, messageID int64, text string) error
}

// botChat is a Conversation bound to one Telegram chat.
type botChat struct {
	b      *Bot
	chatID int64
}

func (c botChat) Send(ctx context.Context, text string, buttons ...[]Button) (int64, error) {
	return c.b.send(ctx, c.chatID, text, buttons...)
}

// CallbackHandler handles a button press from an allowed user and returns
// the short notice Telegram shows to whoever pressed it.
type CallbackHandler func(ctx context.Context, q CallbackQuery) string

func (c botChat) Edit(ctx context.Context, messageID int64, text string) error {
	return c.b.client.EditMessageText(ctx, c.chatID, messageID, text)
}

// Bot long-polls Telegram and hands messages from allowed users to a Handler.
//
// The bot calls Telegram and Telegram never calls the bot, so no port is
// opened: whatever machine runs it needs outbound HTTPS and nothing else.
type Bot struct {
	client  *Client
	allowed map[int64]bool
	handle  Handler

	// Log reports what the bot is doing, for the operator's terminal. Never
	// shown to Telegram users. May be nil.
	Log func(string)
	// OnCallback handles button presses. May be nil.
	OnCallback CallbackHandler

	pollTimeout time.Duration
	minBackoff  time.Duration
	maxBackoff  time.Duration
}

// NewBot returns a bot that only answers the given Telegram user IDs.
func NewBot(client *Client, allowed []int64, handle Handler) *Bot {
	set := make(map[int64]bool, len(allowed))
	for _, id := range allowed {
		set[id] = true
	}
	return &Bot{
		client:      client,
		allowed:     set,
		handle:      handle,
		pollTimeout: 30 * time.Second,
		minBackoff:  time.Second,
		maxBackoff:  time.Minute,
	}
}

func (b *Bot) logf(format string, args ...any) {
	if b.Log != nil {
		b.Log(fmt.Sprintf(format, args...))
	}
}

// Run polls until ctx is cancelled, which is a clean stop and returns nil.
//
// Network errors are expected on a connection that stays open for hours, so
// they are logged and retried with backoff rather than returned. Only a
// rejected token stops the bot: no amount of retrying fixes that.
//
// Each message is handled on its own goroutine, so polling carries on while a
// long turn runs and the handler can answer a second message with "busy".
// Run waits for those goroutines before returning.
func (b *Bot) Run(ctx context.Context) error {
	var (
		offset   int64
		inflight sync.WaitGroup
	)
	defer inflight.Wait()
	backoff := b.minBackoff

	for {
		updates, err := b.client.GetUpdates(ctx, offset, b.pollTimeout)
		if ctx.Err() != nil {
			return nil
		}
		if err != nil {
			var apiErr *APIError
			if errors.As(err, &apiErr) && (apiErr.Code == 401 || apiErr.Code == 404) {
				return fmt.Errorf("the bot token was rejected: %w", err)
			}
			b.logf("%v (retrying in %s)", err, backoff)
			select {
			case <-ctx.Done():
				return nil
			case <-time.After(backoff):
			}
			backoff = min(backoff*2, b.maxBackoff)
			continue
		}
		backoff = b.minBackoff

		for _, u := range updates {
			offset = u.UpdateID + 1
			inflight.Add(1)
			go func() {
				defer inflight.Done()
				b.dispatch(ctx, u)
			}()
		}
	}
}

// dispatch answers an update if it is a private text message from an allowed
// user. Everything else is dropped without a reply, so a stranger cannot even
// confirm the bot is running.
func (b *Bot) dispatch(ctx context.Context, u Update) {
	if q := u.CallbackQuery; q != nil {
		b.dispatchCallback(ctx, *q)
		return
	}
	msg := u.Message
	if msg == nil || msg.From == nil || msg.Text == "" {
		return
	}
	if !b.allowed[msg.From.ID] {
		// Logged locally so the operator can find their own ID on first run.
		b.logf("ignored a message from user %d (not in the allowed list)", msg.From.ID)
		return
	}
	if msg.Chat.Type != "private" {
		b.logf("ignored a message from user %d in a %s chat (only private chats are answered)", msg.From.ID, msg.Chat.Type)
		return
	}

	reply := b.handle(ctx, *msg, botChat{b, msg.Chat.ID})
	if reply == "" || ctx.Err() != nil {
		return
	}
	b.Reply(ctx, msg.Chat.ID, reply)
}

// Conversation returns the Conversation for a chat, for work that is not an
// answer to a message, such as a scheduled task.
func (b *Bot) Conversation(chatID int64) Conversation { return botChat{b, chatID} }

// Reply sends text to a chat, split into as many messages as it takes.
func (b *Bot) Reply(ctx context.Context, chatID int64, text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	chunks := splitMessage(text, maxMessage)
	for i, chunk := range chunks {
		if _, err := b.send(ctx, chatID, chunk); err != nil {
			if ctx.Err() == nil {
				b.logf("could not reply in chat %d (message %d of %d): %v", chatID, i+1, len(chunks), err)
			}
			return
		}
	}
}

// dispatchCallback hands a button press to OnCallback. The allowed list is
// checked again here: a button is a message like any other, and what matters
// is who pressed it, not who the message was sent to.
func (b *Bot) dispatchCallback(ctx context.Context, q CallbackQuery) {
	if !b.allowed[q.From.ID] {
		b.logf("ignored a button press from user %d (not in the allowed list)", q.From.ID)
		return
	}
	notice := ""
	if b.OnCallback != nil {
		notice = b.OnCallback(ctx, q)
	}
	// Always answered, or the button keeps spinning on the phone.
	if err := b.client.AnswerCallbackQuery(ctx, q.ID, notice); err != nil && ctx.Err() == nil {
		b.logf("could not answer a button press: %v", err)
	}
}

// send delivers one message, waiting out Telegram's rate limit when it asks
// to: a long answer is several messages in a row, which is what trips it.
func (b *Bot) send(ctx context.Context, chatID int64, text string, buttons ...[]Button) (int64, error) {
	for attempt := 0; ; attempt++ {
		id, err := b.client.SendMessage(ctx, chatID, text, buttons...)
		var apiErr *APIError
		if !errors.As(err, &apiErr) || apiErr.RetryAfter <= 0 || attempt == 3 {
			return id, err
		}
		select {
		case <-ctx.Done():
			return 0, ctx.Err()
		case <-time.After(apiErr.RetryAfter):
		}
	}
}

// ParseUserIDs parses a comma- or space-separated list of Telegram user IDs.
func ParseUserIDs(s string) ([]int64, error) {
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == '\t' || r == '\n' })
	ids := make([]int64, 0, len(fields))
	for _, f := range fields {
		id, err := strconv.ParseInt(f, 10, 64)
		if err != nil || id <= 0 {
			return nil, fmt.Errorf("%q is not a Telegram user ID", f)
		}
		ids = append(ids, id)
	}
	return ids, nil
}
