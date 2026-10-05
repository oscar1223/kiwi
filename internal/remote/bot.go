package remote

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Handler answers one message from an allowed user. An empty reply sends
// nothing.
type Handler func(ctx context.Context, msg Message) string

// Echo is the handler for kiwi serve before it has an agent: it replies with
// the message it got.
func Echo(_ context.Context, msg Message) string { return msg.Text }

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
func (b *Bot) Run(ctx context.Context) error {
	var offset int64
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
			b.dispatch(ctx, u)
		}
	}
}

// dispatch answers an update if it is a private text message from an allowed
// user. Everything else is dropped without a reply, so a stranger cannot even
// confirm the bot is running.
func (b *Bot) dispatch(ctx context.Context, u Update) {
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

	reply := b.handle(ctx, *msg)
	if reply == "" {
		return
	}
	if err := b.client.SendMessage(ctx, msg.Chat.ID, reply); err != nil && ctx.Err() == nil {
		b.logf("could not reply to user %d: %v", msg.From.ID, err)
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
