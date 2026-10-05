// Package remote lets Kiwi be driven from outside the terminal. Today that is
// a Telegram bot (kiwi serve); it never imports internal/tui.
package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

// apiBase is the Telegram Bot API. The token is part of the path, which is
// why errors from this file never include the request URL.
const apiBase = "https://api.telegram.org/bot"

// Client is a minimal Telegram Bot API client. The API is JSON over HTTPS and
// Kiwi needs a handful of methods, so it is written here rather than pulled in
// as a dependency.
type Client struct {
	base string // apiBase + token, no trailing slash
	http *http.Client
}

// NewClient returns a client for the bot with the given token.
func NewClient(token string) *Client {
	return newClient(apiBase+token, nil)
}

func newClient(base string, hc *http.Client) *Client {
	if hc == nil {
		// No overall timeout: long polls are bounded per request instead, by
		// the context getUpdates builds.
		hc = &http.Client{}
	}
	return &Client{base: base, http: hc}
}

// User is the sender of a message.
type User struct {
	ID       int64  `json:"id"`
	Username string `json:"username,omitempty"`
}

// Chat is where a message was sent. Type is "private" for one-to-one chats.
type Chat struct {
	ID   int64  `json:"id"`
	Type string `json:"type"`
}

// Message is the subset of a Telegram message Kiwi reads.
type Message struct {
	MessageID int64  `json:"message_id"`
	From      *User  `json:"from,omitempty"`
	Chat      Chat   `json:"chat"`
	Text      string `json:"text,omitempty"`
}

// Update is one event from getUpdates. Only messages are requested.
type Update struct {
	UpdateID int64    `json:"update_id"`
	Message  *Message `json:"message,omitempty"`
}

// APIError is a request Telegram answered with ok=false.
type APIError struct {
	Method      string
	Code        int
	Description string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("telegram %s: %d %s", e.Method, e.Code, e.Description)
}

// call posts params as JSON to method and decodes the result into out.
func (c *Client) call(ctx context.Context, method string, params, out any) error {
	body, err := json.Marshal(params)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.base+"/"+method, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("telegram %s: building request failed", method)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		// *url.Error prints the full URL, and the URL holds the token.
		var uerr *url.Error
		if errors.As(err, &uerr) {
			err = uerr.Err
		}
		return fmt.Errorf("telegram %s: %w", method, err)
	}
	defer resp.Body.Close()

	var envelope struct {
		OK          bool            `json:"ok"`
		Result      json.RawMessage `json:"result"`
		ErrorCode   int             `json:"error_code"`
		Description string          `json:"description"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("telegram %s: decoding response (HTTP %d): %w", method, resp.StatusCode, err)
	}
	if !envelope.OK {
		code := envelope.ErrorCode
		if code == 0 {
			code = resp.StatusCode
		}
		return &APIError{Method: method, Code: code, Description: envelope.Description}
	}
	if out == nil {
		return nil
	}
	if err := json.Unmarshal(envelope.Result, out); err != nil {
		return fmt.Errorf("telegram %s: decoding result: %w", method, err)
	}
	return nil
}

// GetMe returns the bot's own user. It is the cheapest way to check a token.
func (c *Client) GetMe(ctx context.Context) (User, error) {
	var u User
	err := c.call(ctx, "getMe", struct{}{}, &u)
	return u, err
}

// GetUpdates long-polls for new messages, waiting up to timeout for one to
// arrive. offset is the last seen update_id + 1, which also acknowledges
// everything before it.
func (c *Client) GetUpdates(ctx context.Context, offset int64, timeout time.Duration) ([]Update, error) {
	// Give the server the full poll plus some slack before giving up on it.
	ctx, cancel := context.WithTimeout(ctx, timeout+15*time.Second)
	defer cancel()

	params := struct {
		Offset         int64    `json:"offset,omitempty"`
		Timeout        int      `json:"timeout"`
		AllowedUpdates []string `json:"allowed_updates"`
	}{offset, int(timeout / time.Second), []string{"message"}}

	var updates []Update
	err := c.call(ctx, "getUpdates", params, &updates)
	return updates, err
}

// SendMessage sends plain text to a chat.
func (c *Client) SendMessage(ctx context.Context, chatID int64, text string) error {
	params := struct {
		ChatID int64  `json:"chat_id"`
		Text   string `json:"text"`
	}{chatID, text}
	return c.call(ctx, "sendMessage", params, nil)
}
