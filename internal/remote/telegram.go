// Package remote lets Kiwi be driven from outside the terminal. Today that is
// a Telegram bot (kiwi serve); it never imports internal/tui.
package remote

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// apiBase is the Telegram Bot API. The token is part of the path, which is
// why errors from this file never include the request URL.
const apiBase = "https://api.telegram.org/bot"

// Client is a minimal Telegram Bot API client. The API is JSON over HTTPS and
// Kiwi needs a handful of methods, so it is written here rather than pulled in
// as a dependency.
type Client struct {
	base     string // apiBase + token, no trailing slash
	fileBase string // where files are downloaded from: .../file/bot<token>
	http     *http.Client
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
	// Files live under /file/bot<token>/ rather than /bot<token>/.
	fileBase := base
	if i := strings.LastIndex(base, "/bot"); i >= 0 {
		fileBase = base[:i] + "/file" + base[i:]
	}
	return &Client{base: base, fileBase: fileBase, http: hc}
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
	// Caption is the text sent along with a photo, video, audio or document.
	Caption string `json:"caption,omitempty"`

	// Photo comes in several sizes, smallest first.
	Photo     []File `json:"photo,omitempty"`
	Voice     *File  `json:"voice,omitempty"`
	Audio     *File  `json:"audio,omitempty"`
	Video     *File  `json:"video,omitempty"`
	VideoNote *File  `json:"video_note,omitempty"`
	Document  *File  `json:"document,omitempty"`

	// Attachment is the media the bot downloaded for this message, set
	// before the Handler sees it. Not part of the Telegram API.
	Attachment *Media `json:"-"`
}

// File is a file attached to a message: one photo size, a voice note, an
// audio, a video or a document. Telegram leaves out what does not apply.
type File struct {
	FileID   string `json:"file_id"`
	FileSize int64  `json:"file_size,omitempty"`
	MIMEType string `json:"mime_type,omitempty"`
	FileName string `json:"file_name,omitempty"`
	Duration int    `json:"duration,omitempty"`
	Width    int    `json:"width,omitempty"`
	Height   int    `json:"height,omitempty"`
}

// CallbackQuery is a press on an inline keyboard button.
type CallbackQuery struct {
	ID      string   `json:"id"`
	From    User     `json:"from"`
	Message *Message `json:"message,omitempty"`
	Data    string   `json:"data,omitempty"`
}

// Update is one event from getUpdates: a message or a button press.
type Update struct {
	UpdateID      int64          `json:"update_id"`
	Message       *Message       `json:"message,omitempty"`
	CallbackQuery *CallbackQuery `json:"callback_query,omitempty"`
}

// Button is one inline keyboard button. Data comes back in the CallbackQuery
// when it is pressed; Telegram allows at most 64 bytes.
type Button struct {
	Text string `json:"text"`
	Data string `json:"callback_data"`
}

type inlineKeyboard struct {
	Rows [][]Button `json:"inline_keyboard"`
}

// APIError is a request Telegram answered with ok=false.
type APIError struct {
	Method      string
	Code        int
	Description string
	// RetryAfter is how long to wait before trying again, when Telegram
	// rate-limited the request (429).
	RetryAfter time.Duration
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
		Parameters  struct {
			RetryAfter int `json:"retry_after"`
		} `json:"parameters"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&envelope); err != nil {
		return fmt.Errorf("telegram %s: decoding response (HTTP %d): %w", method, resp.StatusCode, err)
	}
	if !envelope.OK {
		code := envelope.ErrorCode
		if code == 0 {
			code = resp.StatusCode
		}
		return &APIError{
			Method:      method,
			Code:        code,
			Description: envelope.Description,
			RetryAfter:  time.Duration(envelope.Parameters.RetryAfter) * time.Second,
		}
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
	}{offset, int(timeout / time.Second), []string{"message", "callback_query"}}

	var updates []Update
	err := c.call(ctx, "getUpdates", params, &updates)
	return updates, err
}

// GetFile returns the path to download a file from, given its file_id. The
// path is valid for at least an hour.
func (c *Client) GetFile(ctx context.Context, fileID string) (string, error) {
	params := struct {
		FileID string `json:"file_id"`
	}{fileID}
	var f struct {
		FilePath string `json:"file_path"`
	}
	if err := c.call(ctx, "getFile", params, &f); err != nil {
		return "", err
	}
	if f.FilePath == "" {
		return "", errors.New("telegram getFile: no file_path (the file may be too big for a bot)")
	}
	return f.FilePath, nil
}

// Download writes the file at filePath (from GetFile) to w, refusing to read
// more than limit bytes.
func (c *Client) Download(ctx context.Context, filePath string, w io.Writer, limit int64) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.fileBase+"/"+filePath, nil)
	if err != nil {
		return errors.New("telegram download: building request failed")
	}
	resp, err := c.http.Do(req)
	if err != nil {
		// Like call: the URL holds the token, so it never reaches the error.
		var uerr *url.Error
		if errors.As(err, &uerr) {
			err = uerr.Err
		}
		return fmt.Errorf("telegram download: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("telegram download: HTTP %d", resp.StatusCode)
	}
	n, err := io.Copy(w, io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return fmt.Errorf("telegram download: %w", err)
	}
	if n > limit {
		return fmt.Errorf("telegram download: the file is bigger than %d MB", limit>>20)
	}
	return nil
}

// SendMessage sends plain text to a chat and returns the new message's ID.
// buttons, if any, are shown under it as an inline keyboard, one row each.
func (c *Client) SendMessage(ctx context.Context, chatID int64, text string, buttons ...[]Button) (int64, error) {
	params := struct {
		ChatID      int64           `json:"chat_id"`
		Text        string          `json:"text"`
		ReplyMarkup *inlineKeyboard `json:"reply_markup,omitempty"`
	}{ChatID: chatID, Text: text}
	if len(buttons) > 0 {
		params.ReplyMarkup = &inlineKeyboard{Rows: buttons}
	}
	var sent Message
	err := c.call(ctx, "sendMessage", params, &sent)
	return sent.MessageID, err
}

// AnswerCallbackQuery acknowledges a button press, so the client stops
// showing it as pending. text, if set, is shown briefly to whoever pressed.
func (c *Client) AnswerCallbackQuery(ctx context.Context, id, text string) error {
	params := struct {
		ID   string `json:"callback_query_id"`
		Text string `json:"text,omitempty"`
	}{id, text}
	return c.call(ctx, "answerCallbackQuery", params, nil)
}

// EditMessageText replaces the text of a message the bot sent. Any inline
// keyboard the message had is removed.
func (c *Client) EditMessageText(ctx context.Context, chatID, messageID int64, text string) error {
	params := struct {
		ChatID    int64  `json:"chat_id"`
		MessageID int64  `json:"message_id"`
		Text      string `json:"text"`
	}{chatID, messageID, text}
	err := c.call(ctx, "editMessageText", params, nil)
	// Editing to the same text is an error to Telegram and a no-op to us.
	var apiErr *APIError
	if errors.As(err, &apiErr) && strings.Contains(apiErr.Description, "message is not modified") {
		return nil
	}
	return err
}
