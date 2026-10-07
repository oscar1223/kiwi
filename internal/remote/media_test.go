package remote

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

func mediaUpdate(id, userID int64, m Message) Update {
	m.MessageID = id
	m.From = &User{ID: userID}
	m.Chat = Chat{ID: userID, Type: "private"}
	return Update{UpdateID: id, Message: &m}
}

// captureBot is a test bot whose handler records the messages it gets.
func captureBot(api *fakeAPI, inbox string) (*Bot, func() []Message) {
	var (
		mu  sync.Mutex
		got []Message
	)
	b := testBot(api, 42)
	b.InboxDir = inbox
	b.handle = func(_ context.Context, msg Message, _ Conversation) string {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, msg)
		return "vale"
	}
	return b, func() []Message {
		mu.Lock()
		defer mu.Unlock()
		return append([]Message(nil), got...)
	}
}

func TestMediaPicksTheBiggestPhoto(t *testing.T) {
	m := Message{Photo: []File{
		{FileID: "small", Width: 90, Height: 60},
		{FileID: "big", Width: 1280, Height: 960},
		{FileID: "medium", Width: 320, Height: 240},
	}}
	if got := m.Media(); got == nil || got.Kind != MediaPhoto || got.File.FileID != "big" {
		t.Errorf("Media() = %+v, want the 1280x960 photo", got)
	}
	if got := (Message{VideoNote: &File{FileID: "v"}}).Media(); got == nil || got.Kind != MediaVideo {
		t.Errorf("a round video note should be a video, got %+v", got)
	}
	if got := (Message{Text: "hola"}).Media(); got != nil {
		t.Errorf("a text message has no media, got %+v", got)
	}
}

func TestMediaClass(t *testing.T) {
	cases := []struct {
		m    Media
		want mediaClass
	}{
		{Media{Kind: MediaPhoto}, classImage},
		{Media{Kind: MediaVoice}, classAudio},
		{Media{Kind: MediaVideo}, classVideo},
		{Media{Kind: MediaDocument, File: File{MIMEType: "image/png", FileName: "captura.png"}}, classImage},
		{Media{Kind: MediaDocument, File: File{MIMEType: "application/pdf", FileName: "factura.pdf"}}, classPDF},
		{Media{Kind: MediaDocument, File: File{FileName: "notas.md"}}, classText},
		{Media{Kind: MediaDocument, File: File{MIMEType: "text/plain", FileName: "log"}}, classText},
		{Media{Kind: MediaDocument, File: File{MIMEType: "application/zip", FileName: "a.zip"}}, classOther},
	}
	for _, c := range cases {
		if got := c.m.class(); got != c.want {
			t.Errorf("class(%+v) = %s, want %s", c.m, got, c.want)
		}
	}
}

func TestDownloadsMediaBeforeTheHandler(t *testing.T) {
	api := newFakeAPI(t)
	api.addFile("voz-1", []byte("OggS fake opus"))
	inbox := filepath.Join(t.TempDir(), "inbox")
	b, got := captureBot(api, inbox)
	api.queue(mediaUpdate(10, 42, Message{Voice: &File{FileID: "voz-1", MIMEType: "audio/ogg", Duration: 3}}))

	stop := runBot(t, b)
	waitFor(t, "the handler", func() bool { return len(got()) == 1 })
	stop()

	m := got()[0].Attachment
	if m == nil || m.Path == "" {
		t.Fatalf("the handler got no downloaded attachment: %+v", got()[0])
	}
	if !strings.HasPrefix(m.Path, inbox) || filepath.Ext(m.Path) != ".ogg" {
		t.Errorf("saved to %s, want an .ogg file in %s", m.Path, inbox)
	}
	data, err := os.ReadFile(m.Path)
	if err != nil || string(data) != "OggS fake opus" {
		t.Errorf("saved file = %q, %v", data, err)
	}
	if fi, err := os.Stat(m.Path); err != nil || fi.Mode().Perm() != 0o600 {
		t.Errorf("file mode = %v, %v; want 0600", fi.Mode().Perm(), err)
	}
	if fi, err := os.Stat(inbox); err != nil || fi.Mode().Perm() != 0o700 {
		t.Errorf("inbox mode = %v, %v; want 0700", fi.Mode().Perm(), err)
	}
}

func TestCaptionAloneReachesTheHandler(t *testing.T) {
	api := newFakeAPI(t)
	api.addFile("p", []byte("jpeg"))
	b, got := captureBot(api, t.TempDir())
	api.queue(mediaUpdate(10, 42, Message{Caption: "¿qué error es?", Photo: []File{{FileID: "p", Width: 10, Height: 10}}}))

	stop := runBot(t, b)
	waitFor(t, "the handler", func() bool { return len(got()) == 1 })
	stop()

	if m := got()[0]; m.Caption != "¿qué error es?" || m.Attachment == nil {
		t.Errorf("handler got %+v, want the caption and the photo", m)
	}
}

func TestRefusesFilesTooBigForABot(t *testing.T) {
	api := newFakeAPI(t)
	b, got := captureBot(api, t.TempDir())
	api.queue(mediaUpdate(10, 42, Message{Video: &File{FileID: "v", FileSize: MaxDownload + 1}}))

	stop := runBot(t, b)
	waitFor(t, "the refusal", func() bool { return len(api.sentMessages()) == 1 })
	stop()

	if msg := api.sentMessages()[0].Text; !strings.Contains(msg, "20 MB") {
		t.Errorf("replied %q, want it to explain the 20 MB limit", msg)
	}
	api.mu.Lock()
	fetched := api.getFiles
	api.mu.Unlock()
	if len(got()) != 0 || fetched != 0 {
		t.Errorf("a file over the limit reached the handler (%d) or was fetched (%d)", len(got()), fetched)
	}
}

func TestUnsupportedMessagesGetAReplyOnlyFromAllowedUsers(t *testing.T) {
	api := newFakeAPI(t)
	b, got := captureBot(api, t.TempDir())
	// A sticker or a location: nothing Kiwi reads.
	api.queue(mediaUpdate(10, 42, Message{}), mediaUpdate(11, 7, Message{}))

	stop := runBot(t, b)
	waitFor(t, "the reply", func() bool { return len(api.sentMessages()) == 1 })
	stop()

	if sent := api.sentMessages(); len(sent) != 1 || sent[0].ChatID != 42 || sent[0].Text != Unsupported {
		t.Errorf("sent %+v, want one Unsupported reply to the allowed user only", sent)
	}
	if len(got()) != 0 {
		t.Errorf("the handler got %d messages, want none", len(got()))
	}
}

func TestDownloadErrorsNeverContainTheToken(t *testing.T) {
	c := newClient("http://127.0.0.1:1/bot"+testToken, nil)
	err := c.Download(context.Background(), "files/x", &strings.Builder{}, MaxDownload)
	if err == nil {
		t.Fatal("expected a connection error")
	}
	if strings.Contains(err.Error(), "SECRET") {
		t.Errorf("error leaks the bot token: %v", err)
	}
}

func TestDownloadStopsAtTheLimit(t *testing.T) {
	api := newFakeAPI(t)
	api.addFile("big", []byte(strings.Repeat("x", 100)))
	var out strings.Builder
	err := api.client().Download(context.Background(), "files/big", &out, 10)
	if err == nil {
		t.Fatal("a file over the limit should fail")
	}
	if out.Len() > 11 {
		t.Errorf("read %d bytes, want it to stop right after the limit", out.Len())
	}
}
