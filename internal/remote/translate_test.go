package remote

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fakeOpenRouter answers chat completions and keeps the last request body.
type fakeOpenRouter struct {
	srv    *httptest.Server
	body   map[string]any
	auth   string
	status int
	answer string
}

func newFakeOpenRouter(t *testing.T) *fakeOpenRouter {
	f := &fakeOpenRouter{status: http.StatusOK, answer: `{"choices":[{"message":{"content":"  una captura con un panic  "}}]}`}
	f.srv = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("translator posted to %s", r.URL.Path)
		}
		f.auth = r.Header.Get("Authorization")
		json.NewDecoder(r.Body).Decode(&f.body)
		w.WriteHeader(f.status)
		w.Write([]byte(f.answer))
	}))
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeOpenRouter) translator() *OpenRouterTranslator {
	return &OpenRouterTranslator{APIKey: "sk-test", BaseURL: f.srv.URL, Client: f.srv.Client()}
}

// part returns the content part that carries the file in the last request.
func (f *fakeOpenRouter) part(t *testing.T) map[string]any {
	t.Helper()
	msgs, _ := f.body["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("request has %d messages, want 1: %v", len(msgs), f.body)
	}
	content, _ := msgs[0].(map[string]any)["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("content has %d parts, want the prompt and the file", len(content))
	}
	return content[1].(map[string]any)
}

func mediaFile(t *testing.T, name, data string, m Media) *Media {
	t.Helper()
	m.Path = filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(m.Path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	return &m
}

func TestTranslatorSendsEachKindAsOpenRouterTakesIt(t *testing.T) {
	b64 := func(s string) string { return base64.StdEncoding.EncodeToString([]byte(s)) }
	cases := []struct {
		name  string
		media *Media
		check func(t *testing.T, part map[string]any)
	}{
		{"photo", mediaFile(t, "a.jpg", "JPEG", Media{Kind: MediaPhoto}), func(t *testing.T, p map[string]any) {
			if p["type"] != "image_url" || p["image_url"].(map[string]any)["url"] != "data:image/jpeg;base64,"+b64("JPEG") {
				t.Errorf("photo part = %v", p)
			}
		}},
		{"voice", mediaFile(t, "v.ogg", "OGG", Media{Kind: MediaVoice, File: File{MIMEType: "audio/ogg"}}), func(t *testing.T, p map[string]any) {
			a := p["input_audio"].(map[string]any)
			if p["type"] != "input_audio" || a["format"] != "ogg" || a["data"] != b64("OGG") {
				t.Errorf("voice part = %v", p)
			}
		}},
		{"m4a", mediaFile(t, "s.m4a", "M4A", Media{Kind: MediaAudio, File: File{MIMEType: "audio/mp4"}}), func(t *testing.T, p map[string]any) {
			if f := p["input_audio"].(map[string]any)["format"]; f != "m4a" {
				t.Errorf("m4a sent as format %v", f)
			}
		}},
		{"video", mediaFile(t, "v.mp4", "MP4", Media{Kind: MediaVideo, File: File{MIMEType: "video/mp4"}}), func(t *testing.T, p map[string]any) {
			if p["type"] != "video_url" || p["video_url"].(map[string]any)["url"] != "data:video/mp4;base64,"+b64("MP4") {
				t.Errorf("video part = %v", p)
			}
		}},
		{"pdf", mediaFile(t, "f.pdf", "PDF", Media{Kind: MediaDocument, File: File{MIMEType: "application/pdf", FileName: "factura.pdf"}}), func(t *testing.T, p map[string]any) {
			f := p["file"].(map[string]any)
			if p["type"] != "file" || f["filename"] != "factura.pdf" || f["file_data"] != "data:application/pdf;base64,"+b64("PDF") {
				t.Errorf("pdf part = %v", p)
			}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			api := newFakeOpenRouter(t)
			got, err := api.translator().Translate(context.Background(), c.media)
			if err != nil {
				t.Fatal(err)
			}
			if got != "una captura con un panic" {
				t.Errorf("Translate = %q, want the trimmed answer", got)
			}
			if api.auth != "Bearer sk-test" {
				t.Errorf("Authorization = %q", api.auth)
			}
			if api.body["model"] != DefaultMediaModel {
				t.Errorf("model = %v, want the default", api.body["model"])
			}
			c.check(t, api.part(t))
		})
	}
}

func TestTranslatorUsesTheConfiguredModel(t *testing.T) {
	api := newFakeOpenRouter(t)
	tr := api.translator()
	tr.Model = "google/gemini-3.8-flash"
	tr.Translate(context.Background(), mediaFile(t, "a.jpg", "x", Media{Kind: MediaPhoto}))
	if api.body["model"] != "google/gemini-3.8-flash" {
		t.Errorf("model = %v", api.body["model"])
	}
}

func TestTranslatorErrors(t *testing.T) {
	cases := []struct {
		name, answer, want string
		status             int
	}{
		{"provider error", `{"error":{"message":"No endpoints found that support input audio"}}`, "No endpoints found", http.StatusOK},
		{"http error", `{"error":{"message":"Insufficient credits"}}`, "Insufficient credits", http.StatusPaymentRequired},
		{"not json", `<html>bad gateway</html>`, "HTTP 502", http.StatusBadGateway},
		{"empty", `{"choices":[{"message":{"content":"  "}}]}`, "empty", http.StatusOK},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			api := newFakeOpenRouter(t)
			api.status, api.answer = c.status, c.answer
			_, err := api.translator().Translate(context.Background(), mediaFile(t, "a.jpg", "x", Media{Kind: MediaPhoto}))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err = %v, want it to mention %q", err, c.want)
			}
			if err != nil && strings.Contains(err.Error(), "sk-test") {
				t.Errorf("error leaks the API key: %v", err)
			}
		})
	}
}

func TestTextDocumentsAreNotTranslated(t *testing.T) {
	if translatable(&Media{Kind: MediaDocument, File: File{FileName: "notas.md"}}) {
		t.Error("a markdown document should be left to read_file")
	}
	if !translatable(&Media{Kind: MediaDocument, File: File{MIMEType: "application/pdf"}}) {
		t.Error("a PDF should be translated")
	}
}
