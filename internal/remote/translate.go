package remote

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// Translator turns a photo, audio, video or PDF into text, so a model that
// only reads text can work with it: the main model never sees the file, only
// what the translator says about it.
type Translator interface {
	Translate(ctx context.Context, m *Media) (string, error)
}

// DefaultMediaModel is cheap and reads images, audio, video and PDFs.
const DefaultMediaModel = "google/gemini-3.1-flash-lite"

// OpenRouterTranslator asks a multimodal model on OpenRouter, or on any
// OpenAI-compatible endpoint that takes the same content parts.
//
// It speaks plain JSON instead of going through the openai-go SDK: it needs
// one request with audio, video and file parts, which OpenRouter accepts and
// the SDK's types do not all describe.
type OpenRouterTranslator struct {
	APIKey string
	Model  string // empty means DefaultMediaModel
	// BaseURL is the API root, without /chat/completions. Empty means
	// OpenRouter.
	BaseURL string
	Client  *http.Client
}

const openRouterBase = "https://openrouter.ai/api/v1"

// The model is told who it is helping, so it hears "Kiwi" and "commit"
// rather than "Kigo" and "con Miti".
const translatorContext = "Te envían este fichero a Kiwi, un asistente de programación, por Telegram. " +
	"Tu respuesta la leerá otro modelo que no puede ver el fichero: sé fiel y no inventes. " +
	"Responde en el idioma del contenido y sin preámbulos.\n\n"

var translatorPrompts = map[mediaClass]string{
	classImage: "Describe la imagen con detalle. Si contiene texto (código, errores, una captura), transcríbelo literalmente.",
	classAudio: "Transcribe literalmente lo que se dice. Si no se entiende algo, márcalo con [inaudible]. Puede haber términos técnicos (commit, PR, tests, deploy).",
	classVideo: "Responde con dos apartados.\nTranscripción: lo que se dice, literal (o «sin voz»). Puede haber términos técnicos (commit, PR, tests, deploy).\nImagen: qué se ve, en pocas frases; transcribe el texto que aparezca en pantalla.",
	classPDF:   "Resume el documento y copia literalmente los datos importantes (cifras, fechas, nombres, código).",
}

// translatable reports whether the translator can read m. Text documents are
// left to the agent, which reads them itself.
func translatable(m *Media) bool {
	_, ok := translatorPrompts[m.class()]
	return ok
}

func (t *OpenRouterTranslator) Translate(ctx context.Context, m *Media) (string, error) {
	prompt, ok := translatorPrompts[m.class()]
	if !ok {
		return "", fmt.Errorf("a %s cannot be translated", m.class())
	}
	data, err := os.ReadFile(m.Path)
	if err != nil {
		return "", err
	}

	body, err := json.Marshal(map[string]any{
		"model":      cmp.Or(t.Model, DefaultMediaModel),
		"max_tokens": 4000,
		"messages": []any{map[string]any{
			"role": "user",
			"content": []any{
				map[string]any{"type": "text", "text": translatorContext + prompt},
				mediaPart(m, data),
			},
		}},
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cmp.Or(t.BaseURL, openRouterBase)+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+t.APIKey)

	hc := t.Client
	if hc == nil {
		hc = &http.Client{Timeout: 3 * time.Minute}
	}
	resp, err := hc.Do(req)
	if err != nil {
		return "", fmt.Errorf("translator: %w", err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return "", fmt.Errorf("translator: %w", err)
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Error *struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return "", fmt.Errorf("translator: HTTP %d: %s", resp.StatusCode, truncateRunes(string(raw), 200))
	}
	if out.Error != nil {
		return "", fmt.Errorf("translator: %s", out.Error.Message)
	}
	if resp.StatusCode != http.StatusOK || len(out.Choices) == 0 {
		return "", fmt.Errorf("translator: HTTP %d with no answer", resp.StatusCode)
	}
	text := strings.TrimSpace(out.Choices[0].Message.Content)
	if text == "" {
		return "", errors.New("translator: empty answer")
	}
	return text, nil
}

// mediaPart is the content part that carries the file, as OpenRouter takes
// each kind. All of them inline the file as base64.
func mediaPart(m *Media, data []byte) map[string]any {
	b64 := base64.StdEncoding.EncodeToString(data)
	switch m.class() {
	case classImage:
		return map[string]any{"type": "image_url", "image_url": map[string]any{
			"url": "data:" + cmp.Or(m.File.MIMEType, "image/jpeg") + ";base64," + b64,
		}}
	case classAudio:
		return map[string]any{"type": "input_audio", "input_audio": map[string]any{
			"data": b64, "format": audioFormat(m),
		}}
	case classVideo:
		return map[string]any{"type": "video_url", "video_url": map[string]any{
			"url": "data:" + cmp.Or(m.File.MIMEType, "video/mp4") + ";base64," + b64,
		}}
	}
	return map[string]any{"type": "file", "file": map[string]any{
		"filename":  cmp.Or(m.File.FileName, filepath.Base(m.Path)),
		"file_data": "data:application/pdf;base64," + b64,
	}}
}

// audioFormat names the container for input_audio. Telegram voice notes are
// Ogg/Opus, which the model takes as "ogg".
func audioFormat(m *Media) string {
	switch strings.ToLower(m.File.MIMEType) {
	case "audio/ogg", "audio/opus":
		return "ogg"
	case "audio/mpeg", "audio/mp3":
		return "mp3"
	case "audio/mp4", "audio/x-m4a", "audio/m4a":
		return "m4a"
	case "audio/wav", "audio/x-wav", "audio/wave":
		return "wav"
	case "audio/aac":
		return "aac"
	case "audio/flac", "audio/x-flac":
		return "flac"
	}
	if m.Kind == MediaVoice {
		return "ogg"
	}
	switch ext := strings.TrimPrefix(filepath.Ext(m.Path), "."); ext {
	case "oga", "opus":
		return "ogg"
	case "":
		return "mp3"
	default:
		return ext
	}
}
