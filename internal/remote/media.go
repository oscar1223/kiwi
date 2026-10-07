package remote

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// MaxDownload is the biggest file the Bot API lets a bot download.
const MaxDownload = 20 << 20

// MediaKind is what Telegram says a message carries.
type MediaKind string

const (
	MediaPhoto    MediaKind = "photo"
	MediaVoice    MediaKind = "voice"
	MediaAudio    MediaKind = "audio"
	MediaVideo    MediaKind = "video" // videos and round video notes
	MediaDocument MediaKind = "document"
)

// Media is a file a message carries. Path is set once the bot has
// downloaded it.
type Media struct {
	Kind MediaKind
	File File
	Path string
}

// Media returns the file the message carries, or nil if it has none Kiwi
// handles. A photo is the biggest of the sizes Telegram sends.
func (m Message) Media() *Media {
	switch {
	case len(m.Photo) > 0:
		best := m.Photo[0]
		for _, p := range m.Photo[1:] {
			if p.Width*p.Height > best.Width*best.Height {
				best = p
			}
		}
		return &Media{Kind: MediaPhoto, File: best}
	case m.Voice != nil:
		return &Media{Kind: MediaVoice, File: *m.Voice}
	case m.Audio != nil:
		return &Media{Kind: MediaAudio, File: *m.Audio}
	case m.Video != nil:
		return &Media{Kind: MediaVideo, File: *m.Video}
	case m.VideoNote != nil:
		return &Media{Kind: MediaVideo, File: *m.VideoNote}
	case m.Document != nil:
		return &Media{Kind: MediaDocument, File: *m.Document}
	}
	return nil
}

// mediaClass is what a file is, whatever Telegram called it: a photo sent
// as a document is still an image.
type mediaClass string

const (
	classImage mediaClass = "image"
	classAudio mediaClass = "audio"
	classVideo mediaClass = "video"
	classPDF   mediaClass = "pdf"
	classText  mediaClass = "text"  // the agent reads it itself
	classOther mediaClass = "other" // kept, but nothing can read it
)

// textExts are documents the agent can read with read_file, so they are not
// sent to the translator.
var textExts = map[string]bool{
	".txt": true, ".md": true, ".csv": true, ".tsv": true, ".json": true, ".yaml": true, ".yml": true,
	".toml": true, ".xml": true, ".html": true, ".log": true, ".sql": true, ".sh": true, ".go": true,
	".py": true, ".js": true, ".ts": true, ".tsx": true, ".jsx": true, ".rs": true, ".java": true,
	".c": true, ".h": true, ".cpp": true, ".rb": true, ".php": true, ".css": true, ".swift": true,
	".kt": true, ".ini": true, ".conf": true, ".diff": true, ".patch": true,
}

func (m *Media) class() mediaClass {
	switch m.Kind {
	case MediaPhoto:
		return classImage
	case MediaVoice, MediaAudio:
		return classAudio
	case MediaVideo:
		return classVideo
	}
	mime := strings.ToLower(m.File.MIMEType)
	ext := strings.ToLower(filepath.Ext(m.File.FileName))
	switch {
	case strings.HasPrefix(mime, "image/"):
		return classImage
	case strings.HasPrefix(mime, "audio/"):
		return classAudio
	case strings.HasPrefix(mime, "video/"):
		return classVideo
	case mime == "application/pdf" || ext == ".pdf":
		return classPDF
	case strings.HasPrefix(mime, "text/") || textExts[ext]:
		return classText
	}
	return classOther
}

// label names the media in Spanish, for the chat and for the model.
func (m *Media) label() string {
	switch m.Kind {
	case MediaPhoto:
		return "Foto"
	case MediaVoice:
		return "Nota de voz"
	case MediaAudio:
		return "Audio"
	case MediaVideo:
		return "Vídeo"
	}
	return "Documento"
}

func (m *Media) emoji() string {
	switch m.class() {
	case classImage:
		return "📷"
	case classAudio:
		return "🎙️"
	case classVideo:
		return "🎬"
	}
	return "📄"
}

// ext is the extension to save the file with: the document's own, or one
// that matches what Telegram sends for each kind.
func (m *Media) ext() string {
	if ext := filepath.Ext(m.File.FileName); ext != "" && !strings.ContainsAny(ext, `/\`) {
		return strings.ToLower(ext)
	}
	switch mime := strings.ToLower(m.File.MIMEType); {
	case m.Kind == MediaPhoto || mime == "image/jpeg":
		return ".jpg"
	case m.Kind == MediaVoice || mime == "audio/ogg":
		return ".ogg"
	case mime == "audio/mpeg":
		return ".mp3"
	case mime == "audio/mp4" || mime == "audio/x-m4a":
		return ".m4a"
	case m.Kind == MediaVideo || mime == "video/mp4":
		return ".mp4"
	case mime == "application/pdf":
		return ".pdf"
	}
	return ".bin"
}

// download saves the media into dir and sets Path. Files are named after when
// they arrived and the message, never after the name the sender chose.
func (b *Bot) download(ctx context.Context, msg Message, m *Media, dir string) error {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return err
	}
	filePath, err := b.client.GetFile(ctx, m.File.FileID)
	if err != nil {
		return err
	}
	name := fmt.Sprintf("%s-%d%s", time.Now().Format("20060102-150405"), msg.MessageID, m.ext())
	path := filepath.Join(dir, name)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return err
	}
	err = b.client.Download(ctx, filePath, f, MaxDownload)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		os.Remove(path)
		return err
	}
	m.Path = path
	return nil
}
