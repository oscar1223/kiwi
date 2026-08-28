package tui

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// maxMentionBytes bounds one attached file. A mention is a convenience for
// pointing at a file, not a way to paste a database dump into the window: past
// this, the tail is dropped and the model is told so, which it can follow up
// on with read_file if it needs the rest.
const maxMentionBytes = 64 * 1024

// mentionPattern matches "@path" at the start of the input or after
// whitespace. Requiring that boundary is what keeps an email address or a Go
// struct tag from being read as a file reference.
var mentionPattern = regexp.MustCompile(`(^|\s)@([^\s@]+)`)

// expandFileMentions inlines every @path the user mentioned into the message
// the model receives.
//
// The user's own text is never rewritten — the mentions stay exactly as typed
// and the file contents are appended below — so the transcript still reads
// like what the person wrote, and a mention that resolves to nothing degrades
// into ordinary prose rather than an error. attached and missing are returned
// so the caller can say which files actually made it in: silently sending a
// question about a file that was never attached is the one outcome worth
// protecting against.
func expandFileMentions(workDir, text string) (expanded string, attached, missing []string) {
	matches := mentionPattern.FindAllStringSubmatch(text, -1)
	if len(matches) == 0 {
		return text, nil, nil
	}

	seen := map[string]bool{}
	var blocks strings.Builder

	for _, m := range matches {
		ref := m[2]
		if seen[ref] {
			continue
		}
		seen[ref] = true

		body, err := readMention(workDir, ref)
		if err != nil {
			missing = append(missing, ref)
			continue
		}
		attached = append(attached, ref)
		blocks.WriteString("\n\n--- " + ref + " ---\n")
		blocks.WriteString(body)
	}

	if blocks.Len() == 0 {
		return text, attached, missing
	}
	return text + blocks.String(), attached, missing
}

// readMention resolves one reference and reads it, refusing anything that is
// not a regular file: a directory or a device would either fail confusingly or
// never finish.
func readMention(workDir, ref string) (string, error) {
	path := ref
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		path = filepath.Join(home, strings.TrimPrefix(path, "~"))
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(workDir, path)
	}

	info, err := os.Stat(path)
	if err != nil {
		return "", err
	}
	if !info.Mode().IsRegular() {
		return "", os.ErrInvalid
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	if len(data) > maxMentionBytes {
		return string(data[:maxMentionBytes]) +
			"\n… (truncated; use read_file for the rest)\n", nil
	}
	return string(data), nil
}
