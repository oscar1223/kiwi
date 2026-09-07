package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/oscar1223/kiwi/internal/diagnostics"
	"github.com/oscar1223/kiwi/internal/permission"
)

// MaxFileBytes bounds how much of a file is read into the context window.
const MaxFileBytes = 256 * 1024

// FS carries what the file tools need: where relative paths resolve and who
// approves mutations.
type FS struct {
	WorkDir string
	Perms   *permission.Broker
	// Diag checks what was just written against the project's own tooling and
	// appends anything it finds to the observation. Nil turns that off, which
	// is what the tests below and any non-project working directory use.
	Diag *diagnostics.Runner
}

// checked appends the project checker's verdict on abs to a tool's result.
//
// Reporting it in the same observation rather than as a separate turn is the
// whole point: the model is already reading this text, and a compile error it
// learns about three tool calls later is one it has already built on top of.
func (f *FS) checked(ctx context.Context, result string, abs ...string) string {
	if f.Diag == nil {
		return result
	}
	return result + f.Diag.Check(ctx, abs...)
}

// resolve turns a user- or model-supplied path into an absolute one.
func (f *FS) resolve(path string) (string, error) {
	if path == "" {
		return "", fmt.Errorf("path is required")
	}
	if strings.HasPrefix(path, "~") {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		path = filepath.Join(home, strings.TrimPrefix(path, "~"))
	}
	if !filepath.IsAbs(path) {
		path = filepath.Join(f.WorkDir, path)
	}
	return filepath.Clean(path), nil
}

// display shortens a path for prompts and observations.
func (f *FS) display(abs string) string {
	if rel, err := filepath.Rel(f.WorkDir, abs); err == nil && !strings.HasPrefix(rel, "..") {
		return rel
	}
	return abs
}

// --- read_file ---

type ReadFile struct{ *FS }

func (ReadFile) Name() string { return "read_file" }

func (ReadFile) Description() string {
	return "Read a file from disk. Returns the contents with line numbers. " +
		"Use offset and limit for files too large to read at once."
}

func (ReadFile) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":   map[string]any{"type": "string", "description": "Path to the file, absolute or relative to the working directory."},
			"offset": map[string]any{"type": "integer", "description": "1-based line to start from."},
			"limit":  map[string]any{"type": "integer", "description": "Maximum number of lines to return."},
		},
		"required": []string{"path"},
	}
}

func (t ReadFile) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Path   string `json:"path"`
		Offset int    `json:"offset"`
		Limit  int    `json:"limit"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	abs, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}

	info, err := os.Stat(abs)
	if err != nil {
		return "", err
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s is a directory, not a file", t.display(abs))
	}

	data, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}
	if len(data) == 0 {
		return fmt.Sprintf("(%s is empty)", t.display(abs)), nil
	}

	truncated := false
	if len(data) > MaxFileBytes {
		data = data[:MaxFileBytes]
		truncated = true
	}

	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	start := 0
	if in.Offset > 0 {
		start = in.Offset - 1
	}
	if start >= len(lines) {
		return "", fmt.Errorf("offset %d is past the end of %s (%d lines)", in.Offset, t.display(abs), len(lines))
	}
	end := len(lines)
	if in.Limit > 0 && start+in.Limit < end {
		end = start + in.Limit
	}

	var b strings.Builder
	for i := start; i < end; i++ {
		fmt.Fprintf(&b, "%6d\t%s\n", i+1, lines[i])
	}
	if end < len(lines) {
		fmt.Fprintf(&b, "… %d more lines\n", len(lines)-end)
	}
	if truncated {
		b.WriteString("… (file truncated: too large to read in full)\n")
	}
	return b.String(), nil
}

// --- write_file ---

type WriteFile struct{ *FS }

func (WriteFile) Name() string { return "write_file" }

func (WriteFile) Description() string {
	return "Create a new file, or replace an existing file's entire contents. " +
		"Prefer edit_file for changing part of a file that already exists."
}

func (WriteFile) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":    map[string]any{"type": "string", "description": "Path to the file."},
			"content": map[string]any{"type": "string", "description": "The complete file contents."},
		},
		"required": []string{"path", "content"},
	}
}

func (t WriteFile) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Path    string `json:"path"`
		Content string `json:"content"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	abs, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}

	before, readErr := os.ReadFile(abs)
	exists := readErr == nil
	detail := "create " + t.display(abs)
	if exists {
		detail = "overwrite " + t.display(abs)
	}

	if err := t.Perms.Ask(ctx, permission.Action{
		Name:   permission.ActionWrite,
		Detail: detail,
		Diff:   UnifiedDiff(t.display(abs), string(before), in.Content),
	}); err != nil {
		return "", err
	}

	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(abs, []byte(in.Content), 0o644); err != nil {
		return "", err
	}

	verb := "Created"
	if exists {
		verb = "Overwrote"
	}
	return t.checked(ctx, fmt.Sprintf("%s %s (%d lines)", verb, t.display(abs), countLines(in.Content)), abs), nil
}

// --- edit_file ---

type EditFile struct{ *FS }

func (EditFile) Name() string { return "edit_file" }

func (EditFile) Description() string {
	return "Replace an exact string in an existing file. old_string must appear " +
		"exactly once — include surrounding context to make it unique. Cheaper " +
		"and safer than rewriting the whole file."
}

func (EditFile) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path":        map[string]any{"type": "string", "description": "Path to the file."},
			"old_string":  map[string]any{"type": "string", "description": "Exact text to replace, including indentation."},
			"new_string":  map[string]any{"type": "string", "description": "Replacement text."},
			"replace_all": map[string]any{"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."},
		},
		"required": []string{"path", "old_string", "new_string"},
	}
}

func (t EditFile) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Path       string `json:"path"`
		OldString  string `json:"old_string"`
		NewString  string `json:"new_string"`
		ReplaceAll bool   `json:"replace_all"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if in.OldString == in.NewString {
		return "", fmt.Errorf("old_string and new_string are identical; nothing to do")
	}
	abs, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}
	before := string(data)

	count := strings.Count(before, in.OldString)
	switch {
	case count == 0:
		return "", fmt.Errorf("old_string not found in %s", t.display(abs))
	case count > 1 && !in.ReplaceAll:
		return "", fmt.Errorf("old_string appears %d times in %s; add surrounding context to make it unique, or set replace_all", count, t.display(abs))
	}

	after := before
	if in.ReplaceAll {
		after = strings.ReplaceAll(before, in.OldString, in.NewString)
	} else {
		after = strings.Replace(before, in.OldString, in.NewString, 1)
	}

	if err := t.Perms.Ask(ctx, permission.Action{
		Name:   permission.ActionEdit,
		Detail: "edit " + t.display(abs),
		Diff:   UnifiedDiff(t.display(abs), before, after),
	}); err != nil {
		return "", err
	}

	if err := os.WriteFile(abs, []byte(after), 0o644); err != nil {
		return "", err
	}

	if count > 1 {
		return t.checked(ctx, fmt.Sprintf("Edited %s (%d replacements)", t.display(abs), count), abs), nil
	}
	return t.checked(ctx, fmt.Sprintf("Edited %s", t.display(abs)), abs), nil
}

func countLines(s string) int {
	if s == "" {
		return 0
	}
	return strings.Count(strings.TrimSuffix(s, "\n"), "\n") + 1
}

// --- multi_edit ---

// maxEditsPerCall bounds one multi_edit. Past this it is a rewrite, and
// write_file says so more honestly.
const maxEditsPerCall = 50

// MultiEdit applies several replacements to one file in a single call.
//
// The edits are applied in order against an in-memory copy and only written
// once every one of them has succeeded. That atomicity is the point: a batch
// that failed halfway would leave the file in a state neither the model nor
// the user asked for, and the model would then be reasoning about a file it
// no longer understands.
type MultiEdit struct{ *FS }

func (MultiEdit) Name() string { return "multi_edit" }

func (MultiEdit) Description() string {
	return "Apply several exact-string replacements to one file in a single call. " +
		"Edits apply in order, each seeing the result of the previous one, and " +
		"either all of them succeed or the file is left untouched. Prefer this " +
		"over repeated edit_file calls on the same file."
}

func (MultiEdit) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"path": map[string]any{"type": "string", "description": "Path to the file."},
			"edits": map[string]any{
				"type":        "array",
				"description": "Replacements, applied in order.",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"old_string":  map[string]any{"type": "string", "description": "Exact text to replace, including indentation."},
						"new_string":  map[string]any{"type": "string", "description": "Replacement text."},
						"replace_all": map[string]any{"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match."},
					},
					"required": []string{"old_string", "new_string"},
				},
			},
		},
		"required": []string{"path", "edits"},
	}
}

func (t MultiEdit) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Path  string `json:"path"`
		Edits []struct {
			OldString  string `json:"old_string"`
			NewString  string `json:"new_string"`
			ReplaceAll bool   `json:"replace_all"`
		} `json:"edits"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	if len(in.Edits) == 0 {
		return "", fmt.Errorf("edits is empty; nothing to do")
	}
	if len(in.Edits) > maxEditsPerCall {
		return "", fmt.Errorf("%d edits is too many (max %d); rewrite the file with write_file instead", len(in.Edits), maxEditsPerCall)
	}
	abs, err := t.resolve(in.Path)
	if err != nil {
		return "", err
	}

	data, err := os.ReadFile(abs)
	if err != nil {
		return "", err
	}
	before := string(data)

	after := before
	replacements := 0
	for i, e := range in.Edits {
		if e.OldString == e.NewString {
			return "", fmt.Errorf("edit %d: old_string and new_string are identical", i+1)
		}
		// Counted against the running result, not the original: an edit is
		// allowed to act on what an earlier one produced, and reporting a
		// match that a previous edit has already consumed would be a lie.
		count := strings.Count(after, e.OldString)
		switch {
		case count == 0:
			return "", fmt.Errorf("edit %d: old_string not found in %s (nothing was written)", i+1, t.display(abs))
		case count > 1 && !e.ReplaceAll:
			return "", fmt.Errorf("edit %d: old_string appears %d times in %s; add surrounding context to make it unique, or set replace_all (nothing was written)",
				i+1, count, t.display(abs))
		}
		if e.ReplaceAll {
			after = strings.ReplaceAll(after, e.OldString, e.NewString)
			replacements += count
			continue
		}
		after = strings.Replace(after, e.OldString, e.NewString, 1)
		replacements++
	}

	if after == before {
		return fmt.Sprintf("%s is unchanged", t.display(abs)), nil
	}

	// One prompt for the batch, showing the combined diff: approving each
	// replacement separately would tell the user less, not more.
	if err := t.Perms.Ask(ctx, permission.Action{
		Name:   permission.ActionEdit,
		Detail: fmt.Sprintf("edit %s (%d changes)", t.display(abs), len(in.Edits)),
		Diff:   UnifiedDiff(t.display(abs), before, after),
	}); err != nil {
		return "", err
	}

	if err := os.WriteFile(abs, []byte(after), 0o644); err != nil {
		return "", err
	}
	return t.checked(ctx, fmt.Sprintf("Edited %s (%d edits, %d replacements)", t.display(abs), len(in.Edits), replacements), abs), nil
}
