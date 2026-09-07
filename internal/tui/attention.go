package tui

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	tea "charm.land/bubbletea/v2"
)

// EnvNotify controls desktop notifications: "off" silences everything,
// "desktop" adds a system notification to the bell.
const EnvNotify = "KIWI_NOTIFY"

// notifyAfter is how long a turn must run before finishing it is worth an
// alert. Below this the user is still watching, and a terminal that beeps at
// every short answer is a terminal people mute.
const notifyAfter = 20 * time.Second

// notifyDone alerts the user that a turn has finished.
//
// This stops being a nicety once work mode runs unattended for up to 200
// steps: leaving the terminal is the point, and without an alert the only way
// to know it is done is to keep looking. OpenCode calls the same idea
// attention settings.
//
// The bell is the default because it works everywhere and needs nothing
// installed; the desktop notification is opt-in because it shells out to a
// different binary on every platform, and guessing wrong is worse than staying
// quiet.
func (m *Model) notifyDone() tea.Cmd {
	if m.began.IsZero() || time.Since(m.began) < notifyAfter {
		return nil
	}
	mode := strings.ToLower(strings.TrimSpace(os.Getenv(EnvNotify)))
	if mode == "off" || mode == "false" || mode == "0" {
		return nil
	}

	title := "kiwi finished in " + filepath.Base(m.opts.WorkDir)
	desktop := mode == "desktop" || mode == "all"
	return func() tea.Msg {
		if desktop {
			notifyDesktop(title)
		}
		// Written straight to the terminal rather than into the frame: it is
		// a control character, not content, and putting it in the transcript
		// would re-ring it on every redraw.
		_, _ = os.Stdout.WriteString("\a")
		return nil
	}
}

// notifyDesktop posts a system notification, best effort.
func notifyDesktop(title string) {
	var cmd *exec.Cmd
	switch {
	case hasBinary("osascript"):
		cmd = exec.Command("osascript", "-e",
			`display notification "your turn" with title `+quoteAppleScript(title))
	case hasBinary("notify-send"):
		cmd = exec.Command("notify-send", title, "your turn")
	default:
		return
	}
	_ = cmd.Run()
}

func hasBinary(name string) bool {
	_, err := exec.LookPath(name)
	return err == nil
}

// quoteAppleScript wraps a string as an AppleScript literal. The directory
// name reaches this from the filesystem, so it cannot be pasted in raw.
func quoteAppleScript(s string) string {
	return `"` + strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(s) + `"`
}

// --- the shortcut panel ---

// toggleShortcuts shows the keybindings without leaving what is being typed.
//
// /help already lists them, but it prints into the transcript and scrolls away
// what you were looking at. "?" is the version you use while your hands are
// still on the keyboard, which is when you need it.
func (m *Model) toggleShortcuts() tea.Cmd {
	m.showShortcuts = !m.showShortcuts
	return nil
}

// renderShortcuts draws the panel above the input.
func (m *Model) renderShortcuts(width int) []string {
	if !m.showShortcuts {
		return nil
	}
	rows := []string{styleDim.Render("  shortcuts — ? to hide")}
	for _, r := range keybindRows {
		rows = append(rows, fit("  "+styleTool.Render(sprintf("%-12s", r[0]))+styleDim.Render(r[1]), width))
	}
	return rows
}

// --- composing in $EDITOR ---

// editorRequestMsg carries text back from the external editor.
type editorRequestMsg struct {
	text string
	err  error
}

// composeInEditor opens the current draft in $EDITOR and puts the result back
// in the input.
//
// A prompt worth several paragraphs is miserable to write in a one-line box,
// and the alternative people reach for — writing it elsewhere and pasting —
// loses the draft when the paste goes wrong.
func (m *Model) composeInEditor() tea.Cmd {
	editor := firstNonEmpty(os.Getenv("VISUAL"), os.Getenv("EDITOR"))
	if editor == "" {
		return m.println(styleDim.Render("  set $EDITOR to compose in your editor"))
	}

	draft := m.input.Value()
	return tea.ExecProcess(editorCommand(editor, draft), func(err error) tea.Msg {
		if err != nil {
			return editorRequestMsg{err: err}
		}
		data, readErr := os.ReadFile(editorTempPath)
		_ = os.Remove(editorTempPath)
		if readErr != nil {
			return editorRequestMsg{err: readErr}
		}
		return editorRequestMsg{text: string(data)}
	})
}

// editorTempPath is where the draft is handed to the editor and read back.
// One path per process, since only one editor can be open at a time — the
// program is suspended while it runs.
var editorTempPath string

func editorCommand(editor, draft string) *exec.Cmd {
	if editorTempPath == "" {
		f, err := os.CreateTemp("", "kiwi-*.md")
		if err != nil {
			return exec.Command(editor)
		}
		editorTempPath = f.Name()
		f.Close()
	}
	_ = os.WriteFile(editorTempPath, []byte(draft), 0o600)

	// Split so "code --wait" and "nvim -u NONE" work, not only a bare binary
	// name — an $EDITOR with flags in it is completely ordinary.
	parts := strings.Fields(editor)
	return exec.Command(parts[0], append(parts[1:], editorTempPath)...)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
