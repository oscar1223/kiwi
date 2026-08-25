package tui

import (
	"image/color"
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Kiwi's palette. Kiwi green for the brand, muted greys for chrome, and a
// distinct hue per mode so the header colour alone tells you what is allowed.
// These are package vars, not consts, because applyTheme reassigns them at
// runtime when the user switches themes — see Theme below.
var (
	colKiwi   = lipgloss.Color("#8FD14F")
	colDim    = lipgloss.Color("#6C7086")
	colText   = lipgloss.Color("#CDD6F4")
	colErr    = lipgloss.Color("#F38BA8")
	colWarn   = lipgloss.Color("#F9E2AF")
	colPlan   = lipgloss.Color("#89DCEB")
	colAsk    = lipgloss.Color("#9399B2")
	colAdd    = lipgloss.Color("#A6E3A1")
	colDel    = lipgloss.Color("#F38BA8")
	colAccent = lipgloss.Color("#CBA6F7")
)

var (
	styleUser     = lipgloss.NewStyle().Foreground(colAccent).Bold(true)
	styleKiwi     = lipgloss.NewStyle().Foreground(colText)
	styleDim      = lipgloss.NewStyle().Foreground(colDim)
	styleTool     = lipgloss.NewStyle().Foreground(colKiwi)
	styleToolArgs = lipgloss.NewStyle().Foreground(colDim)
	styleErr      = lipgloss.NewStyle().Foreground(colErr)
	styleWarn     = lipgloss.NewStyle().Foreground(colWarn)
	styleCode     = lipgloss.NewStyle().Foreground(colWarn)
	styleDiffAdd  = lipgloss.NewStyle().Foreground(colAdd)
	styleDiffDel  = lipgloss.NewStyle().Foreground(colDel)
	styleDiffHead = lipgloss.NewStyle().Foreground(colDim).Bold(true)
	stylePrompt   = lipgloss.NewStyle().Foreground(colKiwi).Bold(true)
)

// Theme bundles the whole colour palette so it can be swapped as one unit.
type Theme struct {
	Name                                            string
	Kiwi, Dim, Text, Err, Warn, Plan, Ask, Add, Del color.Color
	Accent                                          color.Color
}

// DefaultThemeName is what an empty config.Theme resolves to.
const DefaultThemeName = "kiwi-dark"

var themeKiwiDark = Theme{
	Name: "kiwi-dark",
	Kiwi: lipgloss.Color("#8FD14F"), Dim: lipgloss.Color("#6C7086"),
	Text: lipgloss.Color("#CDD6F4"), Err: lipgloss.Color("#F38BA8"),
	Warn: lipgloss.Color("#F9E2AF"), Plan: lipgloss.Color("#89DCEB"),
	Ask: lipgloss.Color("#9399B2"), Add: lipgloss.Color("#A6E3A1"),
	Del: lipgloss.Color("#F38BA8"), Accent: lipgloss.Color("#CBA6F7"),
}

// kiwi-light swaps in darker tones so the same roles stay legible on a
// light terminal background instead of washing out.
var themeKiwiLight = Theme{
	Name: "kiwi-light",
	Kiwi: lipgloss.Color("#4C8B1F"), Dim: lipgloss.Color("#6C6F85"),
	Text: lipgloss.Color("#1E1E2E"), Err: lipgloss.Color("#B4304A"),
	Warn: lipgloss.Color("#8A6D00"), Plan: lipgloss.Color("#0E7C86"),
	Ask: lipgloss.Color("#5C5F77"), Add: lipgloss.Color("#1E8449"),
	Del: lipgloss.Color("#B4304A"), Accent: lipgloss.Color("#8839EF"),
}

// themes lists the built-in themes in display order. Adding a new one is
// just another entry here — applyTheme and the /theme picker need no changes.
var themes = []Theme{themeKiwiDark, themeKiwiLight}

func themeByName(name string) (Theme, bool) {
	if name == "" {
		name = DefaultThemeName
	}
	for _, t := range themes {
		if t.Name == name {
			return t, true
		}
	}
	return Theme{}, false
}

// ApplyTheme sets the active theme by name, falling back to the default for
// an unrecognised name (an empty string, or a stale name from an old config).
// It exists for cmd/kiwi to apply the persisted theme before the program
// starts, when Update's goroutine does not exist yet and calling applyTheme
// directly would be just as safe but less discoverable from outside the
// package.
func ApplyTheme(name string) {
	t, ok := themeByName(name)
	if !ok {
		t = themeKiwiDark
	}
	applyTheme(t)
}

// applyTheme reassigns the package-level colour and style vars. It must only
// ever be called from Update's goroutine (or before the program starts) —
// View() reads these vars from Bubble Tea's own goroutine, so a concurrent
// write from a flow goroutine would be a real data race.
func applyTheme(t Theme) {
	colKiwi, colDim, colText, colErr, colWarn = t.Kiwi, t.Dim, t.Text, t.Err, t.Warn
	colPlan, colAsk, colAdd, colDel, colAccent = t.Plan, t.Ask, t.Add, t.Del, t.Accent

	styleUser = lipgloss.NewStyle().Foreground(colAccent).Bold(true)
	styleKiwi = lipgloss.NewStyle().Foreground(colText)
	styleDim = lipgloss.NewStyle().Foreground(colDim)
	styleTool = lipgloss.NewStyle().Foreground(colKiwi)
	styleToolArgs = lipgloss.NewStyle().Foreground(colDim)
	styleErr = lipgloss.NewStyle().Foreground(colErr)
	styleWarn = lipgloss.NewStyle().Foreground(colWarn)
	styleCode = lipgloss.NewStyle().Foreground(colWarn)
	styleDiffAdd = lipgloss.NewStyle().Foreground(colAdd)
	styleDiffDel = lipgloss.NewStyle().Foreground(colDel)
	styleDiffHead = lipgloss.NewStyle().Foreground(colDim).Bold(true)
	stylePrompt = lipgloss.NewStyle().Foreground(colKiwi).Bold(true)
}

// modeStyle colours the header by what the mode permits: grey asks, blue is
// read-only, green writes.
func modeStyle(m permission.Mode) lipgloss.Style {
	switch m {
	case permission.ModePlan:
		return lipgloss.NewStyle().Foreground(colPlan).Bold(true)
	case permission.ModeWork:
		return lipgloss.NewStyle().Foreground(colKiwi).Bold(true)
	default:
		return lipgloss.NewStyle().Foreground(colAsk).Bold(true)
	}
}

// renderDiff colours a unified diff for the approval prompt.
func renderDiff(diff string, maxLines int) string {
	lines := strings.Split(strings.TrimRight(diff, "\n"), "\n")
	truncated := 0
	if maxLines > 0 && len(lines) > maxLines {
		truncated = len(lines) - maxLines
		lines = lines[:maxLines]
	}

	var b strings.Builder
	for i, line := range lines {
		if i > 0 {
			b.WriteString("\n")
		}
		switch {
		case strings.HasPrefix(line, "+++"), strings.HasPrefix(line, "---"):
			b.WriteString(styleDiffHead.Render(line))
		case strings.HasPrefix(line, "@@"):
			b.WriteString(styleDim.Render(line))
		case strings.HasPrefix(line, "+"):
			b.WriteString(styleDiffAdd.Render(line))
		case strings.HasPrefix(line, "-"):
			b.WriteString(styleDiffDel.Render(line))
		default:
			b.WriteString(styleDim.Render(line))
		}
	}
	if truncated > 0 {
		b.WriteString("\n")
		b.WriteString(styleDim.Render(sprintf("… %d more diff lines", truncated)))
	}
	return b.String()
}

// bullet renders a marker plus content, indenting wrapped lines under the text
// rather than under the marker.
func bullet(marker, content string) string {
	indent := strings.Repeat(" ", lipgloss.Width(marker)+1)
	lines := strings.Split(content, "\n")
	for i := 1; i < len(lines); i++ {
		lines[i] = indent + lines[i]
	}
	return marker + " " + strings.Join(lines, "\n")
}
