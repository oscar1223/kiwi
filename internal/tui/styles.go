package tui

import (
	"strings"

	"charm.land/lipgloss/v2"
	"github.com/oscar1223/kiwi/internal/permission"
)

// Kiwi's palette. Kiwi green for the brand, muted greys for chrome, and a
// distinct hue per mode so the header colour alone tells you what is allowed.
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
