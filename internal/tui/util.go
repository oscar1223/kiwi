package tui

import (
	"encoding/json"
	"fmt"
	"strings"
)

func sprintf(format string, args ...any) string { return fmt.Sprintf(format, args...) }

// oneLine collapses text to a single line and caps its length, for the
// summaries shown next to tool calls.
func oneLine(s string, max int) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = strings.TrimSpace(s[:i]) + " …"
	}
	if runes := []rune(s); len(runes) > max {
		return string(runes[:max]) + "…"
	}
	return s
}

// toolSummary renders a tool call's arguments compactly: the value that
// matters most for that tool, rather than raw JSON.
func toolSummary(name string, input []byte) string {
	var args map[string]any
	if err := jsonUnmarshal(input, &args); err != nil {
		return oneLine(string(input), 100)
	}

	// Show the argument that identifies the action for each known tool.
	for _, key := range []string{"command", "path", "pattern", "query", "name"} {
		if v, ok := args[key].(string); ok && v != "" {
			return oneLine(v, 100)
		}
	}
	if len(args) == 0 {
		return ""
	}
	return oneLine(string(input), 100)
}

func jsonUnmarshal(data []byte, v any) error { return json.Unmarshal(data, v) }
