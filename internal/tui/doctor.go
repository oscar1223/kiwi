package tui

import (
	"context"
	"sort"
	"strings"

	"github.com/oscar1223/kiwi/internal/mcp"
	"github.com/oscar1223/kiwi/internal/reach"
)

// doctorFlow prints which ways of reaching the internet work here: the same
// report as `kiwi doctor`. It runs as a flow because checking gh's sign-in
// runs a subprocess, which must not block the UI.
func (m *Model) doctorFlow(ctx context.Context) {
	channels := reach.Check(ctx, reach.SystemProbe(func() ([]string, error) {
		cfg, err := mcp.LoadConfig()
		if err != nil {
			return nil, err
		}
		names := make([]string, 0, len(cfg))
		for n := range cfg {
			names = append(names, n)
		}
		sort.Strings(names)
		return names, nil
	}))

	lines := []string{styleKiwi.Render("  reaching the internet")}
	for _, row := range strings.Split(strings.TrimRight(reach.Render(channels), "\n"), "\n") {
		style := styleDim
		if strings.HasPrefix(row, "✗") {
			style = styleWarn
		}
		lines = append(lines, "  "+style.Render(row))
	}
	m.events.send(ctx, printLinesMsg{lines: lines})
}
