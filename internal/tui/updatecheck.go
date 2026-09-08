package tui

import (
	"context"

	tea "charm.land/bubbletea/v2"

	// Aliased: the package tests already have a helper called update.
	updatecheck "github.com/oscar1223/kiwi/internal/update"
)

// updateAvailableMsg carries a newer release found in the background.
type updateAvailableMsg string

// checkUpdateCmd looks for a newer release without holding up the first frame.
// It returns nil when there is nothing to say, which is the common case — a
// notice that appears every start would stop being read.
func checkUpdateCmd(current string) tea.Cmd {
	if current == "" || current == updatecheck.DevVersion {
		return nil
	}
	return func() tea.Msg {
		// Not the turn context: this outlives no turn and cancelling a turn
		// should not cancel it. Check bounds itself.
		if latest := updatecheck.Check(context.Background(), current); latest != "" {
			return updateAvailableMsg(latest)
		}
		return nil
	}
}

// updateNotice is the line shown once a newer release is known. It names the
// command rather than only the fact, so acting on it costs no searching.
func updateNotice(latest string) string {
	return sprintf("  %s %s",
		styleKiwi.Render("↑"),
		styleDim.Render(sprintf("kiwi %s is available — run `kiwi update`", latest)))
}
