package main

import (
	"context"
	"fmt"
	"os"

	tea "charm.land/bubbletea/v2"
	"github.com/charmbracelet/x/ansi"
	"github.com/charmbracelet/x/term"
	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/tui"
)

// runTUI starts the interactive interface. It is what `kiwi` with no
// subcommand does.
func runTUI(ctx context.Context, g *globalFlags, mode permission.Mode) error {
	// Applied before the program starts, so the very first frame — including
	// an onboarding wizard on a fresh install — already uses the persisted
	// theme instead of flashing the default and then switching.
	if cfg, err := config.Load(); err == nil {
		tui.ApplyTheme(cfg.Theme)
	}

	// One object serves as both the broker's decider and the model's event
	// stream, which resolves the circular dependency between them.
	ev := tui.NewEvents()

	sess, err := newSession(ctx, g, mode, ev)
	if err != nil {
		return err
	}
	defer sess.Close()

	model := tui.New(tui.Options{
		Agent:           sess.agent,
		Broker:          sess.broker,
		WorkDir:         sess.workDir,
		ModelLabel:      sess.modelLabel,
		PromptOptions:   sess.promptOpts,
		Events:          ev,
		History:         sess.history,
		Store:           sess.store,
		SessionID:       sess.meta.ID,
		BaseContext:     ctx,
		Rebuild:         func() (*agent.Agent, string, error) { return sess.rebuildAgent(ctx) },
		NeedsOnboarding: sess.needsOnboarding,
	})

	// Automatic decisions are logged so there is always a trace of why an
	// action was allowed or blocked without a prompt appearing.
	sess.broker.OnAutoDecision(ev.LogAutoDecision)

	p := tea.NewProgram(model, tea.WithContext(ctx))

	// The mouse is deliberately left uncaptured (see tui.Model.View) so the
	// terminal keeps its own selection and copy. Alternate scroll is what
	// still makes the wheel useful under that choice: while the alt screen is
	// up, the terminal turns wheel events into arrow keys, which the model
	// routes to the transcript. A terminal only honours it as long as nothing
	// is capturing the mouse, which is exactly the trade being made here.
	if restore := enableAltScroll(); restore != nil {
		defer restore()
	}

	_, err = p.Run()

	// The alt screen is gone by now, and took the conversation with it. Print
	// it back into the real scrollback so closing Kiwi and scrolling up still
	// shows what happened — and so it can be selected and copied afterwards.
	if out := model.Transcript(); out != "" {
		fmt.Print(out)
	}
	return err
}

// enableAltScroll turns on alternate scroll mode, returning the function that
// turns it back off, or nil when stdout is not a terminal.
//
// Restoring it matters: left set, it changes how the wheel behaves in the
// shell the user drops back into.
func enableAltScroll() func() {
	if !term.IsTerminal(os.Stdout.Fd()) {
		return nil
	}
	const altScroll = ansi.DECMode(1007)
	fmt.Fprint(os.Stdout, ansi.SetMode(altScroll))
	return func() { fmt.Fprint(os.Stdout, ansi.ResetMode(altScroll)) }
}
