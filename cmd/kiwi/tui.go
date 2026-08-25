package main

import (
	"context"

	tea "charm.land/bubbletea/v2"
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
	_, err = p.Run()
	return err
}
