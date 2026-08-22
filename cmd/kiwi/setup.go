package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/prompt"
	sessionstore "github.com/oscar1223/kiwi/internal/session"
	"github.com/oscar1223/kiwi/internal/tools"
)

// runSession bundles everything a run needs, however it is driven.
//
// Both the TUI and `kiwi ask` build one of these; the only thing they choose
// differently is who answers permission questions.
type runSession struct {
	agent      *agent.Agent
	broker     *permission.Broker
	workDir    string
	modelLabel string
	promptOpts prompt.Options

	store   *sessionstore.Store
	meta    *sessionstore.Meta
	history []llm.Message
}

func (s *runSession) Close() error { return s.store.Close() }

func newSession(ctx context.Context, g *globalFlags, mode permission.Mode, decider permission.Decider) (*runSession, error) {
	cwd := g.cwd
	if cwd == "" {
		var err error
		if cwd, err = os.Getwd(); err != nil {
			return nil, err
		}
	}

	cfg, err := config.Load()
	if err != nil {
		return nil, err
	}
	name, profile, err := cfg.Profile(g.model)
	if err != nil {
		return nil, err
	}
	provider, err := config.BuildProvider(name, profile)
	if err != nil {
		return nil, err
	}

	store, err := openSessionStore()
	if err != nil {
		return nil, err
	}
	meta, history, err := resolveSession(ctx, store, cwd, g)
	if err != nil {
		store.Close()
		return nil, err
	}

	projectFile, projectInstructions := config.ProjectInstructions(cwd)
	promptOpts := prompt.Options{
		WorkingDir:          cwd,
		ProjectFile:         projectFile,
		ProjectInstructions: projectInstructions,
		ModeInstructions:    mode.Instructions(),
	}

	broker := permission.NewBroker(mode, decider)

	return &runSession{
		agent: &agent.Agent{
			Provider: provider,
			Tools:    tools.Default(cwd, broker),
			System:   prompt.Build(promptOpts),
		},
		broker:     broker,
		workDir:    cwd,
		modelLabel: fmt.Sprintf("%s/%s", provider.Name(), provider.Model()),
		promptOpts: promptOpts,
		store:      store,
		meta:       meta,
		history:    history,
	}, nil
}

// openSessionStore opens the single, shared sessions database that every
// kiwi invocation on this machine reads and writes.
func openSessionStore() (*sessionstore.Store, error) {
	dataDir, err := config.DataDir()
	if err != nil {
		return nil, err
	}
	return sessionstore.Open(filepath.Join(dataDir, "sessions.db"))
}

// resolveSession picks which session a run continues, or starts a new one.
//
//   - --resume <id>: that exact session, or an unambiguous prefix of it.
//   - --continue: the most recently updated session for this project; if the
//     project has none yet, a new one is started rather than erroring, since
//     "continue if possible, otherwise just start" is what a user actually
//     wants from a flag they may pass out of habit.
//   - neither: always a new session — every run is saved, so it shows up in
//     `kiwi session list` and can be picked up later even if nobody asked to
//     continue at the time.
func resolveSession(ctx context.Context, store *sessionstore.Store, cwd string, g *globalFlags) (*sessionstore.Meta, []llm.Message, error) {
	switch {
	case g.resumeID != "":
		meta, err := store.Get(ctx, g.resumeID)
		if err != nil {
			return nil, nil, fmt.Errorf("--resume %s: %w", g.resumeID, err)
		}
		history, err := store.Load(ctx, meta.ID)
		return meta, history, err

	case g.continueLast:
		meta, err := store.Latest(ctx, cwd)
		switch {
		case errors.Is(err, sessionstore.ErrNotFound):
			meta, err = store.Create(ctx, cwd)
			return meta, nil, err
		case err != nil:
			return nil, nil, err
		}
		history, err := store.Load(ctx, meta.ID)
		return meta, history, err

	default:
		meta, err := store.Create(ctx, cwd)
		return meta, nil, err
	}
}
