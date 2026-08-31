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
	"github.com/oscar1223/kiwi/internal/mcp"
	"github.com/oscar1223/kiwi/internal/memory"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/proc"
	"github.com/oscar1223/kiwi/internal/prompt"
	sessionstore "github.com/oscar1223/kiwi/internal/session"
	"github.com/oscar1223/kiwi/internal/skills"
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

	mcpManager *mcp.Manager
	// procs is created once and reused across rebuilds — a background
	// process started before a /model switch must keep running after it,
	// unlike the MCP manager, which does need to reconnect.
	procs *proc.Registry
	// asker, when non-nil, registers ask_questions with the agent — see
	// assembleAgent. It is the session's decider re-typed as tools.Asker,
	// present only when something can actually show the model's questions
	// to a human (the TUI); `kiwi ask` and other headless deciders leave it
	// nil, which drops ask_questions from the toolset entirely rather than
	// having every call fail at run time.
	asker tools.Asker

	// needsOnboarding is true when no profile's API key is set — the exact
	// shape of a brand-new install. agent is nil in that case; the TUI runs
	// its onboarding wizard instead of the normal banner, and the wizard's
	// own rebuild is what constructs the real agent for the first time.
	needsOnboarding bool
}

// Close releases everything the session opened: the sessions database, any
// live MCP server connections, and any background processes still running.
func (s *runSession) Close() error {
	if s.mcpManager != nil {
		s.mcpManager.Close()
	}
	if s.procs != nil {
		s.procs.KillAll()
	}
	return s.store.Close()
}

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

	store, err := openSessionStore()
	if err != nil {
		return nil, err
	}
	meta, history, err := resolveSession(ctx, store, cwd, g)
	if err != nil {
		store.Close()
		return nil, err
	}

	broker := permission.NewBroker(mode, decider)
	procs := proc.NewRegistry()
	asker, _ := decider.(tools.Asker)

	provider, err := config.BuildProvider(ctx, name, profile)
	if err != nil {
		if !errors.Is(err, config.ErrMissingAPIKey) {
			store.Close()
			return nil, err
		}
		// A missing key is exactly what a brand-new install looks like — not
		// a hard failure. assembleAgent (MCP connect, skills load, task
		// tool) is skipped entirely: none of it matters until the wizard
		// finishes and its own rebuild constructs the real agent.
		return &runSession{
			broker:          broker,
			workDir:         cwd,
			modelLabel:      "not configured",
			store:           store,
			meta:            meta,
			history:         history,
			procs:           procs,
			asker:           asker,
			needsOnboarding: true,
		}, nil
	}

	a, promptOpts, mgr := assembleAgent(ctx, provider, cwd, mode, broker, procs, asker)

	return &runSession{
		agent:      a,
		broker:     broker,
		workDir:    cwd,
		modelLabel: fmt.Sprintf("%s/%s", provider.Name(), provider.Model()),
		promptOpts: promptOpts,
		store:      store,
		meta:       meta,
		history:    history,
		mcpManager: mgr,
		procs:      procs,
		asker:      asker,
	}, nil
}

// assembleAgent gathers everything an Agent needs beyond its provider:
// project instructions, MCP servers, skills, and the task tool's two
// subagent toolsets. Both newSession's initial build and rebuildAgent (used
// after a /model, /mcp, or /skill change) go through this, so "what an agent
// looks like" has one definition.
//
// An MCP server that fails to connect is reported via warnings printed to
// stderr rather than failing the whole build — this mirrors Connect's own
// per-server error isolation: one broken server should not stop Kiwi from
// starting.
func assembleAgent(ctx context.Context, provider llm.Provider, cwd string, mode permission.Mode, broker *permission.Broker, procs *proc.Registry, asker tools.Asker) (*agent.Agent, prompt.Options, *mcp.Manager) {
	projectFile, projectInstructions := config.ProjectInstructions(cwd)

	loadedSkills, err := skills.Load()
	if err != nil {
		fmt.Fprintln(os.Stderr, "kiwi: warning: loading skills:", err)
		loadedSkills = map[string]skills.Skill{}
	}

	mgr, mcpTools, mcpErrs := mcp.Connect(ctx, broker)
	for _, e := range mcpErrs {
		fmt.Fprintln(os.Stderr, "kiwi: warning:", e)
	}

	// Memory is read once per build, not per turn: a rebuild is exactly what
	// happens after the user edits it via /memory, and re-reading two files on
	// every request would buy nothing but syscalls.
	mem := memory.New(cwd)

	promptOpts := prompt.Options{
		WorkingDir:          cwd,
		ProjectFile:         projectFile,
		ProjectInstructions: projectInstructions,
		ModeInstructions:    mode.Instructions(),
		Extra:               []string{mem.Block(), skills.Summary(loadedSkills)},
	}
	// The subagent prompt skips ModeInstructions: a subagent has no
	// permission mode of its own, only the (already restricted, in the
	// explore case) toolset it was handed.
	subagentPromptOpts := promptOpts
	subagentPromptOpts.ModeInstructions = ""

	extraTools := make([]tools.Tool, 0, len(mcpTools)+3)
	for _, t := range mcpTools {
		extraTools = append(extraTools, t)
	}
	if len(loadedSkills) > 0 {
		extraTools = append(extraTools, tools.LoadSkill{Skills: loadedSkills})
	}
	extraTools = append(extraTools,
		tools.BackgroundBash{WorkDir: cwd, Perms: broker, Procs: procs},
		tools.BackgroundOutput{Procs: procs},
		tools.KillShell{Procs: procs},
	)

	fullTools := tools.Default(cwd, broker, extraTools...)

	// generalNames snapshots the parent's toolset *before* task is added to
	// it below — Subset copies into a new Registry, so registering task into
	// fullTools afterward cannot leak into generalTools and hand a subagent
	// the ability to recurse.
	generalNames := make([]string, 0, len(fullTools.Schemas()))
	for _, sc := range fullTools.Schemas() {
		generalNames = append(generalNames, sc.Name)
	}
	generalTools := fullTools.Subset(generalNames...)

	exploreFS := &tools.FS{WorkDir: cwd, Perms: broker}
	exploreTools := tools.NewRegistry(
		tools.ReadFile{FS: exploreFS},
		tools.ReadOnlyBash{Bash: tools.Bash{WorkDir: cwd, Perms: broker}},
	)

	// Registered after the generalTools snapshot above, alongside task and for
	// the same reason: a subagent's context is thrown away when it returns, so
	// nothing it decides is worth writing into the memory every future session
	// pays for. Only the agent the user is actually talking to remembers.
	fullTools.Register(tools.Remember{Store: mem, Perms: broker})

	// Only registered when something can actually show questions to a human
	// — asker is nil for `kiwi ask` and other headless runs. Kept out of
	// generalTools/exploreTools like Remember above: a subagent's context is
	// thrown away when it returns, so blocking on a question only the user
	// (not whoever is waiting on the subagent) can see would be confusing.
	if asker != nil {
		fullTools.Register(tools.AskQuestionsTool{Asker: asker})
	}

	fullTools.Register(agent.TaskTool{
		Provider:     provider,
		System:       prompt.Build(subagentPromptOpts),
		ExploreTools: exploreTools,
		GeneralTools: generalTools,
	})

	a := &agent.Agent{
		Provider: provider,
		Tools:    fullTools,
		System:   prompt.Build(promptOpts),
	}
	return a, promptOpts, mgr
}

// rebuildAgent reconstructs the agent from current on-disk configuration,
// used as the TUI's Options.Rebuild callback after a /model, /mcp, or /skill
// change persists something new. It replaces sess's own MCP manager so old
// server connections are not leaked across a reload.
//
// It always follows cfg.Current rather than any --model override the run
// started with: a /model switch persists a new Current specifically so a
// rebuild picks it up, and re-applying a stale --model here would silently
// undo that switch on the very next unrelated /config or /skill change.
func (s *runSession) rebuildAgent(ctx context.Context) (*agent.Agent, string, error) {
	cfg, err := config.Load()
	if err != nil {
		return nil, "", err
	}
	name, profile, err := cfg.Profile("")
	if err != nil {
		return nil, "", err
	}
	provider, err := config.BuildProvider(ctx, name, profile)
	if err != nil {
		return nil, "", err
	}

	a, _, mgr := assembleAgent(ctx, provider, s.workDir, s.broker.Mode(), s.broker, s.procs, s.asker)

	if s.mcpManager != nil {
		s.mcpManager.Close()
	}
	s.mcpManager = mgr

	return a, fmt.Sprintf("%s/%s", provider.Name(), provider.Model()), nil
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
