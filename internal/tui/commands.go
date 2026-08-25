package tui

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/mcp"
	"github.com/oscar1223/kiwi/internal/skills"
)

const newOptionValue = "__new__"

// --- onboarding ---

// onboardingFlow runs automatically, once, on a fresh install with no model
// provider configured — see runSession.needsOnboarding in cmd/kiwi. It is
// the same Pick/Text/SecretText machinery every other flow uses; the only
// thing that makes it "onboarding" is that Init starts it on its own instead
// of waiting for the user to type a command.
//
// Deliberately minimal: provider, model (defaulted, rarely needs typing),
// API key. Anything more advanced — a custom base URL, Ollama, a second
// profile — stays in /model, which this does not try to replace.
func (m *Model) onboardingFlow(ctx context.Context) {
	m.events.send(ctx, printLinesMsg{lines: []string{
		styleKiwi.Render("🥝 welcome to kiwi"),
		styleDim.Render("  no model provider is configured yet — let's fix that."),
	}})

	skip := func() {
		m.events.send(ctx, systemMsg{"Setup skipped. Run kiwi again, or use /model, whenever you're ready."})
	}

	providerChoice, ok := m.events.Pick(ctx, "Which provider do you want to use?", []pickOption{
		{"Anthropic (Claude)", string(config.KindAnthropic)},
		{"OpenAI-compatible (OpenAI, Ollama, OpenRouter, Groq, ...)", string(config.KindOpenAI)},
	})
	if !ok {
		skip()
		return
	}
	provider := config.ProviderKind(providerChoice)

	var profileName, defaultModel, apiKeyEnv, keyHint string
	if provider == config.KindAnthropic {
		profileName, defaultModel, apiKeyEnv, keyHint =
			"sonnet", "claude-sonnet-5", "ANTHROPIC_API_KEY", "https://console.anthropic.com/settings/keys"
	} else {
		profileName, defaultModel, apiKeyEnv, keyHint =
			"gpt", "gpt-5.5", "OPENAI_API_KEY", "https://platform.openai.com/api-keys"
	}

	modelName, ok := m.events.Text(ctx, "Model name:", "", defaultModel)
	if !ok {
		skip()
		return
	}
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		modelName = defaultModel
	}

	m.events.send(ctx, systemMsg{"Get an API key: " + keyHint})
	apiKey, ok := m.events.SecretText(ctx, "Paste your "+apiKeyEnv+":", "")
	apiKey = strings.TrimSpace(apiKey)
	if !ok || apiKey == "" {
		skip()
		return
	}

	envFile, err := config.OpenEnvFile()
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	if err := envFile.Set(apiKeyEnv, apiKey); err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	// So the rebuild below sees it immediately, without waiting for a
	// process restart to reload .env.
	os.Setenv(apiKeyEnv, apiKey)

	cfg, err := config.Load()
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	// An upsert, not AddProfile's strict "must be new": config.Default()
	// already seeds "sonnet"/"gpt" as placeholder profiles in memory on a
	// fresh install (no kiwi.json on disk yet), so AddProfile would reject
	// this as a duplicate. Onboarding's intent is "this profile now has
	// these settings and is current" regardless of whether the name already
	// existed as a placeholder.
	if cfg.Profiles == nil {
		cfg.Profiles = map[string]config.Profile{}
	}
	cfg.Profiles[profileName] = config.Profile{Provider: provider, Model: modelName, APIKeyEnv: apiKeyEnv}
	if err := cfg.SetCurrent(profileName); err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}

	m.applyRebuild(ctx)
	m.events.send(ctx, systemMsg{"Setup complete. Type something, or /help to see what Kiwi can do."})
}

// --- /theme ---

// themeFlow lets the user browse themes with a live preview: onHighlight
// applies each theme as soon as it is highlighted (not just once confirmed),
// so the picker itself doubles as the preview. Cancelling restores whatever
// theme was active before the picker opened; confirming persists the choice.
//
// Every applyTheme call here happens through onHighlight/onCancel, which
// onPickKey invokes from Update's own goroutine — never directly in this
// function, which runs in the flow's separate goroutine and would race
// View() if it touched the shared style vars itself.
func (m *Model) themeFlow(ctx context.Context) {
	cfg, err := config.Load()
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	before, _ := themeByName(cfg.Theme)

	options := make([]pickOption, len(themes))
	for i, t := range themes {
		marker := "  "
		if t.Name == before.Name {
			marker = "→ "
		}
		options[i] = pickOption{marker + t.Name, t.Name}
	}

	choice, ok := m.events.PickWithPreview(ctx, "Theme", options,
		func(name string) {
			if t, found := themeByName(name); found {
				applyTheme(t)
			}
		},
		func() { applyTheme(before) },
	)
	if !ok {
		return
	}

	// The confirmed choice is already live: it was either the theme the
	// picker opened with (untouched), or the last one onHighlight applied
	// while browsing to it. Only persistence is left to do.
	t, found := themeByName(choice)
	if !found {
		return
	}
	cfg.Theme = t.Name
	if err := cfg.Save(); err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	m.events.send(ctx, systemMsg{"Theme set to " + t.Name + "."})
}

// --- /sessions ---

// sessionsFlow lets the user browse and switch between saved conversations
// for the current project, or start a fresh one, without leaving the TUI.
// Switching away from the current session never loses anything: every turn
// is already persisted immediately (session.Persist, via persistTurn), so
// there is no "pending" state a jump could drop.
func (m *Model) sessionsFlow(ctx context.Context) {
	store := m.opts.Store
	if store == nil {
		m.events.send(ctx, systemMsg{"Sessions are not persisted in this run — nothing to browse."})
		return
	}

	sessions, err := store.List(ctx, m.opts.WorkDir, 50)
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}

	options := make([]pickOption, 0, len(sessions)+1)
	for _, s := range sessions {
		title := s.Title
		if title == "" {
			title = "(untitled)"
		}
		marker := "  "
		if s.ID == m.opts.SessionID {
			marker = "→ "
		}
		label := fmt.Sprintf("%s%s  %-8s  %s", marker, s.ID, relativeTime(s.UpdatedAt), title)
		options = append(options, pickOption{label, s.ID})
	}
	options = append(options, pickOption{"+ New session", newOptionValue})

	choice, ok := m.events.Pick(ctx, "Sessions", options)
	if !ok || choice == m.opts.SessionID {
		return
	}

	if choice == newOptionValue {
		meta, err := store.Create(ctx, m.opts.WorkDir)
		if err != nil {
			m.events.send(ctx, errMsg{err})
			return
		}
		m.events.send(ctx, sessionSwitchedMsg{sessionID: meta.ID, title: "new session"})
		return
	}

	history, err := store.Load(ctx, choice)
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	title := "(untitled)"
	for _, s := range sessions {
		if s.ID == choice && s.Title != "" {
			title = s.Title
		}
	}
	m.events.send(ctx, sessionSwitchedMsg{sessionID: choice, history: history, title: title})
}

// --- /settings ---

// settingsFlow groups every settings area behind one entry point, for anyone
// who would rather browse a menu than remember five separate command names.
// It is pure composition: each choice just calls the flow that already
// implements that area, so there is no new logic to keep in sync with them.
func (m *Model) settingsFlow(ctx context.Context) {
	for {
		choice, ok := m.events.Pick(ctx, "Settings", []pickOption{
			{"Model profiles", "model"},
			{"Environment (.env)", "config"},
			{"MCP servers", "mcp"},
			{"Skills", "skill"},
			{"Theme", "theme"},
			{"Sessions", "sessions"},
			{"Close", "close"},
		})
		if !ok || choice == "close" {
			return
		}
		switch choice {
		case "model":
			m.modelFlow(ctx)
		case "config":
			m.configFlow(ctx)
		case "mcp":
			m.mcpFlow(ctx)
		case "theme":
			m.themeFlow(ctx)
		case "sessions":
			m.sessionsFlow(ctx)
		case "skill":
			m.skillFlow(ctx)
		}
	}
}

// --- /memory ---

// memoryFlow views or clears the conversation. snapshot is a copy taken at
// the moment /memory was typed: flows run on their own goroutine and must
// never read Model state directly once launched.
func (m *Model) memoryFlow(ctx context.Context, snapshot []llm.Message) {
	for {
		title := fmt.Sprintf("Memory — %d messages", len(snapshot))
		choice, ok := m.events.Pick(ctx, title, []pickOption{
			{"View recent messages", "view"},
			{"Clear memory", "clear"},
			{"Close", "close"},
		})
		if !ok || choice == "close" {
			return
		}

		switch choice {
		case "view":
			if len(snapshot) == 0 {
				m.events.send(ctx, systemMsg{"No messages saved yet."})
				continue
			}
			raw, ok := m.events.Text(ctx, "How many recent messages?", "", "5")
			if !ok {
				continue
			}
			n, err := strconv.Atoi(strings.TrimSpace(raw))
			if err != nil || n <= 0 {
				n = 5
			}
			start := len(snapshot) - n
			if start < 0 {
				start = 0
			}
			m.events.send(ctx, printLinesMsg{lines: renderHistoryLines(snapshot[start:])})

		case "clear":
			if m.events.Confirm(ctx, "Clear all conversation memory? This cannot be undone.") {
				m.events.send(ctx, clearHistoryMsg{})
			}
			return
		}
	}
}

func renderHistoryLines(msgs []llm.Message) []string {
	var lines []string
	for _, msg := range msgs {
		switch msg.Role {
		case llm.RoleUser:
			if msg.Content != "" {
				lines = append(lines, bullet(styleUser.Render(">"), styleUser.Render(msg.Content)))
			}
		case llm.RoleAssistant:
			if msg.Content != "" {
				lines = append(lines, bullet(styleKiwi.Render("●"), styleKiwi.Render(msg.Content)))
			}
		}
	}
	if len(lines) == 0 {
		lines = []string{styleDim.Render("  (nothing to show)")}
	}
	return lines
}

// --- /config ---

// configFlow manages the .env file kiwi loads API keys from.
func (m *Model) configFlow(ctx context.Context) {
	for {
		f, err := config.OpenEnvFile()
		if err != nil {
			m.events.send(ctx, errMsg{err})
			return
		}

		keys := f.Keys()
		options := make([]pickOption, 0, len(keys)+1)
		for _, k := range keys {
			v, _ := f.Get(k)
			options = append(options, pickOption{fmt.Sprintf("%s = %s", k, config.MaskValue(v)), k})
		}
		options = append(options, pickOption{"+ New variable", newOptionValue})

		choice, ok := m.events.Pick(ctx, "Environment variables (.env)", options)
		if !ok {
			return
		}

		key := choice
		if choice == newOptionValue {
			name, ok := m.events.Text(ctx, "Variable name:", "e.g. QWEN_API_KEY", "")
			if !ok || strings.TrimSpace(name) == "" {
				continue
			}
			key = strings.TrimSpace(name)
		}

		cur, _ := f.Get(key)
		action, ok := m.events.Pick(ctx, fmt.Sprintf("%s = %s", key, config.MaskValue(cur)), []pickOption{
			{"Edit value", "edit"},
			{"Delete", "delete"},
			{"Cancel", "cancel"},
		})
		if !ok || action == "cancel" {
			continue
		}

		if action == "delete" {
			if err := f.Unset(key); err != nil {
				m.events.send(ctx, errMsg{err})
				continue
			}
			os.Unsetenv(key)
			m.events.send(ctx, systemMsg{key + " removed from .env."})
		} else {
			value, ok := m.events.Text(ctx, "New value for "+key+":", "", cur)
			if !ok {
				continue
			}
			value = strings.TrimSpace(value)
			if err := f.Set(key, value); err != nil {
				m.events.send(ctx, errMsg{err})
				continue
			}
			// So a rebuild right now already sees it, without waiting for a
			// process restart to reload .env.
			os.Setenv(key, value)
			m.events.send(ctx, systemMsg{key + " updated."})
		}

		m.events.send(ctx, requestRebuildMsg{})
	}
}

// --- /model ---

func (m *Model) modelFlow(ctx context.Context) {
	for {
		cfg, err := config.Load()
		if err != nil {
			m.events.send(ctx, errMsg{err})
			return
		}
		names := sortedProfileNames(cfg)

		options := make([]pickOption, 0, len(names)+2)
		for _, name := range names {
			p := cfg.Profiles[name]
			marker := "  "
			if name == cfg.Current {
				marker = "→ "
			}
			warn := ""
			if p.APIKeyEnv != "" && os.Getenv(p.APIKeyEnv) == "" {
				warn = "  ⚠ missing " + p.APIKeyEnv
			}
			label := fmt.Sprintf("%s%s — %s/%s%s", marker, name, p.Provider, p.Model, warn)
			options = append(options, pickOption{label, name})
		}
		options = append(options, pickOption{"+ New model profile", newOptionValue})
		if len(names) > 1 {
			options = append(options, pickOption{"Delete a profile", "__delete__"})
		}

		choice, ok := m.events.Pick(ctx, "Model profile — current: "+cfg.Current, options)
		if !ok {
			return
		}

		switch choice {
		case newOptionValue:
			m.newModelProfileFlow(ctx, cfg)
			continue

		case "__delete__":
			deletable := make([]pickOption, 0, len(names)-1)
			for _, name := range names {
				if name != cfg.Current {
					deletable = append(deletable, pickOption{name, name})
				}
			}
			target, ok := m.events.Pick(ctx, "Delete which profile?", deletable)
			if !ok {
				continue
			}
			if err := cfg.RemoveProfile(target); err != nil {
				m.events.send(ctx, errMsg{err})
			} else {
				m.events.send(ctx, systemMsg{fmt.Sprintf("Profile %q removed.", target)})
			}
			continue

		default:
			if err := cfg.SetCurrent(choice); err != nil {
				m.events.send(ctx, errMsg{err})
				continue
			}
			m.events.send(ctx, requestRebuildMsg{})
			return
		}
	}
}

func (m *Model) newModelProfileFlow(ctx context.Context, cfg *config.Config) {
	providerChoice, ok := m.events.Pick(ctx, "Provider", []pickOption{
		{"Anthropic", string(config.KindAnthropic)},
		{"OpenAI-compatible (OpenAI, Ollama, OpenRouter, Groq, ...)", string(config.KindOpenAI)},
	})
	if !ok {
		return
	}
	provider := config.ProviderKind(providerChoice)

	modelName, ok := m.events.Text(ctx, "Model name:", "e.g. gpt-5.5, claude-opus-5, qwen3-coder", "")
	if !ok {
		return
	}
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		m.events.send(ctx, errMsg{fmt.Errorf("a model name is required")})
		return
	}

	var baseURL, apiKeyEnv string
	if provider == config.KindOpenAI {
		u, ok := m.events.Text(ctx, "Base URL (empty for OpenAI itself):", "https://api.deepseek.com/v1", "")
		if !ok {
			return
		}
		baseURL = strings.TrimSpace(u)
		if baseURL != "" {
			e, ok := m.events.Text(ctx, "Environment variable for its API key:", "DEEPSEEK_API_KEY", "")
			if !ok {
				return
			}
			apiKeyEnv = strings.TrimSpace(e)
		} else {
			apiKeyEnv = "OPENAI_API_KEY"
		}
	} else {
		apiKeyEnv = "ANTHROPIC_API_KEY"
	}

	alias, ok := m.events.Text(ctx, "Alias for this profile:", "", modelName)
	if !ok {
		return
	}
	alias = strings.TrimSpace(alias)
	if alias == "" {
		alias = modelName
	}

	if err := cfg.AddProfile(alias, config.Profile{
		Provider: provider, Model: modelName, BaseURL: baseURL, APIKeyEnv: apiKeyEnv,
	}); err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	m.events.send(ctx, systemMsg{fmt.Sprintf("Profile %q saved.", alias)})

	if m.events.Confirm(ctx, fmt.Sprintf("Switch to %q now?", alias)) {
		if err := cfg.SetCurrent(alias); err != nil {
			m.events.send(ctx, errMsg{err})
			return
		}
		m.events.send(ctx, requestRebuildMsg{})
	}
}

func sortedProfileNames(cfg *config.Config) []string {
	names := make([]string, 0, len(cfg.Profiles))
	for n := range cfg.Profiles {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// --- /skill ---

func (m *Model) skillFlow(ctx context.Context) {
	for {
		sk, err := skills.Load()
		if err != nil {
			m.events.send(ctx, errMsg{err})
			return
		}
		names := make([]string, 0, len(sk))
		for n := range sk {
			names = append(names, n)
		}
		sort.Strings(names)

		options := make([]pickOption, 0, len(names)+1)
		for _, n := range names {
			options = append(options, pickOption{n + ": " + sk[n].Description, n})
		}
		options = append(options, pickOption{"+ New skill", newOptionValue})

		choice, ok := m.events.Pick(ctx, "Skills", options)
		if !ok {
			return
		}

		if choice == newOptionValue {
			m.newSkillFlow(ctx)
			continue
		}

		action, ok := m.events.Pick(ctx, "Skill '"+choice+"'", []pickOption{
			{"View content", "view"},
			{"Delete", "delete"},
			{"Cancel", "cancel"},
		})
		if !ok || action == "cancel" {
			continue
		}

		switch action {
		case "view":
			body := sk[choice].Body
			m.events.send(ctx, printLinesMsg{lines: []string{
				styleDim.Render("--- " + choice + " ---"),
				body,
			}})
		case "delete":
			if m.events.Confirm(ctx, "Delete the skill '"+choice+"'?") {
				if err := skills.Delete(choice); err != nil {
					m.events.send(ctx, errMsg{err})
				} else {
					m.events.send(ctx, systemMsg{"Skill '" + choice + "' deleted."})
					m.events.send(ctx, requestRebuildMsg{})
				}
			}
		}
	}
}

func (m *Model) newSkillFlow(ctx context.Context) {
	name, ok := m.events.Text(ctx, "Skill name (kebab-case):", "my-skill", "")
	if !ok || strings.TrimSpace(name) == "" {
		return
	}
	name = strings.TrimSpace(name)

	description, ok := m.events.Text(ctx, "Short description (when to use it):", "", "")
	if !ok {
		return
	}

	raw, ok := m.events.Text(ctx, "Content: path to an existing .md to import, or text directly:", "", "")
	if !ok {
		return
	}

	body := raw
	if expanded, err := expandHome(strings.TrimSpace(raw)); err == nil {
		if data, err := os.ReadFile(expanded); err == nil {
			body = string(data)
		}
	}

	path, err := skills.Save(name, strings.TrimSpace(description), body)
	if err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	m.events.send(ctx, systemMsg{"Skill saved to " + path + "."})
	m.events.send(ctx, requestRebuildMsg{})
}

func expandHome(path string) (string, error) {
	if !strings.HasPrefix(path, "~") {
		return path, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(home, strings.TrimPrefix(path, "~")), nil
}

// --- /mcp ---

func (m *Model) mcpFlow(ctx context.Context) {
	for {
		cfg, err := mcp.LoadConfig()
		if err != nil {
			m.events.send(ctx, errMsg{err})
			return
		}
		names := make([]string, 0, len(cfg))
		for n := range cfg {
			names = append(names, n)
		}
		sort.Strings(names)

		options := make([]pickOption, 0, len(names)+1)
		for _, n := range names {
			options = append(options, pickOption{n + " — " + describeServer(cfg[n]), n})
		}
		options = append(options, pickOption{"+ Add MCP server", newOptionValue})

		choice, ok := m.events.Pick(ctx, "MCP servers", options)
		if !ok {
			return
		}

		if choice == newOptionValue {
			m.newMCPServerFlow(ctx)
			continue
		}

		action, ok := m.events.Pick(ctx, "Server '"+choice+"'", []pickOption{
			{"Remove", "remove"},
			{"Cancel", "cancel"},
		})
		if !ok || action == "cancel" {
			continue
		}
		if action == "remove" && m.events.Confirm(ctx, "Remove the MCP server '"+choice+"'?") {
			if err := mcp.RemoveServer(choice); err != nil {
				m.events.send(ctx, errMsg{err})
			} else {
				m.events.send(ctx, systemMsg{"Server '" + choice + "' removed."})
				m.events.send(ctx, requestRebuildMsg{})
			}
		}
	}
}

// describeServer renders a one-line summary of a server for the picker list.
func describeServer(sc mcp.ServerConfig) string {
	if sc.IsRemote() {
		typ := sc.Type
		if typ == "" {
			typ = mcp.TransportHTTP
		}
		return string(typ) + " " + sc.URL
	}
	return strings.TrimSpace(sc.Command + " " + strings.Join(sc.Args, " "))
}

func (m *Model) newMCPServerFlow(ctx context.Context) {
	name, ok := m.events.Text(ctx, "Server name:", "", "")
	if !ok || strings.TrimSpace(name) == "" {
		return
	}
	name = strings.TrimSpace(name)

	kind, ok := m.events.Pick(ctx, "How does it connect?", []pickOption{
		{"Local command (stdio)", "stdio"},
		{"Remote URL (HTTP or SSE)", "remote"},
	})
	if !ok {
		return
	}

	var sc mcp.ServerConfig
	if kind == "stdio" {
		sc, ok = m.newStdioServerConfig(ctx)
	} else {
		sc, ok = m.newRemoteServerConfig(ctx)
	}
	if !ok {
		return
	}

	if err := mcp.AddServer(name, sc); err != nil {
		m.events.send(ctx, errMsg{err})
		return
	}
	m.events.send(ctx, systemMsg{"Server '" + name + "' saved. Reconnecting…"})
	m.events.send(ctx, requestRebuildMsg{})
}

func (m *Model) newStdioServerConfig(ctx context.Context) (mcp.ServerConfig, bool) {
	command, ok := m.events.Text(ctx, "Command to run:", "npx", "")
	if !ok || strings.TrimSpace(command) == "" {
		return mcp.ServerConfig{}, false
	}

	argsRaw, ok := m.events.Text(ctx, "Arguments, space-separated:", "-y @modelcontextprotocol/server-filesystem /path", "")
	if !ok {
		return mcp.ServerConfig{}, false
	}

	envRaw, ok := m.events.Text(ctx, "Environment variables (optional):", "KEY=value,KEY2=value2", "")
	if !ok {
		return mcp.ServerConfig{}, false
	}

	var args []string
	if strings.TrimSpace(argsRaw) != "" {
		args = strings.Fields(argsRaw)
	}

	return mcp.ServerConfig{
		Command: strings.TrimSpace(command),
		Args:    args,
		Env:     parsePairs(envRaw),
	}, true
}

func (m *Model) newRemoteServerConfig(ctx context.Context) (mcp.ServerConfig, bool) {
	url, ok := m.events.Text(ctx, "Server URL:", "https://example.com/mcp", "")
	if !ok || strings.TrimSpace(url) == "" {
		return mcp.ServerConfig{}, false
	}

	transportChoice, ok := m.events.Pick(ctx, "Transport", []pickOption{
		{"Streamable HTTP (current spec, the common case)", string(mcp.TransportHTTP)},
		{"SSE (older spec — only if the server needs it)", string(mcp.TransportSSE)},
	})
	if !ok {
		return mcp.ServerConfig{}, false
	}

	headersRaw, ok := m.events.Text(ctx, "Headers, e.g. an auth token (optional):", "Authorization=Bearer sk-...", "")
	if !ok {
		return mcp.ServerConfig{}, false
	}

	typ := mcp.TransportKind(transportChoice)
	if typ == mcp.TransportHTTP {
		typ = "" // the zero value already defaults to HTTP; keep the saved config minimal
	}

	return mcp.ServerConfig{
		URL:     strings.TrimSpace(url),
		Type:    typ,
		Headers: parsePairs(headersRaw),
	}, true
}

// parsePairs turns "KEY=value,KEY2=value2" into a map, used for both env
// vars and HTTP headers. An empty or malformed input yields a nil map.
func parsePairs(raw string) map[string]string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	out := map[string]string{}
	for _, pair := range strings.Split(raw, ",") {
		k, v, ok := strings.Cut(pair, "=")
		if ok {
			out[strings.TrimSpace(k)] = strings.TrimSpace(v)
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
