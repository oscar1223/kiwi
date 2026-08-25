// Package config loads Kiwi's configuration: model profiles, API keys and
// per-project instructions.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/anthropic"
	"github.com/oscar1223/kiwi/internal/llm/openai"
)

// ProjectFiles are the per-directory instruction files Kiwi picks up, in
// priority order. AGENTS.md is the cross-tool convention; KIWI.md wins when
// both exist so a project can override shared instructions.
var ProjectFiles = []string{"KIWI.md", "AGENTS.md"}

type ProviderKind string

const (
	// KindAnthropic uses the Anthropic Messages API.
	KindAnthropic ProviderKind = "anthropic"
	// KindOpenAI uses the chat-completions wire format: OpenAI itself, but
	// also Ollama, LM Studio, OpenRouter, Groq and friends.
	KindOpenAI ProviderKind = "openai"
)

type Profile struct {
	Provider ProviderKind `json:"provider"`
	Model    string       `json:"model"`
	// BaseURL overrides the provider endpoint. Required for local runtimes.
	BaseURL string `json:"base_url,omitempty"`
	// APIKeyEnv names the environment variable holding the key. Kiwi never
	// stores keys in its own config file.
	APIKeyEnv string `json:"api_key_env,omitempty"`
}

type Config struct {
	Current  string             `json:"current"`
	Profiles map[string]Profile `json:"profiles"`
	// Theme names the TUI colour theme to use. Empty means the default
	// ("kiwi-dark") — omitted from fresh configs so the JSON stays minimal.
	Theme string `json:"theme,omitempty"`

	// path records where this config was loaded from, for Save.
	path string
}

func Default() *Config {
	return &Config{
		Current: "sonnet",
		Profiles: map[string]Profile{
			"sonnet": {
				Provider:  KindAnthropic,
				Model:     "claude-sonnet-5",
				APIKeyEnv: "ANTHROPIC_API_KEY",
			},
			"opus": {
				Provider:  KindAnthropic,
				Model:     "claude-opus-5",
				APIKeyEnv: "ANTHROPIC_API_KEY",
			},
			"gpt": {
				Provider:  KindOpenAI,
				Model:     "gpt-5.5",
				APIKeyEnv: "OPENAI_API_KEY",
			},
			"local": {
				Provider: KindOpenAI,
				Model:    "qwen3-coder",
				BaseURL:  "http://localhost:11434/v1",
			},
		},
	}
}

// Dir is the configuration directory, honouring XDG_CONFIG_HOME.
func Dir() (string, error) {
	base := os.Getenv("XDG_CONFIG_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".config")
	}
	return filepath.Join(base, "kiwi"), nil
}

// DataDir is where sessions and other state live.
func DataDir() (string, error) {
	base := os.Getenv("XDG_DATA_HOME")
	if base == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return "", err
		}
		base = filepath.Join(home, ".local", "share")
	}
	return filepath.Join(base, "kiwi"), nil
}

// Load reads the config file, returning defaults if it does not exist yet.
func Load() (*Config, error) {
	dir, err := Dir()
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "kiwi.json")

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		cfg := Default()
		cfg.path = path
		return cfg, nil
	}
	if err != nil {
		return nil, err
	}

	cfg := &Config{}
	if err := json.Unmarshal(data, cfg); err != nil {
		return nil, fmt.Errorf("config %s: %w", path, err)
	}
	cfg.path = path
	if cfg.Profiles == nil {
		cfg.Profiles = map[string]Profile{}
	}
	return cfg, nil
}

func (c *Config) Save() error {
	if c.path == "" {
		dir, err := Dir()
		if err != nil {
			return err
		}
		c.path = filepath.Join(dir, "kiwi.json")
	}
	if err := os.MkdirAll(filepath.Dir(c.path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(c.path, append(data, '\n'), 0o644)
}

// Profile returns the named profile, or the current one when name is empty.
func (c *Config) Profile(name string) (string, Profile, error) {
	if name == "" {
		name = c.Current
	}
	p, ok := c.Profiles[name]
	if !ok {
		return "", Profile{}, fmt.Errorf("unknown model profile %q (available: %s)", name, strings.Join(c.ProfileNames(), ", "))
	}
	return name, p, nil
}

func (c *Config) ProfileNames() []string {
	names := make([]string, 0, len(c.Profiles))
	for n := range c.Profiles {
		names = append(names, n)
	}
	return names
}

// ErrMissingAPIKey means a profile's required environment variable is unset —
// the one condition callers may want to treat as "run the onboarding wizard"
// rather than a hard failure, since it is exactly what a brand-new install
// looks like before any provider has been configured.
var ErrMissingAPIKey = errors.New("config: missing API key")

// BuildProvider instantiates the provider for a profile.
func BuildProvider(name string, p Profile) (llm.Provider, error) {
	var apiKey string
	if p.APIKeyEnv != "" {
		apiKey = os.Getenv(p.APIKeyEnv)
		if apiKey == "" {
			return nil, fmt.Errorf("%w: profile %q needs %s to be set in the environment", ErrMissingAPIKey, name, p.APIKeyEnv)
		}
	}

	switch p.Provider {
	case KindAnthropic:
		return anthropic.New(anthropic.Options{
			APIKey:  apiKey,
			BaseURL: p.BaseURL,
			Model:   p.Model,
		}), nil
	case KindOpenAI:
		// Local runtimes ignore the key but the SDK requires a non-empty one.
		if apiKey == "" && p.BaseURL != "" {
			apiKey = "not-needed"
		}
		return openai.New(openai.Options{
			APIKey:  apiKey,
			BaseURL: p.BaseURL,
			Model:   p.Model,
			Name:    name,
		}), nil
	default:
		return nil, fmt.Errorf("profile %q: unknown provider %q", name, p.Provider)
	}
}

// ProjectInstructions returns the contents of the first project instruction
// file found in dir, or "" when there is none.
func ProjectInstructions(dir string) (string, string) {
	for _, name := range ProjectFiles {
		path := filepath.Join(dir, name)
		data, err := os.ReadFile(path)
		if err == nil && len(data) > 0 {
			return name, string(data)
		}
	}
	return "", ""
}

// Profile management errors.
var (
	ErrProfileExists       = errors.New("config: profile already exists")
	ErrProfileNotFound     = errors.New("config: profile not found")
	ErrCannotRemoveCurrent = errors.New("config: cannot remove the current profile")
)

// AddProfile saves a new named profile and persists the config.
func (c *Config) AddProfile(name string, p Profile) error {
	if strings.TrimSpace(name) == "" {
		return errors.New("config: profile name is required")
	}
	if _, exists := c.Profiles[name]; exists {
		return fmt.Errorf("%w: %q", ErrProfileExists, name)
	}
	if c.Profiles == nil {
		c.Profiles = map[string]Profile{}
	}
	c.Profiles[name] = p
	return c.Save()
}

// RemoveProfile deletes a profile and persists the config. The current
// profile cannot be removed — switch away from it first.
func (c *Config) RemoveProfile(name string) error {
	if name == c.Current {
		return fmt.Errorf("%w: %q", ErrCannotRemoveCurrent, name)
	}
	if _, exists := c.Profiles[name]; !exists {
		return fmt.Errorf("%w: %q", ErrProfileNotFound, name)
	}
	delete(c.Profiles, name)
	return c.Save()
}

// SetCurrent switches the default profile and persists the config.
func (c *Config) SetCurrent(name string) error {
	if _, exists := c.Profiles[name]; !exists {
		return fmt.Errorf("%w: %q", ErrProfileNotFound, name)
	}
	c.Current = name
	return c.Save()
}

// MaskValue renders a secret for display: enough to recognise it, not enough
// to reconstruct it from a screenshot.
func MaskValue(v string) string {
	if len(v) <= 6 {
		return strings.Repeat("*", len(v))
	}
	return v[:3] + "..." + v[len(v)-2:]
}
