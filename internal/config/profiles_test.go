package config

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

func newTestConfig(t *testing.T) *Config {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	cfg := &Config{
		Current:  "sonnet",
		Profiles: map[string]Profile{"sonnet": {Provider: KindAnthropic, Model: "claude-sonnet-5", APIKeyEnv: "ANTHROPIC_API_KEY"}},
		path:     filepath.Join(dir, "kiwi", "kiwi.json"),
	}
	return cfg
}

func TestAddProfilePersists(t *testing.T) {
	cfg := newTestConfig(t)
	if err := cfg.AddProfile("gpt", Profile{Provider: KindOpenAI, Model: "gpt-5.5", APIKeyEnv: "OPENAI_API_KEY"}); err != nil {
		t.Fatalf("AddProfile: %v", err)
	}

	reloaded, err := Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	got, ok := reloaded.Profiles["gpt"]
	if !ok {
		t.Fatal("gpt profile was not persisted")
	}
	if got.Model != "gpt-5.5" {
		t.Errorf("Model = %q", got.Model)
	}
}

func TestAddProfileDuplicateRejected(t *testing.T) {
	cfg := newTestConfig(t)
	err := cfg.AddProfile("sonnet", Profile{Provider: KindAnthropic, Model: "x"})
	if !errors.Is(err, ErrProfileExists) {
		t.Errorf("err = %v, want ErrProfileExists", err)
	}
}

func TestRemoveProfile(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.AddProfile("gpt", Profile{Provider: KindOpenAI, Model: "gpt-5.5"})

	if err := cfg.RemoveProfile("gpt"); err != nil {
		t.Fatalf("RemoveProfile: %v", err)
	}
	if _, exists := cfg.Profiles["gpt"]; exists {
		t.Error("profile still present in memory")
	}

	reloaded, _ := Load()
	if _, exists := reloaded.Profiles["gpt"]; exists {
		t.Error("removal was not persisted")
	}
}

// Removing the profile currently in use would leave Current pointing at
// nothing — the user would have to know to fix kiwi.json by hand.
func TestRemoveCurrentProfileRejected(t *testing.T) {
	cfg := newTestConfig(t)
	err := cfg.RemoveProfile("sonnet")
	if !errors.Is(err, ErrCannotRemoveCurrent) {
		t.Errorf("err = %v, want ErrCannotRemoveCurrent", err)
	}
	if _, exists := cfg.Profiles["sonnet"]; !exists {
		t.Error("the current profile was removed despite the error")
	}
}

func TestRemoveUnknownProfile(t *testing.T) {
	cfg := newTestConfig(t)
	if err := cfg.RemoveProfile("nope"); !errors.Is(err, ErrProfileNotFound) {
		t.Errorf("err = %v, want ErrProfileNotFound", err)
	}
}

func TestSetCurrentPersists(t *testing.T) {
	cfg := newTestConfig(t)
	cfg.AddProfile("gpt", Profile{Provider: KindOpenAI, Model: "gpt-5.5"})

	if err := cfg.SetCurrent("gpt"); err != nil {
		t.Fatalf("SetCurrent: %v", err)
	}
	reloaded, _ := Load()
	if reloaded.Current != "gpt" {
		t.Errorf("Current = %q, want gpt", reloaded.Current)
	}
}

func TestSetCurrentUnknownProfile(t *testing.T) {
	cfg := newTestConfig(t)
	if err := cfg.SetCurrent("nope"); !errors.Is(err, ErrProfileNotFound) {
		t.Errorf("err = %v, want ErrProfileNotFound", err)
	}
}

func TestMaskValue(t *testing.T) {
	cases := map[string]string{
		"":               "",
		"abc":            "***",
		"sk-ant-abcdefg": "sk-...fg",
	}
	for in, want := range cases {
		if got := MaskValue(in); got != want {
			t.Errorf("MaskValue(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestBuildProviderMissingAPIKeyIsErrMissingAPIKey(t *testing.T) {
	profile := Profile{Provider: KindAnthropic, Model: "x", APIKeyEnv: "KIWI_TEST_MISSING_KEY_VAR"}
	os.Unsetenv("KIWI_TEST_MISSING_KEY_VAR")

	_, err := BuildProvider("sonnet", profile)
	if !errors.Is(err, ErrMissingAPIKey) {
		t.Errorf("err = %v, want it to wrap ErrMissingAPIKey", err)
	}
}
