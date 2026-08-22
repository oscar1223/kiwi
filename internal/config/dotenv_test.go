package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// withConfigDir points Dir() at a temp directory for the duration of the
// test, via XDG_CONFIG_HOME, and restores the environment afterwards.
func withConfigDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	return filepath.Join(dir, "kiwi")
}

func writeDotEnv(t *testing.T, configDir, content string) {
	t.Helper()
	if err := os.MkdirAll(configDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configDir, ".env"), []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestLoadDotEnvSetsVariables(t *testing.T) {
	configDir := withConfigDir(t)
	writeDotEnv(t, configDir, "ANTHROPIC_API_KEY=sk-ant-test123\nOPENAI_API_KEY=sk-test456\n")
	t.Setenv("ANTHROPIC_API_KEY", "")
	os.Unsetenv("ANTHROPIC_API_KEY")
	os.Unsetenv("OPENAI_API_KEY")

	if err := LoadDotEnv(); err != nil {
		t.Fatalf("LoadDotEnv: %v", err)
	}
	if got := os.Getenv("ANTHROPIC_API_KEY"); got != "sk-ant-test123" {
		t.Errorf("ANTHROPIC_API_KEY = %q", got)
	}
	if got := os.Getenv("OPENAI_API_KEY"); got != "sk-test456" {
		t.Errorf("OPENAI_API_KEY = %q", got)
	}
}

// The shell's own environment must win: a key already exported must not be
// silently replaced by whatever is sitting in the file.
func TestLoadDotEnvDoesNotOverrideExistingEnv(t *testing.T) {
	configDir := withConfigDir(t)
	writeDotEnv(t, configDir, "ANTHROPIC_API_KEY=from-file\n")
	t.Setenv("ANTHROPIC_API_KEY", "from-shell")

	if err := LoadDotEnv(); err != nil {
		t.Fatalf("LoadDotEnv: %v", err)
	}
	if got := os.Getenv("ANTHROPIC_API_KEY"); got != "from-shell" {
		t.Errorf("ANTHROPIC_API_KEY = %q, want the shell's value preserved", got)
	}
}

func TestLoadDotEnvMissingFileIsNotAnError(t *testing.T) {
	withConfigDir(t) // directory exists, no .env inside it
	if err := LoadDotEnv(); err != nil {
		t.Errorf("a missing .env should not be an error, got %v", err)
	}
}

func TestLoadDotEnvHandlesCommentsQuotesAndExport(t *testing.T) {
	configDir := withConfigDir(t)
	writeDotEnv(t, configDir, strings.Join([]string{
		"# a comment",
		"",
		`export FOO="bar baz"`,
		"BARE=plain",
		"SINGLE='quoted'",
		"   ", // whitespace-only line
	}, "\n"))
	os.Unsetenv("FOO")
	os.Unsetenv("BARE")
	os.Unsetenv("SINGLE")

	if err := LoadDotEnv(); err != nil {
		t.Fatalf("LoadDotEnv: %v", err)
	}
	cases := map[string]string{"FOO": "bar baz", "BARE": "plain", "SINGLE": "quoted"}
	for k, want := range cases {
		if got := os.Getenv(k); got != want {
			t.Errorf("%s = %q, want %q", k, got, want)
		}
	}
}

func TestLoadDotEnvIgnoresMalformedLines(t *testing.T) {
	configDir := withConfigDir(t)
	writeDotEnv(t, configDir, "not-a-valid-line\nOK=value\n")
	os.Unsetenv("OK")

	if err := LoadDotEnv(); err != nil {
		t.Fatalf("LoadDotEnv: %v", err)
	}
	if got := os.Getenv("OK"); got != "value" {
		t.Errorf("OK = %q", got)
	}
}
