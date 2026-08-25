package config

import (
	"os"
	"path/filepath"
	"testing"
)

func newTestEnvFile(t *testing.T) (*EnvFile, string) {
	t.Helper()
	dir := t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", dir)
	f, err := OpenEnvFile()
	if err != nil {
		t.Fatalf("OpenEnvFile: %v", err)
	}
	return f, filepath.Join(dir, "kiwi", ".env")
}

func TestOpenEnvFileMissingIsEmpty(t *testing.T) {
	f, _ := newTestEnvFile(t)
	if len(f.Keys()) != 0 {
		t.Errorf("Keys() = %v, want empty", f.Keys())
	}
}

func TestSetCreatesFileWithRestrictedPermissions(t *testing.T) {
	f, path := newTestEnvFile(t)
	if err := f.Set("ANTHROPIC_API_KEY", "sk-ant-test"); err != nil {
		t.Fatalf("Set: %v", err)
	}

	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("Stat: %v", err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Errorf("permissions = %o, want 0600 (this file holds secrets)", perm)
	}
}

func TestSetThenGetRoundTrip(t *testing.T) {
	f, _ := newTestEnvFile(t)
	f.Set("FOO", "bar")

	v, ok := f.Get("FOO")
	if !ok || v != "bar" {
		t.Errorf("Get(FOO) = (%q, %v)", v, ok)
	}
}

func TestSetUpdatesInPlaceWithoutReordering(t *testing.T) {
	f, path := newTestEnvFile(t)
	f.Set("A", "1")
	f.Set("B", "2")
	f.Set("A", "updated")

	data, _ := os.ReadFile(path)
	got := string(data)
	wantOrder := []string{"A=updated", "B=2"}
	posA := indexOf(got, wantOrder[0])
	posB := indexOf(got, wantOrder[1])
	if posA < 0 || posB < 0 {
		t.Fatalf("file content = %q, missing expected lines", got)
	}
	if posA > posB {
		t.Errorf("A moved after B on update; order should be preserved:\n%s", got)
	}
}

func TestUnsetRemovesOnlyThatKey(t *testing.T) {
	f, _ := newTestEnvFile(t)
	f.Set("KEEP", "1")
	f.Set("DROP", "2")

	if err := f.Unset("DROP"); err != nil {
		t.Fatalf("Unset: %v", err)
	}
	if _, ok := f.Get("DROP"); ok {
		t.Error("DROP still present")
	}
	if v, ok := f.Get("KEEP"); !ok || v != "1" {
		t.Errorf("KEEP = (%q, %v), want (1, true)", v, ok)
	}
}

func TestUnsetMissingKeyIsNotAnError(t *testing.T) {
	f, _ := newTestEnvFile(t)
	if err := f.Unset("NEVER_SET"); err != nil {
		t.Errorf("Unset of a missing key returned %v, want nil", err)
	}
}

// Comments and blank lines are part of what a human wrote in this file;
// editing one variable must not silently strip them.
func TestCommentsAndBlankLinesSurviveAnEdit(t *testing.T) {
	f, path := newTestEnvFile(t)
	os.MkdirAll(filepath.Dir(path), 0o755)
	os.WriteFile(path, []byte("# a comment\n\nFOO=old\n"), 0o600)

	f, err := OpenEnvFile()
	if err != nil {
		t.Fatalf("OpenEnvFile: %v", err)
	}
	f.Set("FOO", "new")

	data, _ := os.ReadFile(path)
	got := string(data)
	if indexOf(got, "# a comment") < 0 {
		t.Errorf("comment was dropped:\n%s", got)
	}
	if indexOf(got, "FOO=new") < 0 {
		t.Errorf("value was not updated:\n%s", got)
	}
}

func TestKeysPreservesFileOrder(t *testing.T) {
	f, _ := newTestEnvFile(t)
	f.Set("Z", "1")
	f.Set("A", "2")
	f.Set("M", "3")

	got := f.Keys()
	want := []string{"Z", "A", "M"}
	if len(got) != len(want) {
		t.Fatalf("Keys() = %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("Keys()[%d] = %q, want %q (order should follow the file, not sort)", i, got[i], want[i])
		}
	}
}

func TestParseEnvLineHandlesQuotesAndExport(t *testing.T) {
	cases := []struct {
		line      string
		wantKey   string
		wantValue string
		wantOK    bool
	}{
		{`FOO=bar`, "FOO", "bar", true},
		{`export FOO=bar`, "FOO", "bar", true},
		{`FOO="bar baz"`, "FOO", "bar baz", true},
		{`FOO='bar'`, "FOO", "bar", true},
		{`# comment`, "", "", false},
		{``, "", "", false},
		{`not a valid line`, "", "", false},
	}
	for _, c := range cases {
		k, v, ok := parseEnvLine(c.line)
		if ok != c.wantOK || k != c.wantKey || v != c.wantValue {
			t.Errorf("parseEnvLine(%q) = (%q, %q, %v), want (%q, %q, %v)",
				c.line, k, v, ok, c.wantKey, c.wantValue, c.wantOK)
		}
	}
}

func indexOf(s, substr string) int {
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
