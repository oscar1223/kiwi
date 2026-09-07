package skills

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func skillDir(t *testing.T) string {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	dir, err := Dir()
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestSeedInstallsTheBuiltInSkills(t *testing.T) {
	dir := skillDir(t)

	written, err := Seed()
	if err != nil {
		t.Fatalf("Seed: %v", err)
	}
	if len(written) == 0 {
		t.Fatal("Seed installed nothing")
	}

	loaded, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"plan", "commit", "code-review", "init", "security-review", "test"} {
		sk, ok := loaded[want]
		if !ok {
			t.Errorf("the %s skill was not installed", want)
			continue
		}
		if sk.Description == "" {
			t.Errorf("%s has no description, so the model cannot judge when it applies", want)
		}
		if !sk.UserInvocable {
			t.Errorf("%s is not user-invocable, so it gets no slash command", want)
		}
		if !sk.Builtin {
			t.Errorf("%s is not marked as built-in", want)
		}
		if len(sk.Body) < 200 {
			t.Errorf("%s has a %d-byte body, which is too short to be instructions", want, len(sk.Body))
		}
	}
	if _, err := os.Stat(filepath.Join(dir, seedRecord)); err != nil {
		t.Errorf("no seed record was written: %v", err)
	}
}

func TestSeedIsIdempotent(t *testing.T) {
	skillDir(t)

	if _, err := Seed(); err != nil {
		t.Fatal(err)
	}
	written, err := Seed()
	if err != nil {
		t.Fatal(err)
	}
	if len(written) != 0 {
		t.Errorf("the second Seed rewrote %v", written)
	}
}

// The rule the whole thing rests on: upgrading Kiwi must never silently undo
// somebody's edit, and there is no way to ask at the moment this runs.
func TestSeedNeverOverwritesAnEditedSkill(t *testing.T) {
	dir := skillDir(t)
	if _, err := Seed(); err != nil {
		t.Fatal(err)
	}

	mine := "---\nname: plan\ndescription: my own version\n---\nDo it my way.\n"
	path := filepath.Join(dir, "plan.md")
	if err := os.WriteFile(path, []byte(mine), 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := Seed(); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != mine {
		t.Error("Seed overwrote a skill the user had edited")
	}
}

// A deletion is a decision too. Putting the skill back would be the same
// mistake as overwriting an edit.
func TestSeedRespectsADeletedSkill(t *testing.T) {
	dir := skillDir(t)
	if _, err := Seed(); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "plan.md")); err != nil {
		t.Fatal(err)
	}

	if _, err := Seed(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "plan.md")); !os.IsNotExist(err) {
		t.Error("Seed reinstalled a skill the user had deleted")
	}
}

// A skills directory populated before the record existed must be adopted, not
// treated as user-written for ever.
func TestSeedAdoptsAnUnrecordedCopyThatMatches(t *testing.T) {
	dir := skillDir(t)
	current, err := builtinFS.ReadFile("builtin/plan.md")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "plan.md"), current, 0o644); err != nil {
		t.Fatal(err)
	}

	if _, err := Seed(); err != nil {
		t.Fatal(err)
	}
	record := readRecord(filepath.Join(dir, seedRecord))
	if record["plan.md"] != hash(current) {
		t.Error("an identical copy was not adopted into the seed record")
	}
}

func TestSeedSurvivesACorruptRecord(t *testing.T) {
	dir := skillDir(t)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, seedRecord), []byte("nonsense\n\n#comment\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := Seed(); err != nil {
		t.Fatalf("Seed on a corrupt record: %v", err)
	}
	if loaded, _ := Load(); len(loaded) == 0 {
		t.Error("nothing was installed after a corrupt record")
	}
}

func TestSeedRecordIsNotLoadedAsASkill(t *testing.T) {
	skillDir(t)
	if _, err := Seed(); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	for name := range loaded {
		if strings.Contains(name, "seeded") {
			t.Errorf("the seed record was picked up as a skill: %q", name)
		}
	}
}

func TestUserInvocableIsOptIn(t *testing.T) {
	sk, ok := parse("---\nname: quiet\ndescription: d\n---\nbody", "quiet")
	if !ok {
		t.Fatal("parse rejected a valid skill")
	}
	if sk.UserInvocable {
		t.Error("a skill without the flag became a slash command")
	}

	for _, yes := range []string{"user-invocable: true", "user_invocable: yes"} {
		sk, _ := parse("---\nname: loud\ndescription: d\n"+yes+"\n---\nbody", "loud")
		if !sk.UserInvocable {
			t.Errorf("%q was not honoured", yes)
		}
	}
}

func TestInvocableIsSortedAndFiltered(t *testing.T) {
	got := Invocable(map[string]Skill{
		"zeta":  {Name: "zeta", UserInvocable: true},
		"alpha": {Name: "alpha", UserInvocable: true},
		"quiet": {Name: "quiet"},
	})
	if len(got) != 2 {
		t.Fatalf("Invocable returned %d skills, want 2: %+v", len(got), got)
	}
	if got[0].Name != "alpha" || got[1].Name != "zeta" {
		t.Errorf("Invocable is not sorted: %+v", got)
	}
}

// Every built-in has to parse, or it would be embedded, shipped, and silently
// skipped at load time.
func TestEveryBuiltInParses(t *testing.T) {
	entries, err := builtinFS.ReadDir("builtin")
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 6 {
		t.Errorf("found %d built-in skills, want the 6 that are meant to ship", len(entries))
	}
	names := BuiltinNames()
	for _, e := range entries {
		data, err := builtinFS.ReadFile("builtin/" + e.Name())
		if err != nil {
			t.Fatal(err)
		}
		base := strings.TrimSuffix(e.Name(), ".md")
		sk, ok := parse(string(data), base)
		if !ok {
			t.Errorf("%s does not parse", e.Name())
			continue
		}
		if sk.Name != base {
			t.Errorf("%s declares the name %q, which will not match its filename", e.Name(), sk.Name)
		}
		if !names[base] {
			t.Errorf("%s is missing from BuiltinNames", base)
		}
	}
}
