package tools

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/skills"
)

func TestLoadSkillReturnsBody(t *testing.T) {
	tool := LoadSkill{Skills: map[string]skills.Skill{
		"commit-style": {Name: "commit-style", Body: "Follow conventional commits."},
	}}
	raw, _ := json.Marshal(map[string]string{"name": "commit-style"})

	out, err := tool.Run(context.Background(), raw)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out != "Follow conventional commits." {
		t.Errorf("out = %q", out)
	}
}

func TestLoadSkillUnknownName(t *testing.T) {
	tool := LoadSkill{Skills: map[string]skills.Skill{}}
	raw, _ := json.Marshal(map[string]string{"name": "nope"})

	_, err := tool.Run(context.Background(), raw)
	if err == nil {
		t.Fatal("expected an error for an unknown skill")
	}
	if !strings.Contains(err.Error(), "nope") {
		t.Errorf("err = %v, should name the skill so the model can self-correct", err)
	}
}

func TestDefaultRegistersExtraTools(t *testing.T) {
	extra := LoadSkill{Skills: map[string]skills.Skill{}}
	r := Default(t.TempDir(), nil, extra)
	if _, ok := r.Get("load_skill"); !ok {
		t.Error("extra tool was not registered")
	}
	if _, ok := r.Get("bash"); !ok {
		t.Error("Default's own tools were dropped when extras were added")
	}
}
