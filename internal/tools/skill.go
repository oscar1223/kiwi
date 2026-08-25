package tools

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/oscar1223/kiwi/internal/skills"
)

// LoadSkill lets the model pull a skill's full instructions into context on
// demand. Only its name and a one-line description are ever in the system
// prompt (see skills.Summary) — the body is loaded only when the model
// judges a task actually matches, so skills that never get used cost nothing
// beyond that one line.
type LoadSkill struct {
	Skills map[string]skills.Skill
}

func (LoadSkill) Name() string { return "load_skill" }

func (LoadSkill) Description() string {
	return "Load the full instructions of a skill by name. Only call this when the " +
		"current task matches a skill's description from the system prompt."
}

func (LoadSkill) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string", "description": "The skill's name, as listed in the system prompt."},
		},
		"required": []string{"name"},
	}
}

func (t LoadSkill) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	sk, ok := t.Skills[in.Name]
	if !ok {
		return "", fmt.Errorf("no skill named %q", in.Name)
	}
	return sk.Body, nil
}
