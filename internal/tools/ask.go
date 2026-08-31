package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// Question is one clarifying question the model can ask the user, in the
// same shape Claude Code's own question tool uses: a short question, a
// header for compact UI, and a small set of concrete options — optionally
// more than one of which may be chosen.
type Question struct {
	Question    string           `json:"question"`
	Header      string           `json:"header"`
	MultiSelect bool             `json:"multi_select"`
	Options     []QuestionOption `json:"options"`
}

// QuestionOption is one choice offered for a Question. The user can always
// answer with free text instead — see Asker — so Description only needs to
// explain the trade-off, not exhaust every possibility.
type QuestionOption struct {
	Label       string `json:"label"`
	Description string `json:"description"`
}

// Answer is the user's response to one Question. Values has one entry for a
// single-select question, and one per chosen option for a multi-select one.
// A typed-in answer (instead of picking a listed option) comes back the same
// way, as its own value.
type Answer struct {
	Question string   `json:"question"`
	Values   []string `json:"values"`
}

// Asker presents questions to a human and blocks for their answers. The TUI
// is the only implementation; a headless run (`kiwi ask`, or a run with no
// terminal attached) has no Asker, so AskQuestionsTool is left out of the
// registry entirely rather than failing every call at run time — see
// assembleAgent in cmd/kiwi.
type Asker interface {
	// AskQuestions returns the collected answers and true, or nil and false
	// if the user cancelled — a partial answer set is not useful to the
	// model, so cancelling any question declines the whole call.
	AskQuestions(ctx context.Context, qs []Question) ([]Answer, bool)
}

// AskQuestionsTool lets the model pause and ask the user one or more
// clarifying questions — multiple choice, optionally multi-select, always
// with room for a free-text answer — before committing to a plan. It exists
// for Plan mode: a plan built on a guessed requirement is worse than one
// that took an extra turn to ask.
type AskQuestionsTool struct {
	Asker Asker
}

func (AskQuestionsTool) Name() string { return "ask_questions" }

func (AskQuestionsTool) Description() string {
	return "Ask the user one or more clarifying questions before finalizing a plan. " +
		"Use it when a design decision, ambiguous requirement, or missing constraint " +
		"would otherwise force a guess. Each question offers 2-4 concrete options (the " +
		"user can always type a custom answer instead) and, optionally, allows picking " +
		"more than one. Ask everything needed in a single call — every question is shown " +
		"in turn — rather than calling this tool repeatedly. Do not use it for anything " +
		"answerable by reading the code."
}

func (AskQuestionsTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"questions": map[string]any{
				"type":        "array",
				"minItems":    1,
				"description": "The questions to ask, in order.",
				"items": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"question": map[string]any{
							"type":        "string",
							"description": "The full question, ending in a question mark.",
						},
						"header": map[string]any{
							"type":        "string",
							"description": "A very short label (<=12 chars) for this question, e.g. \"Auth method\".",
						},
						"multi_select": map[string]any{
							"type":        "boolean",
							"description": "Whether more than one option may be chosen. Defaults to false.",
						},
						"options": map[string]any{
							"type":     "array",
							"minItems": 2,
							"maxItems": 4,
							"items": map[string]any{
								"type": "object",
								"properties": map[string]any{
									"label":       map[string]any{"type": "string"},
									"description": map[string]any{"type": "string"},
								},
								"required": []string{"label"},
							},
						},
					},
					"required": []string{"question", "header", "options"},
				},
			},
		},
		"required": []string{"questions"},
	}
}

func (t AskQuestionsTool) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Questions []Question `json:"questions"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", fmt.Errorf("ask_questions: %w", err)
	}
	if len(in.Questions) == 0 {
		return "", fmt.Errorf("ask_questions: at least one question is required")
	}
	for _, q := range in.Questions {
		if strings.TrimSpace(q.Question) == "" {
			return "", fmt.Errorf("ask_questions: a question cannot be empty")
		}
		if len(q.Options) < 2 {
			return "", fmt.Errorf("ask_questions: %q needs at least two options", q.Question)
		}
	}

	answers, ok := t.Asker.AskQuestions(ctx, in.Questions)
	if !ok {
		return "The user cancelled without answering. Proceed with your best judgment, " +
			"or note the open question in the plan instead of guessing silently.", nil
	}

	var b strings.Builder
	for _, a := range answers {
		b.WriteString("- ")
		b.WriteString(a.Question)
		b.WriteString(" → ")
		b.WriteString(strings.Join(a.Values, ", "))
		b.WriteString("\n")
	}
	return b.String(), nil
}
