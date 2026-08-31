package tools

import (
	"context"
	"strings"
	"testing"
)

// fakeAsker is a scripted Asker: it returns answers or a cancellation
// without any UI, so AskQuestionsTool can be tested in isolation.
type fakeAsker struct {
	got     []Question
	answers []Answer
	ok      bool
}

func (f *fakeAsker) AskQuestions(ctx context.Context, qs []Question) ([]Answer, bool) {
	f.got = qs
	return f.answers, f.ok
}

func twoOptionQuestion(q string) map[string]any {
	return map[string]any{
		"question": q,
		"header":   "h",
		"options": []map[string]any{
			{"label": "A", "description": "first"},
			{"label": "B", "description": "second"},
		},
	}
}

func TestAskQuestionsReturnsAnswersOnSuccess(t *testing.T) {
	asker := &fakeAsker{
		ok: true,
		answers: []Answer{
			{Question: "Auth method?", Values: []string{"OAuth"}},
		},
	}
	tool := AskQuestionsTool{Asker: asker}

	out, err := call(t, tool, map[string]any{
		"questions": []map[string]any{twoOptionQuestion("Auth method?")},
	})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if !strings.Contains(out, "Auth method?") || !strings.Contains(out, "OAuth") {
		t.Errorf("output missing question/answer: %q", out)
	}
	if len(asker.got) != 1 || asker.got[0].Question != "Auth method?" {
		t.Errorf("Asker did not receive the question as sent: %+v", asker.got)
	}
}

func TestAskQuestionsReportsCancellationWithoutError(t *testing.T) {
	asker := &fakeAsker{ok: false}
	tool := AskQuestionsTool{Asker: asker}

	out, err := call(t, tool, map[string]any{
		"questions": []map[string]any{twoOptionQuestion("Storage backend?")},
	})
	if err != nil {
		t.Fatalf("cancellation should not be a tool error: %v", err)
	}
	if !strings.Contains(strings.ToLower(out), "cancelled") {
		t.Errorf("output should tell the model the user cancelled: %q", out)
	}
}

func TestAskQuestionsRejectsEmptyQuestionList(t *testing.T) {
	tool := AskQuestionsTool{Asker: &fakeAsker{}}
	if _, err := call(t, tool, map[string]any{"questions": []map[string]any{}}); err == nil {
		t.Error("an empty question list should be rejected")
	}
}

func TestAskQuestionsRejectsTooFewOptions(t *testing.T) {
	tool := AskQuestionsTool{Asker: &fakeAsker{}}
	_, err := call(t, tool, map[string]any{
		"questions": []map[string]any{{
			"question": "Which one?",
			"header":   "h",
			"options":  []map[string]any{{"label": "only one"}},
		}},
	})
	if err == nil {
		t.Error("a question with fewer than two options should be rejected")
	}
}

func TestAskQuestionsRejectsBlankQuestionText(t *testing.T) {
	tool := AskQuestionsTool{Asker: &fakeAsker{}}
	_, err := call(t, tool, map[string]any{
		"questions": []map[string]any{twoOptionQuestion("   ")},
	})
	if err == nil {
		t.Error("a blank question should be rejected")
	}
}
