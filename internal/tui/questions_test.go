package tui

import (
	"context"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/tools"
)

func waitForQuestionRequest(t *testing.T, m *Model) *questionRequest {
	t.Helper()
	select {
	case msg := <-m.events.ch:
		req, ok := msg.(*questionRequest)
		if !ok {
			t.Fatalf("got %T, want *questionRequest", msg)
		}
		return req
	case <-time.After(2 * time.Second):
		t.Fatal("no question request arrived")
		return nil
	}
}

func twoOptionQuestion(multi bool) tools.Question {
	return tools.Question{
		Question:    "Auth method?",
		Header:      "Auth",
		MultiSelect: multi,
		Options: []tools.QuestionOption{
			{Label: "OAuth", Description: "delegated"},
			{Label: "API key", Description: "simple"},
		},
	}
}

func TestAskQuestionsSingleSelectEnterResolves(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	result := make(chan struct {
		answers []tools.Answer
		ok      bool
	}, 1)
	go func() {
		a, ok := m.events.AskQuestions(context.Background(), []tools.Question{twoOptionQuestion(false)})
		result <- struct {
			answers []tools.Answer
			ok      bool
		}{a, ok}
	}()

	req := waitForQuestionRequest(t, m)
	m = update(t, m, req)
	if m.activeQuestion == nil {
		t.Fatal("question prompt did not open")
	}

	m = update(t, m, key("down"))
	if m.activeQuestion.index != 1 {
		t.Fatalf("index after one down = %d, want 1", m.activeQuestion.index)
	}
	m = update(t, m, key("enter"))
	if m.activeQuestion != nil {
		t.Error("prompt did not close on enter")
	}

	select {
	case r := <-result:
		if !r.ok || len(r.answers) != 1 || len(r.answers[0].Values) != 1 || r.answers[0].Values[0] != "API key" {
			t.Errorf("result = %+v, want single answer \"API key\"", r)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AskQuestions never returned")
	}
}

func TestAskQuestionsEscCancelsWholeBatch(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	result := make(chan bool, 1)
	go func() {
		_, ok := m.events.AskQuestions(context.Background(), []tools.Question{twoOptionQuestion(false)})
		result <- ok
	}()

	req := waitForQuestionRequest(t, m)
	m = update(t, m, req)
	m = update(t, m, key("esc"))

	if m.activeQuestion != nil {
		t.Error("prompt did not close on esc")
	}
	select {
	case ok := <-result:
		if ok {
			t.Error("esc should report ok=false")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AskQuestions never returned")
	}
}

func TestAskQuestionsAsksEachQuestionInOrder(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	qs := []tools.Question{twoOptionQuestion(false), twoOptionQuestion(false)}
	qs[1].Question = "Storage backend?"

	result := make(chan []tools.Answer, 1)
	go func() {
		a, _ := m.events.AskQuestions(context.Background(), qs)
		result <- a
	}()

	req1 := waitForQuestionRequest(t, m)
	if req1.q.Question != "Auth method?" {
		t.Fatalf("first question = %q, want %q", req1.q.Question, "Auth method?")
	}
	m = update(t, m, req1)
	m = update(t, m, key("enter"))

	req2 := waitForQuestionRequest(t, m)
	if req2.q.Question != "Storage backend?" {
		t.Fatalf("second question = %q, want %q", req2.q.Question, "Storage backend?")
	}
	m = update(t, m, req2)
	m = update(t, m, key("down"))
	m = update(t, m, key("enter"))

	select {
	case answers := <-result:
		if len(answers) != 2 || answers[0].Values[0] != "OAuth" || answers[1].Values[0] != "API key" {
			t.Errorf("answers = %+v", answers)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AskQuestions never returned")
	}
}

func TestAskQuestionsMultiSelectTogglesAndConfirms(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	result := make(chan []tools.Answer, 1)
	go func() {
		a, _ := m.events.AskQuestions(context.Background(), []tools.Question{twoOptionQuestion(true)})
		result <- a
	}()

	req := waitForQuestionRequest(t, m)
	m = update(t, m, req)

	m = update(t, m, key("space")) // toggle OAuth on
	m = update(t, m, key("down"))
	m = update(t, m, key("space")) // toggle API key on
	m = update(t, m, key("enter"))

	select {
	case answers := <-result:
		if len(answers) != 1 || len(answers[0].Values) != 2 {
			t.Fatalf("answers = %+v, want both options toggled", answers)
		}
		if answers[0].Values[0] != "OAuth" || answers[0].Values[1] != "API key" {
			t.Errorf("answers[0].Values = %v, want [OAuth API key] in option order", answers[0].Values)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AskQuestions never returned")
	}
}

func TestAskQuestionsMultiSelectEnterWithNoneToggledUsesHighlighted(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	result := make(chan []tools.Answer, 1)
	go func() {
		a, _ := m.events.AskQuestions(context.Background(), []tools.Question{twoOptionQuestion(true)})
		result <- a
	}()

	req := waitForQuestionRequest(t, m)
	m = update(t, m, req)
	m = update(t, m, key("down"))
	m = update(t, m, key("enter"))

	select {
	case answers := <-result:
		if len(answers) != 1 || len(answers[0].Values) != 1 || answers[0].Values[0] != "API key" {
			t.Errorf("answers = %+v, want the highlighted option alone", answers)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AskQuestions never returned")
	}
}

// The trailing "Other" row lets the user answer with free text instead of
// any listed option — the same escape hatch Claude Code's own
// AskUserQuestion gives.
func TestAskQuestionsOtherRowOpensFreeText(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 80

	result := make(chan []tools.Answer, 1)
	go func() {
		a, _ := m.events.AskQuestions(context.Background(), []tools.Question{twoOptionQuestion(false)})
		result <- a
	}()

	req := waitForQuestionRequest(t, m)
	m = update(t, m, req)
	m = update(t, m, key("down")) // API key
	m = update(t, m, key("down")) // Other row
	if m.activeQuestion.index != m.activeQuestion.otherIndex() {
		t.Fatalf("index = %d, want the Other row at %d", m.activeQuestion.index, m.activeQuestion.otherIndex())
	}
	m = update(t, m, key("enter")) // select Other, open free-text entry
	if !m.activeQuestion.otherActive {
		t.Fatal("selecting the Other row did not open free-text entry")
	}

	for _, r := range "SAML" {
		m = update(t, m, key(string(r)))
	}
	m = update(t, m, key("enter"))

	if m.activeQuestion != nil {
		t.Error("prompt did not close after submitting the Other answer")
	}
	select {
	case answers := <-result:
		if len(answers) != 1 || len(answers[0].Values) != 1 || answers[0].Values[0] != "SAML" {
			t.Errorf("answers = %+v, want [\"SAML\"]", answers)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("AskQuestions never returned")
	}
}
