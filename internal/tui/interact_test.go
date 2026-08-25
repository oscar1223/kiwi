package tui

import (
	"context"
	"testing"

	"github.com/oscar1223/kiwi/internal/llm"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
)

func TestPickArrowsMoveSelectionAndEnterResolves(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	result := make(chan struct {
		value string
		ok    bool
	}, 1)
	go func() {
		v, ok := m.events.Pick(context.Background(), "choose one", []pickOption{
			{"First", "a"}, {"Second", "b"}, {"Third", "c"},
		})
		result <- struct {
			value string
			ok    bool
		}{v, ok}
	}()

	req := waitForPickRequest(t, m)
	m = update(t, m, req)
	if m.activePick == nil {
		t.Fatal("picker did not open")
	}
	if m.activePick.index != 0 {
		t.Fatalf("initial index = %d, want 0", m.activePick.index)
	}

	m = update(t, m, key("down"))
	m = update(t, m, key("down"))
	if m.activePick.index != 2 {
		t.Fatalf("index after two downs = %d, want 2", m.activePick.index)
	}
	m = update(t, m, key("up"))
	if m.activePick.index != 1 {
		t.Fatalf("index after one up = %d, want 1", m.activePick.index)
	}

	m = update(t, m, key("enter"))
	if m.activePick != nil {
		t.Error("picker did not close on enter")
	}

	select {
	case r := <-result:
		if !r.ok || r.value != "b" {
			t.Errorf("result = (%q, %v), want (\"b\", true)", r.value, r.ok)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Pick never returned")
	}
}

func TestPickDownStopsAtLastOption(t *testing.T) {
	p := &pickState{req: &pickRequest{options: []pickOption{{"a", "a"}, {"b", "b"}}}}
	p.down()
	p.down()
	p.down()
	if p.index != 1 {
		t.Errorf("index = %d, want it clamped to 1", p.index)
	}
}

func TestPickUpStopsAtFirstOption(t *testing.T) {
	p := &pickState{req: &pickRequest{options: []pickOption{{"a", "a"}, {"b", "b"}}}}
	p.up()
	if p.index != 0 {
		t.Errorf("index = %d, want it clamped to 0", p.index)
	}
}

func TestPickEscCancels(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	result := make(chan bool, 1)
	go func() {
		_, ok := m.events.Pick(context.Background(), "choose", []pickOption{{"a", "a"}})
		result <- ok
	}()

	req := waitForPickRequest(t, m)
	m = update(t, m, req)
	m = update(t, m, key("esc"))

	if m.activePick != nil {
		t.Error("picker did not close on esc")
	}
	select {
	case ok := <-result:
		if ok {
			t.Error("esc should report ok=false")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Pick never returned")
	}
}

func TestTextSubmitReturnsTypedValue(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 80

	result := make(chan struct {
		value string
		ok    bool
	}, 1)
	go func() {
		v, ok := m.events.Text(context.Background(), "name?", "placeholder", "")
		result <- struct {
			value string
			ok    bool
		}{v, ok}
	}()

	req := waitForTextRequest(t, m)
	m = update(t, m, req)
	if m.activeText == nil {
		t.Fatal("text prompt did not open")
	}

	for _, r := range "kiwi" {
		m = update(t, m, key(string(r)))
	}
	m = update(t, m, key("enter"))

	if m.activeText != nil {
		t.Error("text prompt did not close on enter")
	}
	select {
	case r := <-result:
		if !r.ok || r.value != "kiwi" {
			t.Errorf("result = (%q, %v), want (\"kiwi\", true)", r.value, r.ok)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Text never returned")
	}
}

func TestTextDefaultValueIsPrefilled(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 80

	go m.events.Text(context.Background(), "name?", "", "sonnet")

	req := waitForTextRequest(t, m)
	m = update(t, m, req)
	if got := m.activeText.input.Value(); got != "sonnet" {
		t.Errorf("prefilled value = %q, want %q", got, "sonnet")
	}
}

func TestTextEscCancels(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.width = 80

	result := make(chan bool, 1)
	go func() {
		_, ok := m.events.Text(context.Background(), "name?", "", "")
		result <- ok
	}()

	req := waitForTextRequest(t, m)
	m = update(t, m, req)
	m = update(t, m, key("esc"))

	if m.activeText != nil {
		t.Error("text prompt did not close on esc")
	}
	select {
	case ok := <-result:
		if ok {
			t.Error("esc should report ok=false")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Text never returned")
	}
}

// While a picker or text prompt is open, ordinary typing must not fall
// through to the main input box — this is the same guarantee the permission
// prompt already gives, extended to the new modal states.
func TestOrdinaryTypingIsIgnoredWhilePickIsOpen(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	go m.events.Pick(context.Background(), "x", []pickOption{{"a", "a"}})
	req := waitForPickRequest(t, m)
	m = update(t, m, req)

	m = update(t, m, key("a"))
	if m.input.Value() != "" {
		t.Errorf("input = %q; keystrokes leaked into the main prompt while a picker was open", m.input.Value())
	}
}

// Confirm is Pick specialised to yes/no: this locks in that specialisation
// rather than re-testing Pick's mechanics.
func TestConfirmYesAndNo(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	for _, tc := range []struct {
		key  string
		want bool
	}{{"enter", true}, {"down", false}} {
		result := make(chan bool, 1)
		go func() { result <- m.events.Confirm(context.Background(), "sure?") }()

		req := waitForPickRequest(t, m)
		m = update(t, m, req)
		if tc.key == "down" {
			m = update(t, m, key("down"))
			m = update(t, m, key("enter"))
		} else {
			m = update(t, m, key(tc.key))
		}

		select {
		case got := <-result:
			if got != tc.want {
				t.Errorf("Confirm = %v, want %v", got, tc.want)
			}
		case <-time.After(2 * time.Second):
			t.Fatal("Confirm never returned")
		}
	}
}

func waitForPickRequest(t *testing.T, m *Model) *pickRequest {
	t.Helper()
	select {
	case msg := <-m.events.ch:
		req, ok := msg.(*pickRequest)
		if !ok {
			t.Fatalf("got %T, want *pickRequest", msg)
		}
		return req
	case <-time.After(2 * time.Second):
		t.Fatal("no pick request arrived")
		return nil
	}
}

func waitForTextRequest(t *testing.T, m *Model) *textRequest {
	t.Helper()
	select {
	case msg := <-m.events.ch:
		req, ok := msg.(*textRequest)
		if !ok {
			t.Fatalf("got %T, want *textRequest", msg)
		}
		return req
	case <-time.After(2 * time.Second):
		t.Fatal("no text request arrived")
		return nil
	}
}

func TestFilterCommandsEmptyQueryReturnsEverything(t *testing.T) {
	got := filterCommands("/")
	if len(got) != len(commandRegistry) {
		t.Errorf("got %d entries, want all %d", len(got), len(commandRegistry))
	}
}

func TestFilterCommandsNarrowsBySubstring(t *testing.T) {
	got := filterCommands("/mod")
	if len(got) != 1 || got[0].Name != "/model" {
		t.Errorf("filterCommands(\"/mod\") = %+v, want just /model", got)
	}
}

// A query that is a substring of more than one command name must return all
// of them, in registry order — memory contains "mo" too (me-MO-ry).
func TestFilterCommandsAmbiguousSubstringMatchesAll(t *testing.T) {
	got := filterCommands("/mo")
	if len(got) != 2 || got[0].Name != "/model" || got[1].Name != "/memory" {
		t.Errorf("filterCommands(\"/mo\") = %+v, want [/model /memory]", got)
	}
}

func TestFilterCommandsCaseInsensitive(t *testing.T) {
	got := filterCommands("/MOD")
	if len(got) != 1 || got[0].Name != "/model" {
		t.Errorf("filterCommands(\"/MOD\") = %+v", got)
	}
}

func TestIsKnownCommandExactMatch(t *testing.T) {
	if !isKnownCommand("/model") {
		t.Error("/model should be recognised")
	}
	if !isKnownCommand("/model gpt") {
		t.Error("a command with trailing arguments should still be recognised")
	}
	if isKnownCommand("/mod") {
		t.Error("a partial command should not count as known")
	}
	if isKnownCommand("hello") {
		t.Error("ordinary text should not count as a command")
	}
}

func TestSlashSuggestionsOnlyApplyToBareSlashInput(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)

	m.input.SetValue("/mod")
	if got := m.slashSuggestions(); len(got) != 1 {
		t.Errorf("slashSuggestions() = %+v, want just /model", got)
	}

	m.input.SetValue("not a command")
	if got := m.slashSuggestions(); got != nil {
		t.Errorf("slashSuggestions() = %+v, want nil for ordinary text", got)
	}

	m.input.SetValue("/model gpt")
	if got := m.slashSuggestions(); got != nil {
		t.Errorf("slashSuggestions() = %+v, want nil once past the command name", got)
	}
}

func TestSlashSuggestionsSuppressedDuringAnotherModal(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.input.SetValue("/mo")
	m.pending = &permission.Request{}

	if got := m.slashSuggestions(); got != nil {
		t.Errorf("slashSuggestions() = %+v, want nil while a permission prompt owns the keyboard", got)
	}
}

func TestArrowsMoveSlashSuggestionHighlight(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.input.SetValue("/")

	m = update(t, m, key("down"))
	if m.cmdSuggestIndex != 1 {
		t.Errorf("cmdSuggestIndex = %d, want 1 after one down", m.cmdSuggestIndex)
	}
	m = update(t, m, key("up"))
	if m.cmdSuggestIndex != 0 {
		t.Errorf("cmdSuggestIndex = %d, want 0 after up", m.cmdSuggestIndex)
	}
}

func TestTabCompletesToHighlightedCommand(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.input.SetValue("/mod")

	m = update(t, m, key("tab"))
	if got := m.input.Value(); got != "/model " {
		t.Errorf("input.Value() = %q, want %q", got, "/model ")
	}
	// Completing must not submit — no turn should have started.
	if m.gen != 0 || m.busy {
		t.Error("tab-completing a command submitted it")
	}
}

// The core rule from the plan: enter on a partial match completes instead of
// submitting; enter on an exact match submits normally.
func TestEnterOnPartialMatchCompletesInsteadOfSubmitting(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.input.SetValue("/mod")

	m = update(t, m, key("enter"))
	if got := m.input.Value(); got != "/model " {
		t.Errorf("input.Value() = %q, want completion to %q", got, "/model ")
	}
	if m.gen != 0 {
		t.Error("a partial command should not have submitted")
	}
}

func TestEnterOnExactCommandFallsThroughToSubmit(t *testing.T) {
	m, _ := newTestModel(t, permission.ModeAsk)
	m.input.SetValue("/clear")
	m.history = []llm.Message{{Role: llm.RoleUser, Content: "x"}}

	m = update(t, m, key("enter"))
	// /clear's own handler runs: history is wiped and input is consumed.
	if len(m.history) != 0 {
		t.Error("/clear did not run — enter did not fall through past the suggestion intercept")
	}
}
