package remote

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/tools"
)

// newTestSession wires a real agent and real tools on a temp directory to a
// scripted provider, so a turn runs end to end without a network.
func newTestSession(t *testing.T, steps ...llmtest.Step) (*Session, *llmtest.Fake, *[][]llm.Message) {
	t.Helper()
	dir := t.TempDir()
	fake := &llmtest.Fake{Steps: steps}
	broker := permission.NewBroker(permission.ModeWork, permission.NonInteractive{})

	var saved [][]llm.Message
	s := &Session{
		Agent:   &agent.Agent{Provider: fake, Tools: tools.Default(dir, broker), MaxSteps: 10},
		WorkDir: dir,
		Save: func(_ context.Context, turn []llm.Message) ([]llm.Message, error) {
			saved = append(saved, turn)
			var all []llm.Message
			for _, m := range saved {
				all = append(all, m...)
			}
			return all, nil
		},
		Reset: func(context.Context) error { saved = nil; return nil },
	}
	return s, fake, &saved
}

func msg(text string) Message {
	return Message{From: &User{ID: 42}, Chat: Chat{ID: 42, Type: "private"}, Text: text}
}

func TestTurnWithToolCalls(t *testing.T) {
	s, fake, saved := newTestSession(t,
		llmtest.Step{ToolCalls: []llm.ToolCall{
			llmtest.Call("c1", "write_file", map[string]string{"path": "hola.txt", "content": "kiwi"}),
		}},
		llmtest.Step{Text: "He creado hola.txt."},
	)

	reply := s.Handle(context.Background(), msg("crea hola.txt"))

	if reply != "He creado hola.txt." {
		t.Errorf("reply = %q, want the model's final text", reply)
	}
	data, err := os.ReadFile(filepath.Join(s.WorkDir, "hola.txt"))
	if err != nil || string(data) != "kiwi" {
		t.Errorf("hola.txt = %q, %v; the tool call should have run in work mode", data, err)
	}
	if fake.Calls() != 2 {
		t.Errorf("provider called %d times, want 2 (tool call, then answer)", fake.Calls())
	}
	// user, assistant+tool call, tool result, assistant answer
	if len(*saved) != 1 || len((*saved)[0]) != 4 {
		t.Fatalf("saved %v, want one turn of 4 messages", *saved)
	}
	if len(s.History) != 4 {
		t.Errorf("history has %d messages after the turn, want 4", len(s.History))
	}
}

func TestHistoryCarriesOver(t *testing.T) {
	s, fake, _ := newTestSession(t,
		llmtest.Step{Text: "Encantado, Óscar."},
		llmtest.Step{Text: "Te llamas Óscar."},
	)
	s.Handle(context.Background(), msg("me llamo Óscar"))
	s.Handle(context.Background(), msg("¿cómo me llamo?"))

	second := fake.Requests[1].Messages
	if len(second) < 3 || !strings.Contains(second[0].Content, "Óscar") {
		t.Errorf("second request did not include the first turn: %+v", second)
	}
}

func TestNewStartsOver(t *testing.T) {
	s, fake, _ := newTestSession(t,
		llmtest.Step{Text: "uno"},
		llmtest.Step{Text: "dos"},
	)
	s.Handle(context.Background(), msg("primero"))
	if reply := s.Handle(context.Background(), msg("/new")); !strings.Contains(reply, "nueva") {
		t.Errorf("/new replied %q", reply)
	}
	s.Handle(context.Background(), msg("segundo"))

	if n := len(fake.Requests[1].Messages); n != 1 {
		t.Errorf("after /new the request carried %d messages, want only the new one", n)
	}
}

func TestCommandsDoNotReachTheModel(t *testing.T) {
	s, fake, _ := newTestSession(t)
	for _, c := range []string{"/start", "/help", "/start@kiwi_bot"} {
		if reply := s.Handle(context.Background(), msg(c)); !strings.Contains(reply, s.WorkDir) {
			t.Errorf("%s replied %q, want it to say which directory it works in", c, reply)
		}
	}
	if fake.Calls() != 0 {
		t.Errorf("commands reached the provider %d times", fake.Calls())
	}
}

func TestBusyWhileTurnRuns(t *testing.T) {
	started := make(chan struct{})
	s, _, _ := newTestSession(t, llmtest.Step{
		Chunks: []string{"lento", "…"},
		Delay:  100 * time.Millisecond,
		Hook:   func() { close(started) },
	})

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		s.Handle(context.Background(), msg("tarea larga"))
	}()
	<-started

	if reply := s.Handle(context.Background(), msg("otra cosa")); reply != Busy {
		t.Errorf("second message mid-turn replied %q, want Busy", reply)
	}
	wg.Wait()
}

func TestProviderErrorIsReported(t *testing.T) {
	s, _, saved := newTestSession(t, llmtest.Step{Err: errors.New("529 overloaded")})
	reply := s.Handle(context.Background(), msg("hola"))
	if !strings.Contains(reply, "529 overloaded") {
		t.Errorf("reply = %q, want the error passed on", reply)
	}
	if len(*saved) != 0 {
		t.Error("a failed turn should not be saved")
	}
}

func TestCancelledTurnSendsNothing(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	s, _, _ := newTestSession(t, llmtest.Step{Hook: cancel, Text: "nunca"})
	if reply := s.Handle(ctx, msg("hola")); reply != "" {
		t.Errorf("reply = %q, want nothing when shutting down", reply)
	}
}
