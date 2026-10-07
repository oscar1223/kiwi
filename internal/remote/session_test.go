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

	reply := s.Handle(context.Background(), msg("crea hola.txt"), nil)

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
	s.Handle(context.Background(), msg("me llamo Óscar"), nil)
	s.Handle(context.Background(), msg("¿cómo me llamo?"), nil)

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
	s.Handle(context.Background(), msg("primero"), nil)
	if reply := s.Handle(context.Background(), msg("/new"), nil); !strings.Contains(reply, "nueva") {
		t.Errorf("/new replied %q", reply)
	}
	s.Handle(context.Background(), msg("segundo"), nil)

	if n := len(fake.Requests[1].Messages); n != 1 {
		t.Errorf("after /new the request carried %d messages, want only the new one", n)
	}
}

func TestCommandsDoNotReachTheModel(t *testing.T) {
	s, fake, _ := newTestSession(t)
	for _, c := range []string{"/start", "/help", "/start@kiwi_bot"} {
		if reply := s.Handle(context.Background(), msg(c), nil); !strings.Contains(reply, s.WorkDir) {
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
		s.Handle(context.Background(), msg("tarea larga"), nil)
	}()
	<-started

	if reply := s.Handle(context.Background(), msg("otra cosa"), nil); reply != Busy {
		t.Errorf("second message mid-turn replied %q, want Busy", reply)
	}
	wg.Wait()
}

func TestProviderErrorIsReported(t *testing.T) {
	s, _, saved := newTestSession(t, llmtest.Step{Err: errors.New("529 overloaded")})
	reply := s.Handle(context.Background(), msg("hola"), nil)
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
	if reply := s.Handle(ctx, msg("hola"), nil); reply != "" {
		t.Errorf("reply = %q, want nothing when shutting down", reply)
	}
}

// fakeTranslator answers with a fixed text, or fails.
type fakeTranslator struct {
	text  string
	err   error
	calls int
}

func (f *fakeTranslator) Translate(context.Context, *Media) (string, error) {
	f.calls++
	return f.text, f.err
}

func withMedia(caption string, m *Media) Message {
	msg := msg("")
	msg.Caption = caption
	msg.Attachment = m
	return msg
}

// modelInput is what the model got as the user's message in its first call.
func modelInput(t *testing.T, fake *llmtest.Fake) string {
	t.Helper()
	if len(fake.Requests) == 0 {
		t.Fatal("the model was never called")
	}
	msgs := fake.Requests[0].Messages
	return msgs[len(msgs)-1].Content
}

func TestVoiceNoteIsTranscribedForTheModel(t *testing.T) {
	s, fake, _ := newTestSession(t, llmtest.Step{Text: "Los tests pasan."})
	tr := &fakeTranslator{text: "Revisa el último commit."}
	s.Translator = tr
	conv := &fakeConv{}

	reply := s.Handle(context.Background(), withMedia("", &Media{Kind: MediaVoice, Path: "/inbox/v.ogg"}), conv)

	if reply != "Los tests pasan." {
		t.Errorf("reply = %q", reply)
	}
	in := modelInput(t, fake)
	for _, want := range []string{"nota de voz", "/inbox/v.ogg", "Transcripción", "Revisa el último commit."} {
		if !strings.Contains(in, want) {
			t.Errorf("model input lacks %q:\n%s", want, in)
		}
	}
	// The user sees what was heard, so a bad transcription is caught early.
	ops := conv.snapshot()
	if len(ops) < 2 || !strings.Contains(ops[0].text, "Escuchando") || !ops[1].edit || !strings.Contains(ops[1].text, "«Revisa el último commit.»") {
		t.Errorf("chat ops = %+v, want a notice edited into the transcription", ops)
	}
}

func TestPhotoCaptionFollowsTheDescription(t *testing.T) {
	s, fake, _ := newTestSession(t, llmtest.Step{Text: "Es un nil pointer."})
	s.Translator = &fakeTranslator{text: "Captura con un panic en main.go:42."}

	s.Handle(context.Background(), withMedia("¿qué error es?", &Media{Kind: MediaPhoto, Path: "/inbox/a.jpg"}), nil)

	in := modelInput(t, fake)
	desc, caption := strings.Index(in, "panic en main.go:42"), strings.Index(in, "¿qué error es?")
	if desc < 0 || caption < 0 || caption < desc {
		t.Errorf("model input should carry the description, then the caption:\n%s", in)
	}
}

func TestTranslatorFailureStillRunsTheTurn(t *testing.T) {
	s, fake, _ := newTestSession(t, llmtest.Step{Text: "No puedo oírlo, pero lo tengo guardado."})
	s.Translator = &fakeTranslator{err: errors.New("translator: Insufficient credits")}
	conv := &fakeConv{}

	reply := s.Handle(context.Background(), withMedia("", &Media{Kind: MediaVoice, Path: "/inbox/v.ogg"}), conv)

	if reply == "" || fake.Calls() != 1 {
		t.Fatalf("reply = %q after %d model calls; the turn should run anyway", reply, fake.Calls())
	}
	if in := modelInput(t, fake); !strings.Contains(in, "Insufficient credits") || !strings.Contains(in, "/inbox/v.ogg") {
		t.Errorf("model input should say why and where the file is:\n%s", in)
	}
	if ops := conv.snapshot(); len(ops) < 2 || !strings.Contains(ops[1].text, "No he podido escucharlo") {
		t.Errorf("chat ops = %+v, want the notice to say it could not listen", ops)
	}
}

func TestTextDocumentGoesToTheAgentAsAPath(t *testing.T) {
	s, fake, _ := newTestSession(t, llmtest.Step{Text: "Leído."})
	tr := &fakeTranslator{text: "no debería usarse"}
	s.Translator = tr

	s.Handle(context.Background(), withMedia("", &Media{Kind: MediaDocument, File: File{FileName: "notas.md"}, Path: "/inbox/n.md"}), nil)

	if tr.calls != 0 {
		t.Errorf("translator called %d times for a markdown file", tr.calls)
	}
	if in := modelInput(t, fake); !strings.Contains(in, "/inbox/n.md") || !strings.Contains(in, "read_file") {
		t.Errorf("model input should point at the file and read_file:\n%s", in)
	}
}

func TestMediaWithoutTranslatorSaysWhy(t *testing.T) {
	s, fake, _ := newTestSession(t, llmtest.Step{Text: "Vale."})

	s.Handle(context.Background(), withMedia("", &Media{Kind: MediaPhoto, Path: "/inbox/a.jpg"}), nil)

	if in := modelInput(t, fake); !strings.Contains(in, "OPENROUTER_API_KEY") || !strings.Contains(in, "/inbox/a.jpg") {
		t.Errorf("model input should explain there is no translator:\n%s", in)
	}
}

func TestCaptionIsNotACommandWithMedia(t *testing.T) {
	s, fake, saved := newTestSession(t, llmtest.Step{Text: "Visto."})
	s.Translator = &fakeTranslator{text: "una foto"}

	s.Handle(context.Background(), withMedia("/new", &Media{Kind: MediaPhoto, Path: "/inbox/a.jpg"}), nil)

	if fake.Calls() != 1 || len(*saved) != 1 {
		t.Errorf("a photo captioned /new should be a turn, not reset the chat (calls=%d, saved=%d)", fake.Calls(), len(*saved))
	}
}
