package tui

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/prompt"
	"github.com/oscar1223/kiwi/internal/session"
	"github.com/oscar1223/kiwi/internal/tools"
)

var ansi = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[()][AB012]|\x1b[=>]`)

// syncBuffer is safe for the renderer goroutine and the test to share.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

// runProgram drives the real Bubble Tea program with a scripted model and
// returns everything it painted, with escape sequences stripped.
//
// This is the closest thing to running kiwi in a terminal that a test can do,
// and it is what catches wiring mistakes the unit tests cannot see: dropped
// commands, events that never reach Update, output printed out of order.
func runProgram(t *testing.T, steps []llmtest.Step, mode permission.Mode, keys ...string) string {
	t.Helper()
	out, _ := runProgramOpts(t, programOpts{steps: steps, mode: mode, keys: keys})
	return out
}

// programOpts extends runProgram with session persistence, for tests that
// need to see what actually landed in the store.
type programOpts struct {
	steps   []llmtest.Step
	mode    permission.Mode
	keys    []string
	history []llm.Message
	store   *session.Store
	sessID  string
}

func runProgramOpts(t *testing.T, po programOpts) (string, *Model) {
	t.Helper()

	fake := &llmtest.Fake{Steps: po.steps}
	ev := NewEvents()
	broker := permission.NewBroker(po.mode, ev)
	dir := t.TempDir()

	m := New(Options{
		Agent: &agent.Agent{
			Provider: fake,
			Tools:    tools.Default(dir, broker),
			System:   "test",
		},
		Broker:        broker,
		WorkDir:       dir,
		ModelLabel:    "fake/fake-1",
		PromptOptions: prompt.Options{WorkingDir: dir},
		Events:        ev,
		History:       po.history,
		Store:         po.store,
		SessionID:     po.sessID,
	})
	broker.OnAutoDecision(ev.LogAutoDecision)
	keys := po.keys

	in, inWriter := io.Pipe()
	out := &syncBuffer{}

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	p := tea.NewProgram(m, tea.WithContext(ctx), tea.WithInput(in), tea.WithOutput(out))

	done := make(chan error, 1)
	go func() { _, err := p.Run(); done <- err }()

	// Keys are sent in stages with a pause between them. This is not just
	// convenience: answering a permission prompt requires the prompt to
	// already be on screen, and anything typed earlier correctly lands in the
	// input box instead.
	time.Sleep(150 * time.Millisecond)
	for _, k := range keys {
		if _, err := io.WriteString(inWriter, k); err != nil {
			t.Fatalf("writing keys: %v", err)
		}
		time.Sleep(500 * time.Millisecond)
	}

	time.Sleep(300 * time.Millisecond)
	p.Quit()

	select {
	case err := <-done:
		if err != nil && err != tea.ErrProgramKilled {
			t.Fatalf("program: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("the program did not exit")
	}
	inWriter.Close()

	return ansi.ReplaceAllString(out.String(), ""), m
}

func TestProgramStreamsAnAnswer(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		{Chunks: []string{"First line.\n", "Second ", "line.\n"}},
	}, permission.ModeAsk, "hola\r")

	if !strings.Contains(out, "hola") {
		t.Errorf("the user's message was not echoed:\n%s", out)
	}
	if !strings.Contains(out, "First line.") || !strings.Contains(out, "Second line.") {
		t.Errorf("the answer did not reach the screen:\n%s", out)
	}
	// Deltas that arrived split across chunks must be reassembled into one
	// line, not printed piecemeal.
	if strings.Contains(out, "Second\n") {
		t.Errorf("a line was broken at a chunk boundary:\n%s", out)
	}
}

// Batch runs commands concurrently, so printing several lines through it
// shuffles them. This is the regression test for that.
func TestProgramPrintsLinesInOrder(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		{Chunks: []string{"alpha\nbravo\ncharlie\ndelta\necho\n"}},
	}, permission.ModeAsk, "go\r")

	want := []string{"alpha", "bravo", "charlie", "delta", "echo"}
	prev := -1
	for _, w := range want {
		i := strings.Index(out, w)
		if i < 0 {
			t.Fatalf("%q never printed:\n%s", w, out)
		}
		if i < prev {
			t.Fatalf("%q printed out of order:\n%s", w, out)
		}
		prev = i
	}
}

// The command assembles its output from pieces, so the probe string never
// appears in the command line itself. Otherwise "did the command run?" and
// "was the command displayed?" would look identical in the captured output.
const (
	probeCommand = "printf 'PROBE%s' DONE"
	toolProbe    = "PROBEDONE"
)

var probeStep = llmtest.Step{ToolCalls: []llm.ToolCall{
	llmtest.Call("c1", "bash", map[string]any{"command": probeCommand}),
}}

func TestProgramAsksBeforeRunningACommand(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		probeStep,
		{Chunks: []string{"Done.\n"}},
	}, permission.ModeWork, "run it\r")

	// Work mode auto-approves edits but still asks for commands.
	if !strings.Contains(out, "bash") {
		t.Errorf("the tool call was not shown:\n%s", out)
	}
	if !strings.Contains(out, "? printf") {
		t.Errorf("no confirmation was requested for a shell command:\n%s", out)
	}
	if strings.Contains(out, toolProbe) {
		t.Errorf("the command ran without approval:\n%s", out)
	}
	if strings.Contains(out, "Done.") {
		t.Errorf("the turn continued while a tool was still blocked:\n%s", out)
	}
}

func TestProgramApprovesAndRunsTheTool(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		probeStep,
		{Chunks: []string{"Done.\n"}},
	}, permission.ModeWork, "run it\r", "y")

	if !strings.Contains(out, toolProbe) {
		t.Errorf("the approved command did not run:\n%s", out)
	}
	if !strings.Contains(out, "Done.") {
		t.Errorf("the turn did not continue after the tool:\n%s", out)
	}
}

// Denying must release the tool and let the model carry on with the refusal
// as an observation, not hang the turn.
func TestProgramDenialReleasesTheTurn(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		probeStep,
		{Chunks: []string{"Understood.\n"}},
	}, permission.ModeWork, "run it\r", "n")

	if strings.Contains(out, toolProbe) {
		t.Errorf("the denied command ran anyway:\n%s", out)
	}
	if !strings.Contains(out, "Understood.") {
		t.Errorf("the turn did not resume after the denial:\n%s", out)
	}
}

// Plan mode decides on its own: a mutating command is refused with no prompt,
// and the refusal is visible rather than silent.
func TestProgramPlanModeBlocksWithATrace(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		{ToolCalls: []llm.ToolCall{
			llmtest.Call("c1", "bash", map[string]any{"command": "mkdir new-dir"}),
		}},
		{Chunks: []string{"I cannot do that here.\n"}},
	}, permission.ModePlan, "make a dir\r")

	if strings.Contains(out, "allow?") {
		t.Errorf("Plan mode should decide without prompting:\n%s", out)
	}
	if !strings.Contains(out, "blocked") {
		t.Errorf("the automatic block left no trace:\n%s", out)
	}
}

func TestProgramShowsAnErrorFromTheModel(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		{Err: errContext},
	}, permission.ModeAsk, "hola\r")

	if !strings.Contains(out, "boom") {
		t.Errorf("the provider error was not surfaced:\n%s", out)
	}
}

func TestProgramShiftTabShowsTheNewMode(t *testing.T) {
	// \x1b[Z is the escape sequence a terminal sends for shift+tab.
	out := runProgram(t, nil, permission.ModeAsk, "\x1b[Z")
	if !strings.Contains(out, "Plan") {
		t.Errorf("shift+tab did not switch to Plan mode:\n%s", out)
	}
}

type constErr string

func (e constErr) Error() string { return string(e) }

const errContext = constErr("boom: the provider exploded")

// Esc must stop a turn in flight: the answer stops arriving, the cancellation
// is announced, and nothing from the abandoned turn appears afterwards. In the
// Python prototype this only stopped the rendering while the work carried on.
func TestProgramEscCancelsAnInFlightTurn(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		{
			Chunks: []string{"starting up\n", "NEVER-ARRIVES\n"},
			Delay:  400 * time.Millisecond,
		},
	}, permission.ModeAsk, "go\r", "\x1b")

	if !strings.Contains(out, "starting up") {
		t.Errorf("the turn never started:\n%s", out)
	}
	if !strings.Contains(out, "cancelled") {
		t.Errorf("the cancellation was not announced:\n%s", out)
	}
	if strings.Contains(out, "NEVER-ARRIVES") {
		t.Errorf("output kept arriving after cancelling:\n%s", out)
	}
}

func newTUIStore(t *testing.T) *session.Store {
	t.Helper()
	s, err := session.Open(filepath.Join(t.TempDir(), "sessions.db"))
	if err != nil {
		t.Fatalf("session.Open: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// A completed turn must actually land on disk, not just in the model's
// in-memory history — persistence is a background command (see
// Model.persistTurn), so this also proves the background write really
// happens rather than silently being dropped.
func TestProgramPersistsATurnToDisk(t *testing.T) {
	store := newTUIStore(t)
	ctx := context.Background()
	sess, err := store.Create(ctx, "/proj")
	if err != nil {
		t.Fatal(err)
	}

	runProgramOpts(t, programOpts{
		steps:  []llmtest.Step{{Chunks: []string{"the answer\n"}}},
		mode:   permission.ModeAsk,
		keys:   []string{"a question\r"},
		store:  store,
		sessID: sess.ID,
	})

	// persistTurn runs in the background; give it a moment after the program
	// has already exited (which itself waited out the turn) before reading.
	var saved []llm.Message
	for range 20 {
		saved, err = store.Load(ctx, sess.ID)
		if err != nil {
			t.Fatalf("Load: %v", err)
		}
		if len(saved) > 0 {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if len(saved) < 2 {
		t.Fatalf("got %d persisted messages, want at least a user+assistant pair:\n%+v", len(saved), saved)
	}
	if saved[0].Role != llm.RoleUser || saved[0].Content != "a question" {
		t.Errorf("first message = %+v", saved[0])
	}
	if strings.TrimSpace(saved[len(saved)-1].Content) != "the answer" {
		t.Errorf("last message = %+v", saved[len(saved)-1])
	}
}

// History loaded from a resumed session must reach the model on the very
// first turn of the new process — this is what makes --continue actually
// continue the conversation instead of starting the model from scratch.
func TestProgramSeedsHistoryFromAResumedSession(t *testing.T) {
	store := newTUIStore(t)
	ctx := context.Background()
	sess, _ := store.Create(ctx, "/proj")
	store.Append(ctx, sess.ID, []llm.Message{
		{Role: llm.RoleUser, Content: "what is the project called"},
		{Role: llm.RoleAssistant, Content: "it is called kiwi"},
	})
	priorHistory, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatal(err)
	}

	fake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "follow-up answer"}}}
	_, m := runProgramOpts(t, programOpts{
		steps:   fake.Steps,
		mode:    permission.ModeAsk,
		keys:    []string{"and who made it\r"},
		history: priorHistory,
		store:   store,
		sessID:  sess.ID,
	})

	if len(m.history) < len(priorHistory) {
		t.Fatalf("model history shrank: got %d messages, had %d loaded", len(m.history), len(priorHistory))
	}
	if m.history[0].Content != "what is the project called" {
		t.Errorf("loaded history was not seeded into the model: %+v", m.history[0])
	}
}

// Compaction must summarize through the real provider, not a mock: the
// summarizer's output ends up stored on disk and adopted into the live
// session, replacing the verbose original.
func TestProgramCompactsAndPersistsTheSummary(t *testing.T) {
	store := newTUIStore(t)
	ctx := context.Background()
	sess, _ := store.Create(ctx, "/proj")

	// Seed enough turns to cross both compaction thresholds at once: more
	// messages than DefaultCompactOptions.KeepRecent (20), and enough total
	// characters to exceed its CharBudget. A single oversized message would
	// exceed the budget but not the count, and Compact treats anything within
	// KeepRecent messages of the end as "too recent to touch" regardless of
	// size — this needs real turns, not one giant one.
	var seed []llm.Message
	for i := range 15 {
		seed = append(seed,
			llm.Message{Role: llm.RoleUser, Content: strings.Repeat(fmt.Sprintf("turn%d ", i), 300)},
			llm.Message{Role: llm.RoleAssistant, Content: "ok"},
		)
	}
	if err := store.Append(ctx, sess.ID, seed); err != nil {
		t.Fatal(err)
	}
	history, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatal(err)
	}

	runProgramOpts(t, programOpts{
		steps: []llmtest.Step{
			{Text: "short reply"}, // the turn itself
			{Text: "the summary"}, // the summarizer's call, once persisted
		},
		mode:    permission.ModeAsk,
		keys:    []string{"go\r"},
		history: history,
		store:   store,
		sessID:  sess.ID,
	})

	var saved []llm.Message
	for range 30 {
		saved, err = store.Load(ctx, sess.ID)
		if err != nil {
			t.Fatal(err)
		}
		if len(saved) > 0 && saved[0].Content == "(summary of earlier context in this session)" {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	if len(saved) == 0 || saved[0].Content != "(summary of earlier context in this session)" {
		t.Fatalf("the stored history was not compacted:\n%+v", saved)
	}
	if saved[1].Content != "the summary" {
		t.Errorf("the summarizer's output was not stored: %+v", saved[1])
	}
}
