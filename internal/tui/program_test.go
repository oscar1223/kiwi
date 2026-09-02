package tui

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	tea "charm.land/bubbletea/v2"
	"github.com/oscar1223/kiwi/internal/agent"
	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/prompt"
	"github.com/oscar1223/kiwi/internal/session"
	"github.com/oscar1223/kiwi/internal/tools"
)

var ansiRE = regexp.MustCompile(`\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[()][AB012]|\x1b[=>]`)

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
	// workDir overrides the temp dir the program otherwise creates for
	// itself — needed by tests that must create sessions (session.Store)
	// against the same project dir the running program will use.
	workDir string
	// needsOnboarding and rebuild let a test exercise the setup wizard: with
	// it set, the Model starts with no Agent, the same shape a fresh install
	// is in, and rebuild stands in for Options.Rebuild (normally cmd/kiwi's
	// job) to hand back the fake agent once the wizard finishes.
	needsOnboarding bool
	rebuild         func() (*agent.Agent, string, error)
}

func runProgramOpts(t *testing.T, po programOpts) (string, *Model) {
	t.Helper()

	fake := &llmtest.Fake{Steps: po.steps}
	ev := NewEvents()
	broker := permission.NewBroker(po.mode, ev)
	dir := po.workDir
	if dir == "" {
		dir = t.TempDir()
	}

	var a *agent.Agent
	if !po.needsOnboarding {
		a = &agent.Agent{
			Provider: fake,
			Tools:    tools.Default(dir, broker),
			System:   "test",
		}
	}

	m := New(Options{
		Agent:           a,
		Broker:          broker,
		WorkDir:         dir,
		ModelLabel:      "fake/fake-1",
		PromptOptions:   prompt.Options{WorkingDir: dir},
		Events:          ev,
		History:         po.history,
		Store:           po.store,
		SessionID:       po.sessID,
		NeedsOnboarding: po.needsOnboarding,
		Rebuild:         po.rebuild,
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

	return ansiRE.ReplaceAllString(out.String(), ""), m
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
	}, permission.ModeAsk, "run it\r")

	// Ask mode confirms every command.
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
	}, permission.ModeAsk, "run it\r", "y")

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
	}, permission.ModeAsk, "run it\r", "n")

	if strings.Contains(out, toolProbe) {
		t.Errorf("the denied command ran anyway:\n%s", out)
	}
	if !strings.Contains(out, "Understood.") {
		t.Errorf("the turn did not resume after the denial:\n%s", out)
	}
}

// Work mode auto-approves safe commands, same as edits.
func TestProgramWorkModeRunsSafeCommandWithoutAsking(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		probeStep,
		{Chunks: []string{"Done.\n"}},
	}, permission.ModeWork, "run it\r")

	if !strings.Contains(out, toolProbe) {
		t.Errorf("the safe command did not run without approval:\n%s", out)
	}
	if !strings.Contains(out, "Done.") {
		t.Errorf("the turn did not continue after the tool:\n%s", out)
	}
	if strings.Contains(out, "? printf") {
		t.Errorf("work mode should not confirm a safe command:\n%s", out)
	}
}

// Dangerous commands still confirm even in Work mode.
func TestProgramWorkModeStillAsksForDangerousCommands(t *testing.T) {
	out := runProgram(t, []llmtest.Step{
		{ToolCalls: []llm.ToolCall{
			llmtest.Call("c1", "bash", map[string]any{"command": "rm -rf /tmp/probe"}),
		}},
		{Chunks: []string{"Done.\n"}},
	}, permission.ModeWork, "run it\r")

	if !strings.Contains(out, "rm -rf /tmp/probe") || !strings.Contains(out, "?") {
		t.Errorf("no confirmation was requested for a dangerous command:\n%s", out)
	}
	if strings.Contains(out, "Done.") {
		t.Errorf("the turn continued while a dangerous command was still blocked:\n%s", out)
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
	// messages than DefaultKeepRecent (20), and enough estimated tokens to
	// exceed the budget for the fake provider's model. A single oversized
	// message would exceed the budget but not the count, and Compact treats
	// anything within KeepRecent messages of the end as "too recent to touch"
	// regardless of size — this needs real turns, not one giant one.
	//
	// The filler is sized from the budget rather than hard-coded, so this test
	// keeps testing compaction rather than quietly becoming a no-op the next
	// time the window table or historyShare moves.
	const turns = 15
	budget := session.CompactOptionsFor(new(llmtest.Fake).Model()).TokenBudget
	filler := strings.Repeat("filler ", (budget*4)/(turns*7)+1)

	var seed []llm.Message
	for i := range turns {
		seed = append(seed,
			llm.Message{Role: llm.RoleUser, Content: fmt.Sprintf("turn%d %s", i, filler)},
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

// End-to-end: typing "/memory", picking "View recent messages", entering a
// count, and seeing the prior turn echoed back — the full pipeline from
// slash-command dispatch through runFlow's goroutine, Events.Pick/Text,
// Update's rendering, and back. Unit tests above cover each piece in
// isolation; this is the one that would catch a wiring mistake between them.
func TestProgramMemoryFlowEndToEnd(t *testing.T) {
	// /memory reads the saved-note files to label its menu; point them at a
	// temp dir so the test never touches the developer's real memory.
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	out, _ := runProgramOpts(t, programOpts{
		mode: permission.ModeAsk,
		history: []llm.Message{
			{Role: llm.RoleUser, Content: "what does kiwi do"},
			{Role: llm.RoleAssistant, Content: "it is a local coding agent"},
		},
		keys: []string{"/memory\r", "\r" /* Conversation */, "\r" /* View recent messages */, "3\r" /* how many */},
	})

	if !strings.Contains(out, "Memory") {
		t.Errorf("the memory picker never opened:\n%s", out)
	}
	if !strings.Contains(out, "what does kiwi do") {
		t.Errorf("the prior user turn was not shown:\n%s", out)
	}
	if !strings.Contains(out, "it is a local coding agent") {
		t.Errorf("the prior assistant turn was not shown:\n%s", out)
	}
}

// Ordinary chat input must stay blocked for the whole time a flow is
// running, not just while a picker happens to be visible — the gap between
// dispatch and the first prompt appearing must not let a message slip through.
func TestProgramCannotSubmitChatWhileAFlowIsOpen(t *testing.T) {
	out, _ := runProgramOpts(t, programOpts{
		mode: permission.ModeAsk,
		keys: []string{"/memory\r", "should not reach the model\r"},
	})

	if strings.Contains(out, "should not reach the model") {
		t.Errorf("chat text typed while the memory picker was open leaked through:\n%s", out)
	}
}

// End-to-end: /mcp → "+ Add MCP server" → remote → URL → HTTP transport →
// no headers, and the server then shows up in the list. This exercises the
// full remote-server config path added for M5 (previously only stdio was
// reachable from the TUI at all).
func TestProgramAddRemoteMCPServerEndToEnd(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	out, _ := runProgramOpts(t, programOpts{
		mode: permission.ModeAsk,
		keys: []string{
			"/mcp\r",                    // open the picker: only "+ Add MCP server" exists
			"\r",                        // pick it
			"myserver\r",                // server name
			"\x1b[B\r",                  // down to "Remote URL (HTTP or SSE)", then select it
			"https://example.com/mcp\r", // URL
			"\r",                        // transport: Streamable HTTP (first option)
			"\r",                        // headers: leave empty
			"\x1b",                      // esc: close the /mcp picker once we're back at the list
		},
	})

	if !strings.Contains(out, "saved") {
		t.Errorf("the server was not reported as saved:\n%s", out)
	}
	// Proves the remote branch specifically ran, not a coincidental stdio
	// path with the same number of prompts: the server's own list entry (via
	// describeServer) must show its URL and transport, and the flow must
	// never have asked for a stdio command at all.
	if !strings.Contains(out, "https://example.com/mcp") {
		t.Errorf("the server URL never appeared, so the remote branch may not have run:\n%s", out)
	}
	if strings.Contains(out, "Command to run") {
		t.Errorf("the stdio branch ran instead of the remote one:\n%s", out)
	}
}

// End-to-end: the setup wizard runs automatically (no /command needed) on a
// session with no provider configured, writes .env and kiwi.json, and the
// very next chat turn reaches the freshly rebuilt agent — proving the
// agentRebuiltMsg-before-flowDoneMsg ordering that rules out the nil-Agent
// race described in applyRebuild's doc comment.
func TestProgramOnboardingWizardEndToEnd(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Cleanup(func() { os.Unsetenv("ANTHROPIC_API_KEY") }) // onboardingFlow sets it directly, outside t.Setenv's tracking

	postWizardFake := &llmtest.Fake{Steps: []llmtest.Step{{Text: "hello after setup"}}}
	rebuild := func() (*agent.Agent, string, error) {
		return &agent.Agent{
			Provider: postWizardFake,
			Tools:    tools.NewRegistry(),
			System:   "post-onboarding",
		}, "anthropic/claude-sonnet-5", nil
	}

	out, m := runProgramOpts(t, programOpts{
		needsOnboarding: true,
		rebuild:         rebuild,
		mode:            permission.ModeAsk,
		keys: []string{
			"\r",                // pick provider: Anthropic (first option)
			"\r",                // model name: accept the default
			"sk-ant-test-key\r", // paste the API key
			"go ahead\r",        // a real chat turn once onboarding is done
		},
	})

	if !strings.Contains(out, "Setup complete") {
		t.Errorf("the wizard never reported completion:\n%s", out)
	}
	if !strings.Contains(out, "hello after setup") {
		t.Errorf("the post-onboarding agent was never reached:\n%s", out)
	}
	if strings.Contains(out, "sk-ant-test-key") {
		t.Errorf("the API key leaked into scrollback in plaintext:\n%s", out)
	}
	if m.opts.Agent == nil {
		t.Error("Agent is still nil after onboarding completed")
	}

	f, err := config.OpenEnvFile()
	if err != nil {
		t.Fatalf("OpenEnvFile: %v", err)
	}
	if v, ok := f.Get("ANTHROPIC_API_KEY"); !ok || v != "sk-ant-test-key" {
		t.Errorf("ANTHROPIC_API_KEY in .env = (%q, %v)", v, ok)
	}

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Current != "sonnet" {
		t.Errorf("cfg.Current = %q, want sonnet", cfg.Current)
	}
}

// Cancelling the wizard partway through must leave the TUI safely idle —
// chat still blocked (no agent exists), no panic — rather than half-applied.
func TestProgramOnboardingCancelledLeavesChatSafelyBlocked(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	out, m := runProgramOpts(t, programOpts{
		needsOnboarding: true,
		rebuild: func() (*agent.Agent, string, error) {
			t.Fatal("rebuild must not be called when onboarding was cancelled")
			return nil, "", nil
		},
		mode: permission.ModeAsk,
		keys: []string{
			"\x1b", // esc during the provider picker: cancel
			"this should not reach any agent\r",
		},
	})

	if !strings.Contains(out, "Setup skipped") {
		t.Errorf("cancelling did not report skipped setup:\n%s", out)
	}
	if m.opts.Agent != nil {
		t.Error("Agent should still be nil after a cancelled wizard")
	}
	// enter must have been refused, not silently swallowed: the typed text
	// stays sitting in the input box rather than being cleared by a submit()
	// that then (with a nil Agent) would have panicked.
	if got := m.input.Value(); got != "this should not reach any agent" {
		t.Errorf("input.Value() = %q, want the unsent text still there", got)
	}
}

// /settings groups the existing flows behind one menu — verified end-to-end
// by opening it and picking "Model profiles", landing exactly where /model
// itself would.
func TestProgramSettingsMenuOpensModelFlow(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())

	out, _ := runProgramOpts(t, programOpts{
		mode: permission.ModeAsk,
		keys: []string{
			"/settings\r", // open settings
			"\r",          // "Model profiles" is the first option
			"\x1b",        // esc out of the model picker
			"\x1b",        // esc out of the settings picker
		},
	})

	if !strings.Contains(out, "Settings") {
		t.Errorf("the settings menu never opened:\n%s", out)
	}
	if !strings.Contains(out, "Model profile") {
		t.Errorf("picking \"Model profiles\" did not open the model flow:\n%s", out)
	}
}

// /sessions must let the user jump between saved conversations for the
// current project without losing anything — each turn is already persisted
// immediately, so a switch is just swapping which history the model shows.
func TestProgramSessionsFlowSwitchesActiveHistory(t *testing.T) {
	store := newTUIStore(t)
	ctx := context.Background()
	dir := t.TempDir()

	first, err := store.Create(ctx, dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Append(ctx, first.ID, []llm.Message{
		{Role: llm.RoleUser, Content: "first session question"},
		{Role: llm.RoleAssistant, Content: "first session answer"},
	}); err != nil {
		t.Fatal(err)
	}

	second, err := store.Create(ctx, dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Append(ctx, second.ID, []llm.Message{
		{Role: llm.RoleUser, Content: "second session question"},
		{Role: llm.RoleAssistant, Content: "second session answer"},
	}); err != nil {
		t.Fatal(err)
	}

	// updated_at has only second resolution, so which of first/second sorts
	// first in sessionsFlow's list is not guaranteed by this test's timing —
	// ask the store the same way sessionsFlow does, instead of assuming an
	// order, so the test does not depend on how fast it happens to run.
	listed, err := store.List(ctx, dir, 50)
	if err != nil {
		t.Fatal(err)
	}
	downs := -1
	for i, s := range listed {
		if s.ID == first.ID {
			downs = i
		}
	}
	if downs < 0 {
		t.Fatalf("the first session did not show up in List(): %+v", listed)
	}

	_, m := runProgramOpts(t, programOpts{
		mode:    permission.ModeAsk,
		workDir: dir,
		store:   store,
		sessID:  second.ID,
		history: []llm.Message{
			{Role: llm.RoleUser, Content: "second session question"},
			{Role: llm.RoleAssistant, Content: "second session answer"},
		},
		keys: []string{
			"/sessions\r",                          // open the picker
			strings.Repeat("\x1b[B", downs) + "\r", // move to "first", then pick it
		},
	})

	if m.opts.SessionID != first.ID {
		t.Errorf("SessionID = %q, want the first session %q", m.opts.SessionID, first.ID)
	}
	if len(m.history) != 2 || m.history[0].Content != "first session question" {
		t.Errorf("history after switching = %+v, want the first session's messages", m.history)
	}
}

// Picking "+ New session" must start with empty history and a fresh id,
// distinct from every existing session for the project.
func TestProgramSessionsFlowCreatesNewSession(t *testing.T) {
	store := newTUIStore(t)
	ctx := context.Background()
	dir := t.TempDir()

	existing, err := store.Create(ctx, dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Append(ctx, existing.ID, []llm.Message{
		{Role: llm.RoleUser, Content: "old question"},
	}); err != nil {
		t.Fatal(err)
	}

	_, m := runProgramOpts(t, programOpts{
		mode:    permission.ModeAsk,
		workDir: dir,
		store:   store,
		sessID:  existing.ID,
		history: []llm.Message{{Role: llm.RoleUser, Content: "old question"}},
		keys: []string{
			"/sessions\r",
			"\x1b[B\r", // down once, to "+ New session", then pick it
		},
	})

	if m.opts.SessionID == existing.ID || m.opts.SessionID == "" {
		t.Errorf("SessionID after creating a new session = %q, want a fresh id", m.opts.SessionID)
	}
	if len(m.history) != 0 {
		t.Errorf("history after creating a new session = %+v, want empty", m.history)
	}
}

// /compact must summarize on demand even when the conversation is nowhere
// near the automatic threshold — that is the entire point of asking for it by
// hand — and the result has to reach both the screen and the database.
func TestProgramCompactOnDemand(t *testing.T) {
	store := newTUIStore(t)
	ctx := context.Background()
	sess, _ := store.Create(ctx, "/proj")

	var seed []llm.Message
	for i := range 10 {
		seed = append(seed,
			llm.Message{Role: llm.RoleUser, Content: fmt.Sprintf("question %d", i)},
			llm.Message{Role: llm.RoleAssistant, Content: fmt.Sprintf("answer %d", i)},
		)
	}
	if err := store.Append(ctx, sess.ID, seed); err != nil {
		t.Fatal(err)
	}
	history, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatal(err)
	}

	out, m := runProgramOpts(t, programOpts{
		steps:   []llmtest.Step{{Text: "the summary"}},
		mode:    permission.ModeAsk,
		keys:    []string{"/compact\r"},
		history: history,
		store:   store,
		sessID:  sess.ID,
	})

	if !strings.Contains(out, "compacted") {
		t.Errorf("the compaction was never reported to the user:\n%s", out)
	}
	if len(m.history) >= len(history) {
		t.Errorf("history is %d messages, no shorter than the original %d", len(m.history), len(history))
	}
	if len(m.history) == 0 || m.history[0].Content != "(summary of earlier context in this session)" {
		t.Fatalf("the live history was not replaced with the compacted one: %+v", m.history)
	}

	saved, err := store.Load(ctx, sess.ID)
	if err != nil {
		t.Fatal(err)
	}
	if len(saved) == 0 || saved[0].Content != "(summary of earlier context in this session)" {
		t.Errorf("the compacted history was not persisted:\n%+v", saved)
	}
}

// An empty conversation must say so rather than calling the model.
func TestProgramCompactWithNothingToCompact(t *testing.T) {
	out, _ := runProgramOpts(t, programOpts{
		mode: permission.ModeAsk,
		keys: []string{"/compact\r"},
	})

	if !strings.Contains(out, "empty") {
		t.Errorf("compacting an empty conversation did not explain itself:\n%s", out)
	}
}

// Typing @file must reach the model with the file inlined, while the
// transcript keeps showing what the user actually typed.
func TestProgramExpandsFileMentions(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "note.txt"), []byte("MENTIONED-BODY"), 0o644); err != nil {
		t.Fatal(err)
	}

	out, m := runProgramOpts(t, programOpts{
		steps:   []llmtest.Step{{Text: "read it"}},
		mode:    permission.ModeAsk,
		workDir: dir,
		keys:    []string{"summarize @note.txt\r"},
	})

	if !strings.Contains(out, "attached note.txt") {
		t.Errorf("the attachment was not reported:\n%s", out)
	}
	if len(m.history) == 0 {
		t.Fatal("no history was recorded")
	}
	if !strings.Contains(m.history[0].Content, "MENTIONED-BODY") {
		t.Errorf("the file was not inlined into the message sent to the model: %q", m.history[0].Content)
	}
}

func TestProgramReportsUnreadableMentions(t *testing.T) {
	out, _ := runProgramOpts(t, programOpts{
		steps: []llmtest.Step{{Text: "ok"}},
		mode:  permission.ModeAsk,
		keys:  []string{"look at @missing.txt\r"},
	})

	if !strings.Contains(out, "could not read") {
		t.Errorf("a mention that resolved to nothing was not reported:\n%s", out)
	}
}
