package remote

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
)

// fakeConv records every Send and Edit with when it happened.
type fakeConv struct {
	mu     sync.Mutex
	ops    []convOp
	nextID int64
}

type convOp struct {
	at   time.Time
	edit bool
	id   int64
	text string
}

func (c *fakeConv) Send(_ context.Context, text string) (int64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.nextID++
	c.ops = append(c.ops, convOp{at: time.Now(), id: c.nextID, text: text})
	return c.nextID, nil
}

func (c *fakeConv) Edit(_ context.Context, id int64, text string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.ops = append(c.ops, convOp{at: time.Now(), edit: true, id: id, text: text})
	return nil
}

func (c *fakeConv) snapshot() []convOp {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]convOp(nil), c.ops...)
}

func call(id, name string, input any) llm.ToolCall { return llmtest.Call(id, name, input) }

func TestLiveIsThrottled(t *testing.T) {
	const interval = 50 * time.Millisecond
	conv := &fakeConv{}
	o := startLive(context.Background(), conv, interval, nil)

	// Many tool calls, from several goroutines at once like task subagents,
	// for well over the interval.
	start := time.Now()
	var wg sync.WaitGroup
	for g := range 4 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range 75 {
				c := call(fmt.Sprintf("%d-%d", g, i), "bash", map[string]string{"command": "go test ./..."})
				o.OnToolCall(c)
				o.OnToolResult(c, "ok", i%10 == 0)
				time.Sleep(2 * time.Millisecond)
			}
		}()
	}
	wg.Wait()
	o.finish(context.Background(), true)
	elapsed := time.Since(start)

	ops := conv.snapshot()
	if len(ops) == 0 {
		t.Fatal("nothing was sent")
	}
	// Timers are not exact; allow a little jitter but nothing like a burst.
	for i := 1; i < len(ops); i++ {
		if gap := ops[i].at.Sub(ops[i-1].at); gap < interval*8/10 {
			t.Errorf("updates %d and %d were %s apart, under the %s interval", i-1, i, gap, interval)
		}
	}
	if max := int(elapsed/interval) + 2; len(ops) > max {
		t.Errorf("%d updates in %s, want at most %d", len(ops), elapsed, max)
	}
	if !strings.Contains(ops[len(ops)-1].text, "300 pasos") {
		t.Errorf("the final update should count every call:\n%s", ops[len(ops)-1].text)
	}
}

func TestLiveSendsOnceThenEdits(t *testing.T) {
	conv := &fakeConv{}
	o := startLive(context.Background(), conv, 20*time.Millisecond, nil)

	o.OnToolCall(call("1", "read_file", map[string]string{"path": "go.mod"}))
	time.Sleep(60 * time.Millisecond)
	o.OnToolCall(call("2", "bash", map[string]string{"command": "make test"}))
	o.finish(context.Background(), true)

	ops := conv.snapshot()
	if len(ops) < 2 {
		t.Fatalf("got %d operations, want a send and at least one edit", len(ops))
	}
	if ops[0].edit {
		t.Error("the first update should create the message")
	}
	for _, op := range ops[1:] {
		if !op.edit || op.id != ops[0].id {
			t.Errorf("later updates should edit message %d, got %+v", ops[0].id, op)
		}
	}
	final := ops[len(ops)-1].text
	for _, want := range []string{"✓ Hecho", "2 pasos", "● read_file: go.mod", "● bash: make test"} {
		if !strings.Contains(final, want) {
			t.Errorf("final message lacks %q:\n%s", want, final)
		}
	}
}

func TestLiveNothingWithoutToolCalls(t *testing.T) {
	conv := &fakeConv{}
	o := startLive(context.Background(), conv, 10*time.Millisecond, nil)
	o.OnText("una respuesta sin herramientas")
	time.Sleep(40 * time.Millisecond)
	o.finish(context.Background(), true)
	if ops := conv.snapshot(); len(ops) != 0 {
		t.Errorf("a turn with no tool calls should not post a progress message, got %+v", ops)
	}
}

func TestLiveMarksFailuresAndInterruptions(t *testing.T) {
	conv := &fakeConv{}
	o := startLive(context.Background(), conv, 10*time.Millisecond, nil)
	c := call("1", "bash", map[string]string{"command": "go build"})
	o.OnToolCall(c)
	o.OnToolResult(c, "exit status 1", true)
	o.finish(context.Background(), false)

	ops := conv.snapshot()
	final := ops[len(ops)-1].text
	if !strings.Contains(final, "✗ bash: go build") || !strings.HasPrefix(final, "✗ Interrumpido") {
		t.Errorf("final message:\n%s", final)
	}
}

func TestLiveKeepsTheLastLines(t *testing.T) {
	conv := &fakeConv{}
	o := startLive(context.Background(), conv, 10*time.Millisecond, nil)
	for i := range 20 {
		o.OnToolCall(call(fmt.Sprint(i), "read_file", map[string]string{"path": fmt.Sprintf("f%02d.go", i)}))
	}
	o.finish(context.Background(), true)

	final := conv.snapshot()[len(conv.snapshot())-1].text
	if !strings.Contains(final, "… y 8 antes") || strings.Contains(final, "f07.go") || !strings.Contains(final, "f19.go") {
		t.Errorf("want the last %d calls and a count of the rest:\n%s", liveLines, final)
	}
	if n := utf16Len(final); n > maxMessage {
		t.Errorf("live message is %d units, over the limit", n)
	}
}

func TestSessionShowsLiveProgress(t *testing.T) {
	s, _, _ := newTestSession(t,
		llmtest.Step{ToolCalls: []llm.ToolCall{
			llmtest.Call("c1", "write_file", map[string]string{"path": "a.txt", "content": "x"}),
		}},
		llmtest.Step{Text: "Listo."},
	)
	s.LiveInterval = 10 * time.Millisecond
	conv := &fakeConv{}

	reply := s.Handle(context.Background(), msg("crea a.txt"), conv)
	if reply != "Listo." {
		t.Errorf("reply = %q", reply)
	}
	ops := conv.snapshot()
	if len(ops) == 0 {
		t.Fatal("no progress message during a turn with tool calls")
	}
	if final := ops[len(ops)-1].text; !strings.Contains(final, "✓ Hecho") || !strings.Contains(final, "write_file: a.txt") {
		t.Errorf("final progress message:\n%s", final)
	}
}

func TestDescribeCall(t *testing.T) {
	tests := []struct {
		call llm.ToolCall
		want string
	}{
		{call("1", "bash", map[string]string{"command": "go test ./..."}), "bash: go test ./..."},
		{call("2", "task", map[string]string{"description": "buscar usos", "prompt": "…"}), "task: buscar usos"},
		{call("3", "todo_read", map[string]string{}), "todo_read"},
		{llm.ToolCall{ID: "4", Name: "weird", Input: []byte("not json")}, "weird"},
		{call("5", "bash", map[string]string{"command": "echo " + strings.Repeat("a", 200)}), "bash: echo " + strings.Repeat("a", 64) + "…"},
	}
	for _, tt := range tests {
		if got := describeCall(tt.call); got != tt.want {
			t.Errorf("describeCall(%s) = %q, want %q", tt.call.Name, got, tt.want)
		}
	}
}
