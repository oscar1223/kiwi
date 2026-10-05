package remote

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
)

// DefaultLiveInterval is how often the live message may be edited. Telegram
// allows about one message per second per chat, edits included, and the
// final answer still has to get through.
const DefaultLiveInterval = 2500 * time.Millisecond

// liveLines is how many tool calls the live message shows; older ones are
// summarised as a count.
const liveLines = 12

// liveObserver is an agent.Observer that keeps one Telegram message up to
// date with what the agent is doing.
//
// It does not stream: the model's text is not sent until the turn ends, as the
// reply. Tool calls are milestones, collected as they happen and flushed to
// the message by a ticker, at most once per interval. The agent calls the
// observer from its own goroutine (and task subagents from theirs), so all
// state is behind mu, and the network is never touched while holding it.
type liveObserver struct {
	conv     Conversation
	interval time.Duration
	log      func(string)

	mu      sync.Mutex
	lines   []string
	byID    map[string]int // tool call ID -> index in lines
	calls   int
	dirty   bool
	started time.Time

	// Owned by the flushing goroutine, and by finish once it has stopped.
	msgID    int64
	lastEdit time.Time

	stop chan struct{}
	done chan struct{}
}

func startLive(ctx context.Context, conv Conversation, interval time.Duration, log func(string)) *liveObserver {
	if interval <= 0 {
		interval = DefaultLiveInterval
	}
	o := &liveObserver{
		conv:     conv,
		interval: interval,
		log:      log,
		byID:     map[string]int{},
		started:  time.Now(),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}
	go o.run(ctx)
	return o
}

func (o *liveObserver) OnText(string)     {}
func (o *liveObserver) OnUsage(llm.Usage) {}

func (o *liveObserver) OnToolCall(call llm.ToolCall) {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.byID[call.ID] = len(o.lines)
	o.lines = append(o.lines, "● "+describeCall(call))
	o.calls++
	o.dirty = true
}

func (o *liveObserver) OnToolResult(call llm.ToolCall, _ string, isErr bool) {
	if !isErr {
		return
	}
	o.mu.Lock()
	defer o.mu.Unlock()
	if i, ok := o.byID[call.ID]; ok {
		o.lines[i] = "✗" + strings.TrimPrefix(o.lines[i], "●")
		o.dirty = true
	}
}

// run flushes the message once per interval while there is something new.
func (o *liveObserver) run(ctx context.Context) {
	defer close(o.done)
	t := time.NewTicker(o.interval)
	defer t.Stop()
	for {
		select {
		case <-o.stop:
			return
		case <-ctx.Done():
			return
		case <-t.C:
			o.flush(ctx, o.render("⏳ Trabajando"))
		}
	}
}

// finish stops the ticker and leaves the message in its final state. It
// waits out the interval since the last edit, so even the closing edit never
// goes faster than the limit.
func (o *liveObserver) finish(ctx context.Context, ok bool) {
	close(o.stop)
	<-o.done

	o.mu.Lock()
	sentAny := o.msgID != 0 || o.calls > 0
	o.mu.Unlock()
	if !sentAny {
		return // no tool calls: a plain answer needs no progress message
	}

	header := "✓ Hecho"
	if !ok {
		header = "✗ Interrumpido"
	}
	o.mu.Lock()
	o.dirty = true // the header changed even if the lines did not
	o.mu.Unlock()
	if wait := o.interval - time.Since(o.lastEdit); !o.lastEdit.IsZero() && wait > 0 {
		select {
		case <-ctx.Done():
			return
		case <-time.After(wait):
		}
	}
	o.flush(ctx, o.render(header))
}

// render returns the message text if anything changed since the last flush,
// or "" if not.
func (o *liveObserver) render(header string) string {
	o.mu.Lock()
	defer o.mu.Unlock()
	if !o.dirty {
		return ""
	}
	o.dirty = false

	elapsed := time.Since(o.started).Round(time.Second)
	var b strings.Builder
	fmt.Fprintf(&b, "%s · %d %s · %s", header, o.calls, plural(o.calls, "paso", "pasos"), elapsed)
	lines := o.lines
	if hidden := len(lines) - liveLines; hidden > 0 {
		fmt.Fprintf(&b, "\n… y %d antes", hidden)
		lines = lines[hidden:]
	}
	for _, l := range lines {
		b.WriteString("\n")
		b.WriteString(l)
	}
	return b.String()
}

func (o *liveObserver) flush(ctx context.Context, text string) {
	if text == "" {
		return
	}
	var err error
	if o.msgID == 0 {
		o.msgID, err = o.conv.Send(ctx, text)
	} else {
		err = o.conv.Edit(ctx, o.msgID, text)
	}
	o.lastEdit = time.Now()
	if err != nil && ctx.Err() == nil && o.log != nil {
		// A missed edit is not worth stopping for: the next one carries the
		// same lines, and the answer arrives either way.
		o.log(fmt.Sprintf("live message: %v", err))
	}
}

// describeCall is one line for a tool call: its name and the argument that
// says the most about it.
func describeCall(call llm.ToolCall) string {
	var args map[string]any
	_ = json.Unmarshal(call.Input, &args)

	for _, key := range []string{"command", "description", "path", "file_path", "pattern", "query", "url", "name"} {
		if v, ok := args[key].(string); ok && strings.TrimSpace(v) != "" {
			return call.Name + ": " + truncateRunes(v, 70)
		}
	}
	return call.Name
}

func plural(n int, one, many string) string {
	if n == 1 {
		return one
	}
	return many
}
