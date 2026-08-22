package session

import (
	"context"
	"fmt"
	"strings"

	"github.com/oscar1223/kiwi/internal/llm"
)

// CompactOptions bounds when and how much history gets condensed.
type CompactOptions struct {
	// CharBudget is the rough size, in characters, above which the oldest
	// messages get summarized. Character count is a stand-in for tokens — a
	// real tokenizer is not worth the dependency for a threshold this coarse.
	CharBudget int
	// KeepRecent is how many trailing messages, at minimum, stay verbatim.
	// The actual cut point moves forward from here to the next turn
	// boundary; see Compact.
	KeepRecent int
}

func DefaultCompactOptions() CompactOptions {
	return CompactOptions{CharBudget: 24000, KeepRecent: 20}
}

const summaryMarker = "(summary of earlier context in this session)"

// Compact condenses the oldest part of history into a short summary once it
// grows past opts.CharBudget, keeping the most recent messages untouched. It
// reports changed=false, and returns history as given, when nothing needed to
// happen or nothing safely could.
//
// The cut between "old" and "recent" must land on a turn boundary — right
// before a user message — never inside one. Every turn is
// [user, assistant(+tool_calls)?, tool*, ..., assistant(final)]; both provider
// adapters require a tool result to immediately follow the assistant message
// that requested it, so summarizing away half of that pairing would hand the
// next request a dangling tool_use or an orphaned tool_result and the API
// would reject it outright.
//
// The boundary search walks backward from the KeepRecent cut, not forward:
// walking forward can skip clean over the turn straddling the cut and land on
// the *next* turn, which would summarize away a tool round-trip that just
// happened. Walking backward always keeps at least KeepRecent messages —
// occasionally a few more, never fewer — by extending "recent" to cover the
// whole straddling turn instead of discarding it.
func Compact(ctx context.Context, provider llm.Provider, history []llm.Message, opts CompactOptions) ([]llm.Message, bool, error) {
	if estimateChars(history) <= opts.CharBudget {
		return history, false, nil
	}

	cut := len(history) - opts.KeepRecent
	if cut <= 0 {
		return history, false, nil
	}
	for cut > 0 && history[cut].Role != llm.RoleUser {
		cut--
	}
	if cut == 0 {
		// No turn boundary before the cut: it is all one giant turn. Leave it
		// for next time rather than risk an unsafe split.
		return history, false, nil
	}

	old, recent := history[:cut], history[cut:]

	summary, err := summarize(ctx, provider, old)
	if err != nil {
		return nil, false, fmt.Errorf("session: compacting history: %w", err)
	}

	condensed := make([]llm.Message, 0, len(recent)+2)
	condensed = append(condensed,
		llm.Message{Role: llm.RoleUser, Content: summaryMarker},
		llm.Message{Role: llm.RoleAssistant, Content: summary},
	)
	condensed = append(condensed, recent...)
	return condensed, true, nil
}

func estimateChars(history []llm.Message) int {
	total := 0
	for _, m := range history {
		total += len(m.Content)
	}
	return total
}

const summarizePrompt = `Summarize the conversation below between a user and the Kiwi coding agent in a
short paragraph (6 lines maximum). Preserve decisions made, files or paths
mentioned, and anything needed to continue the conversation naturally.
Tool calls and their raw output are omitted; infer their effect from what the
assistant said about them.

`

// summarize asks the model itself for a condensed version of old messages.
// It talks to the provider directly rather than through agent.Agent: a
// summarization call has no tools and no system prompt of its own, so the
// full turn loop would be pure overhead.
func summarize(ctx context.Context, provider llm.Provider, history []llm.Message) (string, error) {
	var transcript strings.Builder
	for _, m := range history {
		switch m.Role {
		case llm.RoleUser:
			if m.Content != "" && m.Content != summaryMarker {
				fmt.Fprintf(&transcript, "User: %s\n", m.Content)
			}
		case llm.RoleAssistant:
			if m.Content != "" {
				fmt.Fprintf(&transcript, "Kiwi: %s\n", m.Content)
			}
		}
	}

	req := llm.Request{
		Messages:  []llm.Message{{Role: llm.RoleUser, Content: summarizePrompt + transcript.String()}},
		MaxTokens: 512,
	}

	var out strings.Builder
	for ev, err := range provider.Stream(ctx, req) {
		if err != nil {
			return "", err
		}
		if ev.Type == llm.EventDone && ev.Message != nil {
			out.WriteString(ev.Message.Content)
		}
	}
	if out.Len() == 0 {
		return "", fmt.Errorf("summarization produced no text")
	}
	return out.String(), nil
}
