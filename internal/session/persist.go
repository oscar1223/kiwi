package session

import (
	"context"
	"fmt"
	"strings"

	"github.com/oscar1223/kiwi/internal/llm"
)

// maxTitleRunes bounds the auto-generated session title shown by
// `kiwi session list`.
const maxTitleRunes = 60

// Persist saves one turn's messages, names the session from its first user
// message if it does not have a title yet, and compacts the stored history
// once it grows past budget. It returns the history the next turn should
// start from — the freshly compacted version when compaction fired, or the
// full history otherwise.
//
// Compaction failing (the summarizer call errors) does not fail Persist: the
// turn itself is already durably saved by the time compaction runs, so the
// safe fallback is to keep the uncompacted history and simply try again next
// turn, rather than lose a successfully persisted exchange over a transient
// summarization failure.
func Persist(ctx context.Context, store *Store, sessionID string, provider llm.Provider, turnMessages []llm.Message) ([]llm.Message, error) {
	if err := store.Append(ctx, sessionID, turnMessages); err != nil {
		return nil, fmt.Errorf("session: persisting turn: %w", err)
	}

	if title := firstUserTitle(turnMessages); title != "" {
		if meta, err := store.Get(ctx, sessionID); err == nil && meta.Title == "" {
			_ = store.SetTitle(ctx, sessionID, title) // best-effort; a naming failure should not fail the turn
		}
	}

	history, err := store.Load(ctx, sessionID)
	if err != nil {
		return nil, fmt.Errorf("session: reloading history: %w", err)
	}

	compacted, changed, err := Compact(ctx, provider, history, DefaultCompactOptions())
	if err != nil {
		return history, nil
	}
	if !changed {
		return history, nil
	}
	if err := store.Replace(ctx, sessionID, compacted); err != nil {
		return nil, fmt.Errorf("session: replacing compacted history: %w", err)
	}
	return compacted, nil
}

func firstUserTitle(msgs []llm.Message) string {
	if len(msgs) == 0 || msgs[0].Role != llm.RoleUser {
		return ""
	}
	return truncateTitle(msgs[0].Content)
}

func truncateTitle(s string) string {
	s = strings.TrimSpace(s)
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = strings.TrimSpace(s[:i])
	}
	r := []rune(s)
	if len(r) <= maxTitleRunes {
		return s
	}
	return string(r[:maxTitleRunes]) + "…"
}
