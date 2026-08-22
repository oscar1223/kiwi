// Package session persists conversations to SQLite so they survive restarts,
// crashes and machine reboots.
//
// Storage is per-project: a session belongs to the working directory it was
// started in, identified by a hash of its absolute path, which is how
// `kiwi --continue` knows which conversation to pick up without asking.
package session

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// ErrNotFound is returned when a session lookup matches nothing.
var ErrNotFound = errors.New("session: not found")

// Meta is a session's metadata, without its messages.
type Meta struct {
	ID         string
	ProjectDir string
	Title      string
	CreatedAt  time.Time
	UpdatedAt  time.Time
}

// Store is a SQLite-backed session store. It is safe for concurrent use.
type Store struct {
	db *sql.DB
}

const schema = `
CREATE TABLE IF NOT EXISTS sessions (
	id           TEXT PRIMARY KEY,
	project_dir  TEXT NOT NULL,
	project_hash TEXT NOT NULL,
	title        TEXT NOT NULL DEFAULT '',
	created_at   INTEGER NOT NULL,
	updated_at   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_hash, updated_at DESC);

CREATE TABLE IF NOT EXISTS messages (
	id            INTEGER PRIMARY KEY AUTOINCREMENT,
	session_id    TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
	seq           INTEGER NOT NULL,
	role          TEXT NOT NULL,
	content       TEXT NOT NULL DEFAULT '',
	tool_calls    TEXT,
	tool_call_id  TEXT NOT NULL DEFAULT '',
	tool_name     TEXT NOT NULL DEFAULT '',
	is_error      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);
`

// Open creates the database file (and its parent directory) if it does not
// already exist, and applies the schema.
func Open(path string) (*Store, error) {
	if err := ensureParentDir(path); err != nil {
		return nil, err
	}
	// _pragma sets foreign_keys and a busy timeout inline: modernc.org/sqlite
	// reads them from the DSN, so every connection in the pool gets them
	// without a separate PRAGMA round trip per connection.
	dsn := "file:" + path + "?_pragma=foreign_keys(1)&_pragma=busy_timeout(5000)"
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, err
	}
	// SQLite allows only one writer at a time; a single connection avoids
	// SQLITE_BUSY errors from the driver's own pool contending with itself.
	db.SetMaxOpenConns(1)

	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("session: applying schema: %w", err)
	}
	return &Store{db: db}, nil
}

func (s *Store) Close() error { return s.db.Close() }

// ProjectHash identifies a project by its absolute, cleaned path. Two kiwi
// invocations from the same directory always resolve to the same sessions,
// regardless of how the path was spelled (trailing slash, ./, symlinks are
// not resolved on purpose: a symlinked checkout is treated as its own
// project, matching what the user typed).
func ProjectHash(dir string) string {
	abs, err := filepath.Abs(dir)
	if err != nil {
		abs = dir
	}
	sum := sha256.Sum256([]byte(filepath.Clean(abs)))
	return hex.EncodeToString(sum[:])[:16]
}

// Create starts a new, empty session for projectDir.
func (s *Store) Create(ctx context.Context, projectDir string) (*Meta, error) {
	id, err := newID()
	if err != nil {
		return nil, err
	}
	now := time.Now()
	m := &Meta{ID: id, ProjectDir: projectDir, CreatedAt: now, UpdatedAt: now}

	_, err = s.db.ExecContext(ctx,
		`INSERT INTO sessions (id, project_dir, project_hash, title, created_at, updated_at)
		 VALUES (?, ?, ?, '', ?, ?)`,
		m.ID, m.ProjectDir, ProjectHash(projectDir), m.CreatedAt.Unix(), m.UpdatedAt.Unix(),
	)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Latest returns the most recently updated session for projectDir, or
// ErrNotFound if the project has none yet.
func (s *Store) Latest(ctx context.Context, projectDir string) (*Meta, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, project_dir, title, created_at, updated_at FROM sessions
		 WHERE project_hash = ? ORDER BY updated_at DESC LIMIT 1`,
		ProjectHash(projectDir),
	)
	return scanMeta(row)
}

// Get resolves id, accepting either a full id or an unambiguous prefix of
// one — the same convenience `git` gives you for commit hashes, since a
// session id is not something a user types often enough to memorise in full.
func (s *Store) Get(ctx context.Context, id string) (*Meta, error) {
	row := s.db.QueryRowContext(ctx,
		`SELECT id, project_dir, title, created_at, updated_at FROM sessions WHERE id = ?`, id)
	m, err := scanMeta(row)
	if err == nil {
		return m, nil
	}
	if !errors.Is(err, ErrNotFound) {
		return nil, err
	}

	// No exact id matched; try id as a prefix, reporting ambiguity rather
	// than silently picking whichever row the database returns first.
	matches, err := s.listByPrefix(ctx, id)
	if err != nil {
		return nil, err
	}
	switch len(matches) {
	case 0:
		return nil, ErrNotFound
	case 1:
		return &matches[0], nil
	default:
		return nil, fmt.Errorf("session: %q matches %d sessions, need more characters", id, len(matches))
	}
}

func (s *Store) listByPrefix(ctx context.Context, prefix string) ([]Meta, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, project_dir, title, created_at, updated_at FROM sessions
		 WHERE id LIKE ? ORDER BY updated_at DESC`,
		prefix+"%",
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanMetas(rows)
}

// List returns sessions for projectDir, most recently updated first.
func (s *Store) List(ctx context.Context, projectDir string, limit int) ([]Meta, error) {
	if limit <= 0 {
		limit = 50
	}
	rows, err := s.db.QueryContext(ctx,
		`SELECT id, project_dir, title, created_at, updated_at FROM sessions
		 WHERE project_hash = ? ORDER BY updated_at DESC LIMIT ?`,
		ProjectHash(projectDir), limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanMetas(rows)
}

// SetTitle names a session. Called once, from the first user message.
func (s *Store) SetTitle(ctx context.Context, id, title string) error {
	_, err := s.db.ExecContext(ctx, `UPDATE sessions SET title = ? WHERE id = ?`, title, id)
	return err
}

// Append persists new messages produced by one turn and bumps the session's
// updated_at, which is what makes it "the latest" for --continue.
//
// seq starts after whatever is already stored, so callers never have to track
// a running offset themselves.
func (s *Store) Append(ctx context.Context, sessionID string, msgs []llm.Message) error {
	if len(msgs) == 0 {
		return nil
	}
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	var next int
	if err := tx.QueryRowContext(ctx,
		`SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?`, sessionID,
	).Scan(&next); err != nil {
		return err
	}

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO messages (session_id, seq, role, content, tool_calls, tool_call_id, tool_name, is_error)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for i, m := range msgs {
		var toolCalls any
		if len(m.ToolCalls) > 0 {
			data, err := json.Marshal(m.ToolCalls)
			if err != nil {
				return err
			}
			toolCalls = string(data)
		}
		if _, err := stmt.ExecContext(ctx, sessionID, next+i, string(m.Role), m.Content,
			toolCalls, m.ToolCallID, m.ToolName, boolToInt(m.IsError)); err != nil {
			return err
		}
	}

	if _, err := tx.ExecContext(ctx,
		`UPDATE sessions SET updated_at = ? WHERE id = ?`, time.Now().Unix(), sessionID,
	); err != nil {
		return err
	}
	return tx.Commit()
}

// Load returns every message stored for a session, in the order they were
// appended.
func (s *Store) Load(ctx context.Context, sessionID string) ([]llm.Message, error) {
	rows, err := s.db.QueryContext(ctx,
		`SELECT role, content, tool_calls, tool_call_id, tool_name, is_error
		 FROM messages WHERE session_id = ? ORDER BY seq`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []llm.Message
	for rows.Next() {
		var (
			m         llm.Message
			role      string
			toolCalls sql.NullString
			isErr     int
		)
		if err := rows.Scan(&role, &m.Content, &toolCalls, &m.ToolCallID, &m.ToolName, &isErr); err != nil {
			return nil, err
		}
		m.Role = llm.Role(role)
		m.IsError = isErr != 0
		if toolCalls.Valid && toolCalls.String != "" {
			if err := json.Unmarshal([]byte(toolCalls.String), &m.ToolCalls); err != nil {
				return nil, fmt.Errorf("session: decoding tool_calls: %w", err)
			}
		}
		out = append(out, m)
	}
	return out, rows.Err()
}

// Replace overwrites a session's stored messages, used after compaction: the
// verbose history is swapped for the condensed one in a single transaction so
// a crash mid-write cannot leave the session half-truncated.
func (s *Store) Replace(ctx context.Context, sessionID string, msgs []llm.Message) error {
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer tx.Rollback()

	if _, err := tx.ExecContext(ctx, `DELETE FROM messages WHERE session_id = ?`, sessionID); err != nil {
		return err
	}

	stmt, err := tx.PrepareContext(ctx, `
		INSERT INTO messages (session_id, seq, role, content, tool_calls, tool_call_id, tool_name, is_error)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer stmt.Close()

	for i, m := range msgs {
		var toolCalls any
		if len(m.ToolCalls) > 0 {
			data, err := json.Marshal(m.ToolCalls)
			if err != nil {
				return err
			}
			toolCalls = string(data)
		}
		if _, err := stmt.ExecContext(ctx, sessionID, i, string(m.Role), m.Content,
			toolCalls, m.ToolCallID, m.ToolName, boolToInt(m.IsError)); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func scanMeta(row *sql.Row) (*Meta, error) {
	var (
		m                Meta
		created, updated int64
	)
	if err := row.Scan(&m.ID, &m.ProjectDir, &m.Title, &created, &updated); err != nil {
		if errors.Is(err, sql.ErrNoRows) {
			return nil, ErrNotFound
		}
		return nil, err
	}
	m.CreatedAt = time.Unix(created, 0)
	m.UpdatedAt = time.Unix(updated, 0)
	return &m, nil
}

func scanMetas(rows *sql.Rows) ([]Meta, error) {
	var out []Meta
	for rows.Next() {
		var (
			m                Meta
			created, updated int64
		)
		if err := rows.Scan(&m.ID, &m.ProjectDir, &m.Title, &created, &updated); err != nil {
			return nil, err
		}
		m.CreatedAt = time.Unix(created, 0)
		m.UpdatedAt = time.Unix(updated, 0)
		out = append(out, m)
	}
	return out, rows.Err()
}

func newID() (string, error) {
	b := make([]byte, 6)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}

func ensureParentDir(path string) error {
	return os.MkdirAll(filepath.Dir(path), 0o755)
}
