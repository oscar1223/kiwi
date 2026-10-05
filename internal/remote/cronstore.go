package remote

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	_ "modernc.org/sqlite" // registers the "sqlite" database/sql driver
)

// ErrJobNotFound is returned for a job ID that does not exist in that chat.
var ErrJobNotFound = errors.New("cron: no such job")

// Job is a scheduled task: a prompt that runs on a cron schedule and reports
// to the chat that created it.
type Job struct {
	ID        int64
	ChatID    int64
	Spec      string
	Prompt    string
	Paused    bool
	CreatedAt time.Time
	// LastRun is when it last started, zero if never.
	LastRun time.Time
}

// CronStore keeps jobs in SQLite so they survive restarts. It is safe for
// concurrent use.
type CronStore struct {
	db *sql.DB
}

const cronSchema = `
CREATE TABLE IF NOT EXISTS cron_jobs (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	chat_id    INTEGER NOT NULL,
	spec       TEXT NOT NULL,
	prompt     TEXT NOT NULL,
	paused     INTEGER NOT NULL DEFAULT 0,
	created_at INTEGER NOT NULL,
	last_run   INTEGER NOT NULL DEFAULT 0
);
`

// OpenCronStore opens (or creates) the jobs database at path.
func OpenCronStore(path string) (*CronStore, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", "file:"+path+"?_pragma=busy_timeout(5000)")
	if err != nil {
		return nil, err
	}
	db.SetMaxOpenConns(1)
	if _, err := db.Exec(cronSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("cron: applying schema: %w", err)
	}
	return &CronStore{db: db}, nil
}

func (s *CronStore) Close() error { return s.db.Close() }

// Add saves a new job and returns it with its ID.
func (s *CronStore) Add(ctx context.Context, chatID int64, spec, prompt string, now time.Time) (Job, error) {
	res, err := s.db.ExecContext(ctx,
		`INSERT INTO cron_jobs (chat_id, spec, prompt, created_at) VALUES (?, ?, ?, ?)`,
		chatID, spec, prompt, now.Unix())
	if err != nil {
		return Job{}, err
	}
	id, err := res.LastInsertId()
	if err != nil {
		return Job{}, err
	}
	return Job{ID: id, ChatID: chatID, Spec: spec, Prompt: prompt, CreatedAt: now.Truncate(time.Second)}, nil
}

// List returns every job, or only one chat's when chatID is not zero.
func (s *CronStore) List(ctx context.Context, chatID int64) ([]Job, error) {
	q := `SELECT id, chat_id, spec, prompt, paused, created_at, last_run FROM cron_jobs`
	var args []any
	if chatID != 0 {
		q += ` WHERE chat_id = ?`
		args = append(args, chatID)
	}
	rows, err := s.db.QueryContext(ctx, q+` ORDER BY id`, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var jobs []Job
	for rows.Next() {
		var (
			j                Job
			paused           int
			created, lastRun int64
		)
		if err := rows.Scan(&j.ID, &j.ChatID, &j.Spec, &j.Prompt, &paused, &created, &lastRun); err != nil {
			return nil, err
		}
		j.Paused = paused != 0
		j.CreatedAt = time.Unix(created, 0)
		if lastRun != 0 {
			j.LastRun = time.Unix(lastRun, 0)
		}
		jobs = append(jobs, j)
	}
	return jobs, rows.Err()
}

// Get returns one of a chat's jobs.
func (s *CronStore) Get(ctx context.Context, chatID, id int64) (Job, error) {
	jobs, err := s.List(ctx, chatID)
	if err != nil {
		return Job{}, err
	}
	for _, j := range jobs {
		if j.ID == id {
			return j, nil
		}
	}
	return Job{}, ErrJobNotFound
}

// SetPaused pauses or resumes one of a chat's jobs.
func (s *CronStore) SetPaused(ctx context.Context, chatID, id int64, paused bool) error {
	return s.exec(ctx, `UPDATE cron_jobs SET paused = ? WHERE id = ? AND chat_id = ?`, boolInt(paused), id, chatID)
}

// Delete removes one of a chat's jobs.
func (s *CronStore) Delete(ctx context.Context, chatID, id int64) error {
	return s.exec(ctx, `DELETE FROM cron_jobs WHERE id = ? AND chat_id = ?`, id, chatID)
}

// MarkRun records that a job started at t.
func (s *CronStore) MarkRun(ctx context.Context, id int64, t time.Time) error {
	_, err := s.db.ExecContext(ctx, `UPDATE cron_jobs SET last_run = ? WHERE id = ?`, t.Unix(), id)
	return err
}

// exec runs a statement scoped to one chat's job, so a chat can never touch
// another chat's jobs, and reports ErrJobNotFound when nothing matched.
func (s *CronStore) exec(ctx context.Context, q string, args ...any) error {
	res, err := s.db.ExecContext(ctx, q, args...)
	if err != nil {
		return err
	}
	if n, err := res.RowsAffected(); err == nil && n == 0 {
		return ErrJobNotFound
	}
	return nil
}

func boolInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
