package remote

import (
	"context"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/llm/llmtest"
)

// fakeClock only moves when Advance says so.
type fakeClock struct {
	mu      sync.Mutex
	now     time.Time
	waiters []fakeWaiter
}

type fakeWaiter struct {
	at time.Time
	ch chan time.Time
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

func (c *fakeClock) After(d time.Duration) <-chan time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	ch := make(chan time.Time, 1)
	if d <= 0 {
		ch <- c.now
		return ch
	}
	c.waiters = append(c.waiters, fakeWaiter{c.now.Add(d), ch})
	return ch
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
	kept := c.waiters[:0]
	for _, w := range c.waiters {
		if !w.at.After(c.now) {
			w.ch <- c.now
		} else {
			kept = append(kept, w)
		}
	}
	c.waiters = kept
}

func (c *fakeClock) pending() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.waiters)
}

type schedFixture struct {
	sc    *Scheduler
	clock *fakeClock
	store *CronStore
	runs  chan Job
	loc   *time.Location
}

func newSchedFixture(t *testing.T, start string, run JobRunner) *schedFixture {
	t.Helper()
	loc := mustLoc(t, "Europe/Madrid")
	now, err := time.ParseInLocation("2006-01-02 15:04", start, loc)
	if err != nil {
		t.Fatal(err)
	}
	store, err := OpenCronStore(filepath.Join(t.TempDir(), "cron.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })

	f := &schedFixture{clock: &fakeClock{now: now}, store: store, runs: make(chan Job, 10), loc: loc}
	if run == nil {
		run = func(_ context.Context, j Job) { f.runs <- j }
	}
	f.sc = newScheduler(store, loc, run, f.clock)
	return f
}

func (f *schedFixture) start(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { f.sc.Start(ctx); close(done) }()
	t.Cleanup(func() { cancel(); <-done })
}

// settle waits until the loop is parked on the clock again.
func (f *schedFixture) settle(t *testing.T) {
	t.Helper()
	waitFor(t, "the scheduler to wait on the clock", func() bool { return f.clock.pending() > 0 })
}

// advance moves the clock and waits for the loop to park again.
func (f *schedFixture) advance(t *testing.T, d time.Duration) {
	t.Helper()
	f.settle(t)
	f.clock.Advance(d)
	f.settle(t)
}

func (f *schedFixture) expectRun(t *testing.T, id int64) {
	t.Helper()
	select {
	case j := <-f.runs:
		if j.ID != id {
			t.Fatalf("ran job #%d, want #%d", j.ID, id)
		}
	case <-time.After(2 * time.Second):
		t.Fatalf("job #%d did not run", id)
	}
}

func (f *schedFixture) expectNoRun(t *testing.T) {
	t.Helper()
	select {
	case j := <-f.runs:
		t.Fatalf("job #%d ran when nothing was due", j.ID)
	case <-time.After(50 * time.Millisecond):
	}
}

const chat = int64(42)

func TestRunsOnSchedule(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	reply := f.sc.Command(context.Background(), chat, `add "0 9 * * *" resume el día`)
	if !strings.Contains(reply, "Tarea #1 creada") || !strings.Contains(reply, "mar 6 oct 09:00") {
		t.Fatalf("add replied:\n%s", reply)
	}
	f.start(t)

	f.advance(t, 59*time.Minute)
	f.expectNoRun(t)
	f.advance(t, time.Minute)
	f.expectRun(t, 1)

	// And again the next day, not before.
	f.advance(t, 23*time.Hour+59*time.Minute)
	f.expectNoRun(t)
	f.advance(t, time.Minute)
	f.expectRun(t, 1)

	jobs, _ := f.store.List(context.Background(), chat)
	if want := time.Date(2026, 10, 7, 9, 0, 0, 0, f.loc); !jobs[0].LastRun.Equal(want) {
		t.Errorf("LastRun = %s, want %s", jobs[0].LastRun, want)
	}
}

func TestMissedRunsAreNotCaughtUp(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 12:00", nil)
	// Created, then the process was "down" for days.
	j, _ := f.store.Add(context.Background(), chat, "0 9 * * *", "x", time.Date(2026, 10, 1, 8, 0, 0, 0, f.loc))
	f.store.MarkRun(context.Background(), j.ID, time.Date(2026, 10, 2, 9, 0, 0, 0, f.loc))

	f.start(t)
	f.settle(t)
	f.expectNoRun(t) // no burst for 3, 4, 5 and 6 October

	f.advance(t, 21*time.Hour) // 7 Oct 09:00
	f.expectRun(t, j.ID)
}

func TestPauseAndResume(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	f.sc.Command(context.Background(), chat, "add @hourly comprueba la CI")
	f.start(t)

	if r := f.sc.Command(context.Background(), chat, "pause 1"); r != "Tarea #1 en pausa." {
		t.Errorf("pause: %q", r)
	}
	f.advance(t, 3*time.Hour)
	f.expectNoRun(t)

	f.sc.Command(context.Background(), chat, "resume 1")
	f.settle(t)
	f.expectNoRun(t) // resuming does not make up the paused hours
	f.advance(t, time.Hour)
	f.expectRun(t, 1)
}

func TestRemovedJobStops(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:30", nil)
	f.sc.Command(context.Background(), chat, "add @hourly x")
	f.start(t)
	if r := f.sc.Command(context.Background(), chat, "rm 1"); r != "Tarea #1 borrada." {
		t.Errorf("rm: %q", r)
	}
	f.advance(t, 2*time.Hour)
	f.expectNoRun(t)
}

func TestSlowJobIsNotStacked(t *testing.T) {
	release := make(chan struct{})
	var mu sync.Mutex
	started := 0
	f := newSchedFixture(t, "2026-10-06 08:00", func(ctx context.Context, j Job) {
		mu.Lock()
		started++
		mu.Unlock()
		select {
		case <-release:
		case <-ctx.Done():
		}
	})
	f.sc.Command(context.Background(), chat, "add * * * * * algo lento")
	f.start(t)

	for range 5 {
		f.advance(t, time.Minute)
	}
	close(release)

	mu.Lock()
	defer mu.Unlock()
	if started != 1 {
		t.Errorf("started %d times while the first run was still going, want 1", started)
	}
}

func TestRunNow(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	f.sc.Command(context.Background(), chat, "add @weekly informe")
	if r := f.sc.Command(context.Background(), chat, "run 1"); r != "Ejecutando la tarea #1." {
		t.Errorf("run: %q", r)
	}
	f.expectRun(t, 1)
}

func TestCommandsAreScopedToTheChat(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	f.sc.Command(context.Background(), chat, "add @daily mío")

	other := int64(7)
	for _, cmd := range []string{"pause 1", "resume 1", "rm 1", "run 1"} {
		if r := f.sc.Command(context.Background(), other, cmd); r != "No hay ninguna tarea #1." {
			t.Errorf("%s from another chat: %q", cmd, r)
		}
	}
	if r := f.sc.Command(context.Background(), other, "list"); !strings.Contains(r, "No hay tareas") {
		t.Errorf("another chat's list shows this chat's jobs:\n%s", r)
	}
	f.expectNoRun(t)
}

func TestAddParsing(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	tests := []struct {
		args, spec, prompt string
	}{
		{`add "0 9 * * 1-5" revisa los PRs`, "0 9 * * 1-5", "revisa los PRs"},
		{`add 0 9 * * 1-5 revisa   los PRs`, "0 9 * * 1-5", "revisa   los PRs"},
		{`add @daily resume ayer`, "@daily", "resume ayer"},
		{"add */30 8-20 * * * mira el correo\ny avísame", "*/30 8-20 * * *", "mira el correo\ny avísame"},
	}
	for _, tt := range tests {
		if r := f.sc.Command(context.Background(), chat, tt.args); !strings.Contains(r, "creada") {
			t.Errorf("%q: %s", tt.args, r)
		}
	}
	jobs, _ := f.store.List(context.Background(), chat)
	for i, tt := range tests {
		if jobs[i].Spec != tt.spec || jobs[i].Prompt != tt.prompt {
			t.Errorf("%q stored as %q / %q", tt.args, jobs[i].Spec, jobs[i].Prompt)
		}
	}
}

func TestAddRejects(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	tests := []struct{ args, want string }{
		{`add "61 * * * *" x`, "minute: 61"},
		{`add "0 0 31 2 *" x`, "no se cumpliría nunca"},
		{`add "0 9 * * *"`, "Falta la tarea"},
		{`add "0 9 * * * sin cerrar`, "comillas"},
		{`add 0 9 * *`, "Faltan"},
		{`frobnicate`, "No conozco"},
		{`pause`, "Falta el número"},
	}
	for _, tt := range tests {
		if r := f.sc.Command(context.Background(), chat, tt.args); !strings.Contains(r, tt.want) {
			t.Errorf("%q replied %q, want it to mention %q", tt.args, r, tt.want)
		}
	}
	if jobs, _ := f.store.List(context.Background(), chat); len(jobs) != 0 {
		t.Errorf("rejected commands stored %d jobs", len(jobs))
	}
}

func TestList(t *testing.T) {
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	f.sc.Command(context.Background(), chat, `add "0 9 * * 1-5" revisa los PRs`)
	f.sc.Command(context.Background(), chat, `add @daily resume`)
	f.sc.Command(context.Background(), chat, "pause 2")

	r := f.sc.Command(context.Background(), chat, "list")
	for _, want := range []string{"#1 · 0 9 * * 1-5 · próxima: mar 6 oct 09:00", "revisa los PRs", "#2 · @daily · ⏸ en pausa"} {
		if !strings.Contains(r, want) {
			t.Errorf("list lacks %q:\n%s", want, r)
		}
	}
}

func TestJobsSurviveRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cron.db")
	s1, _ := OpenCronStore(path)
	s1.Add(context.Background(), chat, "@daily", "persiste", time.Now())
	s1.Close()

	s2, err := OpenCronStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()
	jobs, _ := s2.List(context.Background(), 0)
	if len(jobs) != 1 || jobs[0].Prompt != "persiste" {
		t.Errorf("after reopening: %+v", jobs)
	}
}

func TestScheduledRunWaitsAndKeepsTheConversation(t *testing.T) {
	started := make(chan struct{})
	s, fake, _ := newTestSession(t,
		llmtest.Step{Chunks: []string{"turno", " largo"}, Delay: 50 * time.Millisecond, Hook: func() { close(started) }},
		llmtest.Step{Text: "informe listo"},
	)
	var savedRuns [][]llm.Message
	s.NewRun = func(context.Context) (func(context.Context, []llm.Message) error, error) {
		return func(_ context.Context, turn []llm.Message) error {
			savedRuns = append(savedRuns, turn)
			return nil
		}, nil
	}

	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); s.Handle(context.Background(), msg("hola"), nil) }()
	<-started

	// Arrives mid-turn: waits instead of answering "busy".
	reply := s.RunScheduled(context.Background(), "haz el informe", nil)
	wg.Wait()

	if reply != "informe listo" {
		t.Errorf("scheduled reply = %q", reply)
	}
	if n := len(fake.Requests[1].Messages); n != 1 {
		t.Errorf("the scheduled run carried %d messages, want only its own prompt", n)
	}
	if len(savedRuns) != 1 {
		t.Errorf("saved %d scheduled runs, want 1", len(savedRuns))
	}
	// The chat's conversation is the first turn only.
	if len(s.History) != 2 || !strings.Contains(s.History[0].Content, "hola") {
		t.Errorf("the scheduled run leaked into the conversation: %+v", s.History)
	}
}

func TestCronCommandWorksMidTurn(t *testing.T) {
	started := make(chan struct{})
	s, _, _ := newTestSession(t, llmtest.Step{Chunks: []string{"a", "b"}, Delay: 50 * time.Millisecond, Hook: func() { close(started) }})
	f := newSchedFixture(t, "2026-10-06 08:00", nil)
	s.Cron = f.sc

	var wg sync.WaitGroup
	wg.Add(1)
	go func() { defer wg.Done(); s.Handle(context.Background(), msg("tarea"), nil) }()
	<-started
	if r := s.Handle(context.Background(), msg("/cron list"), nil); r == Busy {
		t.Error("/cron answered busy; it only touches the job list")
	}
	wg.Wait()
}
