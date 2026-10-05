package remote

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

// JobRunner runs one scheduled job to completion, reporting to its chat.
type JobRunner func(ctx context.Context, job Job)

// clock is time, behind an interface so tests can move it by hand.
type clock interface {
	Now() time.Time
	After(d time.Duration) <-chan time.Time
}

type realClock struct{}

func (realClock) Now() time.Time                         { return time.Now() }
func (realClock) After(d time.Duration) <-chan time.Time { return time.After(d) }

// Scheduler runs Jobs on their cron schedules and answers the /cron command.
//
// Runs that were missed while the process was down are not made up: on start,
// and after a job is resumed, the next run is computed from now. A job that is
// still running when it comes due again is skipped that time rather than
// queued, so a slow job cannot pile up behind itself.
type Scheduler struct {
	Store *CronStore
	Loc   *time.Location
	Run   JobRunner
	// Log reports to the operator's terminal. May be nil.
	Log func(string)

	clock clock
	wake  chan struct{}

	mu      sync.Mutex
	running map[int64]bool
	wg      sync.WaitGroup
}

// NewScheduler returns a scheduler using the real clock.
func NewScheduler(store *CronStore, loc *time.Location, run JobRunner) *Scheduler {
	return newScheduler(store, loc, run, realClock{})
}

func newScheduler(store *CronStore, loc *time.Location, run JobRunner, c clock) *Scheduler {
	if loc == nil {
		loc = time.Local
	}
	return &Scheduler{
		Store:   store,
		Loc:     loc,
		Run:     run,
		clock:   c,
		wake:    make(chan struct{}, 1),
		running: map[int64]bool{},
	}
}

func (sc *Scheduler) logf(format string, args ...any) {
	if sc.Log != nil {
		sc.Log(fmt.Sprintf(format, args...))
	}
}

func (sc *Scheduler) now() time.Time { return sc.clock.Now().In(sc.Loc) }

// poke makes the loop re-read the jobs, after one is added or changed.
func (sc *Scheduler) poke() {
	select {
	case sc.wake <- struct{}{}:
	default:
	}
}

// Start runs the scheduling loop until ctx is done, then waits for the jobs
// in flight (which see the same cancellation) to stop.
func (sc *Scheduler) Start(ctx context.Context) {
	defer sc.wg.Wait()
	next := map[int64]time.Time{}

	for {
		now := sc.now()
		jobs, err := sc.Store.List(ctx, 0)
		if err != nil {
			if ctx.Err() != nil {
				return
			}
			sc.logf("cron: reading jobs: %v (retrying in a minute)", err)
			jobs = nil
		}

		seen := map[int64]bool{}
		var earliest time.Time
		for _, j := range jobs {
			if j.Paused {
				continue
			}
			sched, err := ParseSchedule(j.Spec)
			if err != nil {
				continue // validated when added; never expected
			}
			seen[j.ID] = true

			due, known := next[j.ID]
			switch {
			case !known:
				due = sched.Next(now)
			case !due.After(now):
				sc.fire(ctx, j)
				due = sched.Next(now)
			}
			if due.IsZero() {
				delete(next, j.ID)
				continue
			}
			next[j.ID] = due
			if earliest.IsZero() || due.Before(earliest) {
				earliest = due
			}
		}
		// Deleted and paused jobs drop out, so resuming one computes its
		// next run from then, without catching up.
		for id := range next {
			if !seen[id] {
				delete(next, id)
			}
		}

		wait := time.Minute
		if err == nil && !earliest.IsZero() {
			wait = earliest.Sub(now)
		}
		select {
		case <-ctx.Done():
			return
		case <-sc.wake:
		case <-sc.clock.After(wait):
		}
	}
}

// fire starts a job unless it is still running from last time.
func (sc *Scheduler) fire(ctx context.Context, j Job) {
	sc.mu.Lock()
	if sc.running[j.ID] {
		sc.mu.Unlock()
		sc.logf("cron: job #%d is still running; skipping this run", j.ID)
		return
	}
	sc.running[j.ID] = true
	sc.mu.Unlock()

	sc.wg.Add(1)
	go func() {
		defer sc.wg.Done()
		defer func() {
			sc.mu.Lock()
			delete(sc.running, j.ID)
			sc.mu.Unlock()
		}()
		if err := sc.Store.MarkRun(ctx, j.ID, sc.now()); err != nil && ctx.Err() == nil {
			sc.logf("cron: job #%d: recording the run: %v", j.ID, err)
		}
		sc.logf("cron: running job #%d", j.ID)
		sc.Run(ctx, j)
	}()
}

const cronUsage = `Tareas programadas:

/cron add "0 9 * * 1-5" revisa los PRs abiertos
/cron add @daily resume los commits de ayer
/cron list
/cron pause 3 · /cron resume 3 · /cron rm 3
/cron run 3   (ejecutarla ya)

Formato: minuto hora día mes día-de-la-semana (0 o 7 = domingo), o @hourly, @daily, @weekly, @monthly.`

// Command answers "/cron <args>" from chatID.
func (sc *Scheduler) Command(ctx context.Context, chatID int64, args string) string {
	sub, rest, _ := strings.Cut(strings.TrimSpace(args), " ")
	rest = strings.TrimSpace(rest)

	switch strings.ToLower(sub) {
	case "", "help":
		return cronUsage
	case "add":
		return sc.add(ctx, chatID, rest)
	case "list", "ls":
		return sc.list(ctx, chatID)
	case "pause", "resume", "rm", "run":
		id, err := strconv.ParseInt(strings.TrimPrefix(rest, "#"), 10, 64)
		if err != nil {
			return fmt.Sprintf("Falta el número de la tarea: /cron %s 3", sub)
		}
		return sc.act(ctx, chatID, strings.ToLower(sub), id)
	default:
		return "No conozco «" + sub + "».\n\n" + cronUsage
	}
}

func (sc *Scheduler) add(ctx context.Context, chatID int64, rest string) string {
	spec, prompt, err := splitSpec(rest)
	if err != nil {
		return err.Error() + "\n\n" + cronUsage
	}
	sched, err := ParseSchedule(spec)
	if err != nil {
		return "Expresión no válida: " + err.Error()
	}
	first := sched.Next(sc.now())
	if first.IsZero() {
		return "«" + spec + "» no se cumpliría nunca (¿31 de febrero?)."
	}

	j, err := sc.Store.Add(ctx, chatID, spec, prompt, sc.now())
	if err != nil {
		return "No he podido guardarla: " + err.Error()
	}
	sc.poke()
	return fmt.Sprintf("Tarea #%d creada.\n%s · próxima: %s\n%s", j.ID, spec, formatWhen(first), prompt)
}

// splitSpec separates the schedule from the prompt. The schedule is quoted,
// an @macro, or the first five fields.
func splitSpec(s string) (spec, prompt string, err error) {
	switch {
	case strings.HasPrefix(s, `"`):
		end := strings.Index(s[1:], `"`)
		if end < 0 {
			return "", "", errors.New("Falta cerrar las comillas de la expresión.")
		}
		spec, prompt = s[1:end+1], s[end+2:]
	case strings.HasPrefix(s, "@"):
		spec, prompt, _ = strings.Cut(s, " ")
	default:
		f := strings.Fields(s)
		if len(f) < 6 {
			return "", "", errors.New("Faltan la expresión o la tarea.")
		}
		spec = strings.Join(f[:5], " ")
		// The prompt keeps its own spacing: skip past the fifth field.
		rest := s
		for range 5 {
			rest = strings.TrimLeft(rest, " \t")
			i := strings.IndexAny(rest, " \t")
			rest = rest[i:]
		}
		prompt = rest
	}
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", "", errors.New("Falta la tarea después de la expresión.")
	}
	return strings.TrimSpace(spec), prompt, nil
}

func (sc *Scheduler) list(ctx context.Context, chatID int64) string {
	jobs, err := sc.Store.List(ctx, chatID)
	if err != nil {
		return "No he podido leer las tareas: " + err.Error()
	}
	if len(jobs) == 0 {
		return "No hay tareas programadas.\n\n" + cronUsage
	}
	now := sc.now()
	var b strings.Builder
	for i, j := range jobs {
		if i > 0 {
			b.WriteString("\n\n")
		}
		state := "próxima: "
		if j.Paused {
			state = "⏸ en pausa"
		} else if sched, err := ParseSchedule(j.Spec); err == nil {
			state += formatWhen(sched.Next(now))
		}
		fmt.Fprintf(&b, "#%d · %s · %s\n%s", j.ID, j.Spec, state, truncateRunes(j.Prompt, 120))
	}
	return b.String()
}

func (sc *Scheduler) act(ctx context.Context, chatID int64, verb string, id int64) string {
	var err error
	switch verb {
	case "pause":
		err = sc.Store.SetPaused(ctx, chatID, id, true)
	case "resume":
		err = sc.Store.SetPaused(ctx, chatID, id, false)
	case "rm":
		err = sc.Store.Delete(ctx, chatID, id)
	case "run":
		var j Job
		if j, err = sc.Store.Get(ctx, chatID, id); err == nil {
			// Detached from this message's context: the run outlives the
			// command that started it, and stops with the scheduler's.
			sc.fire(context.WithoutCancel(ctx), j)
			return fmt.Sprintf("Ejecutando la tarea #%d.", id)
		}
	}
	if errors.Is(err, ErrJobNotFound) {
		return fmt.Sprintf("No hay ninguna tarea #%d.", id)
	}
	if err != nil {
		return "No he podido: " + err.Error()
	}
	sc.poke()
	done := map[string]string{
		"pause":  "Tarea #%d en pausa.",
		"resume": "Tarea #%d reanudada.",
		"rm":     "Tarea #%d borrada.",
	}
	return fmt.Sprintf(done[verb], id)
}

var weekdaysES = [...]string{"dom", "lun", "mar", "mié", "jue", "vie", "sáb"}
var monthsES = [...]string{"", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"}

// formatWhen is a short Spanish date: "lun 12 oct 09:00".
func formatWhen(t time.Time) string {
	if t.IsZero() {
		return "nunca"
	}
	return fmt.Sprintf("%s %d %s %s", weekdaysES[t.Weekday()], t.Day(), monthsES[t.Month()], t.Format("15:04"))
}
