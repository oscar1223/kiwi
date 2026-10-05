package remote

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// Schedule is a parsed cron expression: five fields, minute hour
// day-of-month month day-of-week, or one of the @ macros.
//
// It follows classic cron, including its one surprise: when both day-of-month
// and day-of-week are restricted, a day matches if either does ("the 1st, and
// also every Monday").
type Schedule struct {
	minute, hour, dom, month, dow uint64 // bit i set: value i matches
	domStar, dowStar              bool
}

var cronMacros = map[string]string{
	"@yearly":   "0 0 1 1 *",
	"@annually": "0 0 1 1 *",
	"@monthly":  "0 0 1 * *",
	"@weekly":   "0 0 * * 0",
	"@daily":    "0 0 * * *",
	"@midnight": "0 0 * * *",
	"@hourly":   "0 * * * *",
}

type cronField struct {
	name     string
	min, max int
}

var cronFields = [5]cronField{
	{"minute", 0, 59},
	{"hour", 0, 23},
	{"day of month", 1, 31},
	{"month", 1, 12},
	{"day of week", 0, 7}, // 0 and 7 are both Sunday
}

// ParseSchedule parses a cron expression. Errors say which field is wrong
// and why, since they go straight back to the user.
func ParseSchedule(expr string) (Schedule, error) {
	expr = strings.TrimSpace(expr)
	if m, ok := cronMacros[strings.ToLower(expr)]; ok {
		expr = m
	}
	parts := strings.Fields(expr)
	if len(parts) != 5 {
		return Schedule{}, fmt.Errorf("se esperan 5 campos (minuto hora día mes día-de-la-semana) o @daily/@hourly…, y hay %d", len(parts))
	}

	var bits [5]uint64
	for i, p := range parts {
		b, err := parseCronField(p, cronFields[i])
		if err != nil {
			return Schedule{}, err
		}
		bits[i] = b
	}
	// Sunday is 0 and also 7; fold 7 onto 0.
	if bits[4]&(1<<7) != 0 {
		bits[4] = bits[4]&^(1<<7) | 1
	}
	return Schedule{
		minute: bits[0], hour: bits[1], dom: bits[2], month: bits[3], dow: bits[4],
		domStar: parts[2] == "*", dowStar: parts[4] == "*",
	}, nil
}

func parseCronField(s string, f cronField) (uint64, error) {
	var bits uint64
	for _, item := range strings.Split(s, ",") {
		rng, stepStr, hasStep := strings.Cut(item, "/")
		step := 1
		if hasStep {
			n, err := strconv.Atoi(stepStr)
			if err != nil || n <= 0 {
				return 0, fmt.Errorf("%s: el paso %q no es un número positivo", f.name, stepStr)
			}
			step = n
		}

		lo, hi := f.min, f.max
		switch {
		case rng == "*":
		case strings.Contains(rng, "-"):
			a, b, _ := strings.Cut(rng, "-")
			var err error
			if lo, err = cronNumber(a, f); err != nil {
				return 0, err
			}
			if hi, err = cronNumber(b, f); err != nil {
				return 0, err
			}
			if lo > hi {
				return 0, fmt.Errorf("%s: el rango %q va hacia atrás", f.name, rng)
			}
		default:
			n, err := cronNumber(rng, f)
			if err != nil {
				return 0, err
			}
			lo = n
			if !hasStep {
				hi = n
			}
		}
		for v := lo; v <= hi; v += step {
			bits |= 1 << v
		}
	}
	return bits, nil
}

func cronNumber(s string, f cronField) (int, error) {
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0, fmt.Errorf("%s: %q no es un número", f.name, s)
	}
	if n < f.min || n > f.max {
		return 0, fmt.Errorf("%s: %d está fuera de %d-%d", f.name, n, f.min, f.max)
	}
	return n, nil
}

// Next returns the first time strictly after t that matches, in t's
// location, or the zero time if there is none within five years (an
// expression like "0 0 31 2 *" never fires).
func (s Schedule) Next(t time.Time) time.Time {
	loc := t.Location()
	t = t.Truncate(time.Minute).Add(time.Minute)
	limit := t.AddDate(5, 0, 0)

	for t.Before(limit) {
		if s.month&(1<<uint(t.Month())) == 0 {
			t = time.Date(t.Year(), t.Month()+1, 1, 0, 0, 0, 0, loc)
			continue
		}
		if !s.dayMatches(t) {
			t = time.Date(t.Year(), t.Month(), t.Day()+1, 0, 0, 0, 0, loc)
			continue
		}
		if s.hour&(1<<uint(t.Hour())) == 0 {
			t = time.Date(t.Year(), t.Month(), t.Day(), t.Hour()+1, 0, 0, 0, loc)
			continue
		}
		if s.minute&(1<<uint(t.Minute())) == 0 {
			t = t.Add(time.Minute)
			continue
		}
		return t
	}
	return time.Time{}
}

func (s Schedule) dayMatches(t time.Time) bool {
	dom := s.dom&(1<<uint(t.Day())) != 0
	dow := s.dow&(1<<uint(t.Weekday())) != 0
	switch {
	case s.domStar && s.dowStar:
		return true
	case s.domStar:
		return dow
	case s.dowStar:
		return dom
	default:
		return dom || dow
	}
}
