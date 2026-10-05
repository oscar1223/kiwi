package remote

import (
	"strings"
	"testing"
	"time"
)

func mustLoc(t *testing.T, name string) *time.Location {
	t.Helper()
	loc, err := time.LoadLocation(name)
	if err != nil {
		t.Fatal(err)
	}
	return loc
}

func TestScheduleNext(t *testing.T) {
	madrid := mustLoc(t, "Europe/Madrid")
	at := func(s string) time.Time {
		v, err := time.ParseInLocation("2006-01-02 15:04", s, madrid)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}

	tests := []struct {
		expr, from, want string
	}{
		{"* * * * *", "2026-10-06 09:00", "2026-10-06 09:01"},
		{"0 9 * * *", "2026-10-06 08:59", "2026-10-06 09:00"},
		{"0 9 * * *", "2026-10-06 09:00", "2026-10-07 09:00"}, // strictly after
		{"*/15 * * * *", "2026-10-06 09:07", "2026-10-06 09:15"},
		{"30 8-10/2 * * *", "2026-10-06 09:00", "2026-10-06 10:30"},
		// Weekdays only: Tuesday 6 Oct 2026 at 10:00 → next is Wednesday.
		{"0 9 * * 1-5", "2026-10-06 10:00", "2026-10-07 09:00"},
		// Friday evening → Monday.
		{"0 9 * * 1-5", "2026-10-09 18:00", "2026-10-12 09:00"},
		// Sunday as 7 and as 0.
		{"0 12 * * 7", "2026-10-06 00:00", "2026-10-11 12:00"},
		{"0 12 * * 0", "2026-10-06 00:00", "2026-10-11 12:00"},
		// Month rollover, and a month without a 31st.
		{"0 0 31 * *", "2026-11-01 00:00", "2026-12-31 00:00"},
		// Both day fields restricted: either matches (the 15th, or Mondays).
		{"0 0 15 * 1", "2026-10-06 00:00", "2026-10-12 00:00"},
		{"0 0 15 * 1", "2026-10-13 00:00", "2026-10-15 00:00"},
		// Leap day.
		{"0 0 29 2 *", "2026-10-06 00:00", "2028-02-29 00:00"},
		{"@daily", "2026-10-06 13:00", "2026-10-07 00:00"},
		{"@hourly", "2026-10-06 13:20", "2026-10-06 14:00"},
		{"@weekly", "2026-10-06 13:20", "2026-10-11 00:00"},
		{"0 10 1,15 * *", "2026-10-02 00:00", "2026-10-15 10:00"},
	}
	for _, tt := range tests {
		s, err := ParseSchedule(tt.expr)
		if err != nil {
			t.Errorf("ParseSchedule(%q): %v", tt.expr, err)
			continue
		}
		if got := s.Next(at(tt.from)); !got.Equal(at(tt.want)) {
			t.Errorf("%q from %s = %s, want %s", tt.expr, tt.from, got.Format("2006-01-02 15:04 Mon"), tt.want)
		}
	}
}

func TestScheduleDST(t *testing.T) {
	madrid := mustLoc(t, "Europe/Madrid")
	s, _ := ParseSchedule("0 9 * * *")

	// 9:00 local stays 9:00 local across the October change (UTC+2 → UTC+1).
	before := time.Date(2026, 10, 24, 10, 0, 0, 0, madrid)
	got := s.Next(before)
	if got.Hour() != 9 || got.Day() != 25 {
		t.Errorf("next after the clocks go back = %s, want 25 Oct 09:00 local", got)
	}
	if _, off := got.Zone(); off != 3600 {
		t.Errorf("offset %d, want +1h (winter time)", off)
	}

	// 2:30 does not exist on 29 March 2026 in Madrid; it must not loop.
	s, _ = ParseSchedule("30 2 * * *")
	got = s.Next(time.Date(2026, 3, 28, 12, 0, 0, 0, madrid))
	if got.IsZero() || got.Before(time.Date(2026, 3, 29, 0, 0, 0, 0, madrid)) {
		t.Errorf("next 2:30 across spring-forward = %s", got)
	}
}

func TestScheduleNeverFires(t *testing.T) {
	s, err := ParseSchedule("0 0 31 2 *")
	if err != nil {
		t.Fatal(err)
	}
	if got := s.Next(time.Now()); !got.IsZero() {
		t.Errorf("31 February fired at %s", got)
	}
}

func TestParseScheduleErrors(t *testing.T) {
	tests := []struct{ expr, want string }{
		{"", "5 campos"},
		{"* * * *", "5 campos"},
		{"60 * * * *", "minute: 60 está fuera de 0-59"},
		{"* 24 * * *", "hour: 24"},
		{"* * 0 * *", "day of month: 0"},
		{"* * * 13 *", "month: 13"},
		{"* * * * 8", "day of week: 8"},
		{"*/0 * * * *", "paso"},
		{"10-5 * * * *", "hacia atrás"},
		{"a * * * *", "no es un número"},
		{"@sometimes", "5 campos"},
	}
	for _, tt := range tests {
		_, err := ParseSchedule(tt.expr)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("ParseSchedule(%q) = %v, want an error mentioning %q", tt.expr, err, tt.want)
		}
	}
}
