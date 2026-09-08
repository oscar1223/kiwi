package update

import (
	"io"
	"runtime"
	"strings"
	"testing"
)

func newReader(s string) io.Reader { return strings.NewReader(s) }

func TestNewer(t *testing.T) {
	tests := []struct {
		name            string
		current, latest string
		want            bool
	}{
		{"patch bump", "v0.1.0", "v0.1.1", true},
		{"minor bump", "v0.1.9", "v0.2.0", true},
		{"major bump", "v0.9.9", "v1.0.0", true},
		{"same", "v0.1.0", "v0.1.0", false},
		{"older", "v0.2.0", "v0.1.0", false},
		{"no v prefix", "0.1.0", "0.1.1", true},
		{"mixed prefix", "v0.1.0", "0.1.1", true},
		// 10 > 9 numerically but "10" < "9" as a string: the whole reason
		// parse() exists rather than comparing the tags directly.
		{"two digit patch", "v0.1.9", "v0.1.10", true},
		{"two digit minor", "v0.9.0", "v0.10.0", true},
		// A prerelease on the current build still compares by its numbers.
		{"prerelease current", "v0.1.0-rc1", "v0.1.1", true},
		{"build metadata", "v0.1.0+abc", "v0.1.0", false},
		// Anything unparseable means silence, not a guess.
		{"dev current", "dev", "v0.1.0", false},
		{"garbage latest", "v0.1.0", "banana", false},
		{"too few parts", "v0.1", "v0.2", false},
		{"empty", "", "", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := newer(tt.current, tt.latest); got != tt.want {
				t.Errorf("newer(%q, %q) = %v, want %v", tt.current, tt.latest, got, tt.want)
			}
		})
	}
}

func TestMethodForPath(t *testing.T) {
	tests := []struct {
		path string
		want Method
	}{
		{"/opt/homebrew/Cellar/kiwi/0.1.0/bin/kiwi", Homebrew},
		{"/usr/local/Caskroom/kiwi/0.1.0/kiwi", Homebrew},
		{"/home/linuxbrew/.linuxbrew/Cellar/kiwi/0.1.0/bin/kiwi", Homebrew},
		{`C:\Users\me\scoop\apps\kiwi\current\kiwi.exe`, Scoop},
		{"/home/me/go/bin/kiwi", GoInstall},
		{"/home/me/.local/bin/kiwi", Standalone},
		{"/usr/local/bin/kiwi", Standalone},
		{`C:\Users\me\AppData\Local\Programs\kiwi\kiwi.exe`, Standalone},
	}

	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			// The Windows cases use backslashes, which only normalise on
			// Windows; skip them elsewhere rather than assert a wrong answer.
			if runtime.GOOS != "windows" && containsBackslash(tt.path) {
				t.Skip("windows path")
			}
			if got := methodForPath(tt.path); got != tt.want {
				t.Errorf("methodForPath(%q) = %v, want %v", tt.path, got, tt.want)
			}
		})
	}
}

func containsBackslash(s string) bool {
	for _, r := range s {
		if r == '\\' {
			return true
		}
	}
	return false
}

func TestSelfManaged(t *testing.T) {
	if !Standalone.SelfManaged() {
		t.Error("a standalone install must be updatable in place")
	}
	for _, m := range []Method{Homebrew, Scoop, GoInstall} {
		if m.SelfManaged() {
			t.Errorf("%v is package-managed and must not be replaced in place", m)
		}
		if m.Command() == "kiwi update" {
			t.Errorf("%v must point at its own package manager, not back at kiwi update", m)
		}
	}
}

func TestChecksumFor(t *testing.T) {
	sums := []byte(
		"aaaa1111bbbb2222cccc3333dddd4444eeee5555ffff6666aaaa7777bbbb8888  kiwi_darwin_arm64.tar.gz\n" +
			"1111aaaa2222bbbb3333cccc4444dddd5555eeee6666ffff7777aaaa8888bbbb  kiwi_linux_amd64.tar.gz\n" +
			"9999aaaa2222bbbb3333cccc4444dddd5555eeee6666ffff7777aaaa8888cccc  kiwi_windows_amd64.zip\n")

	got, ok := checksumFor(sums, "kiwi_linux_amd64.tar.gz")
	if !ok {
		t.Fatal("expected to find kiwi_linux_amd64.tar.gz")
	}
	want := "1111aaaa2222bbbb3333cccc4444dddd5555eeee6666ffff7777aaaa8888bbbb"
	if got != want {
		t.Errorf("checksum = %q, want %q", got, want)
	}

	// A name that is only a substring of a listed one must not match: a
	// partial hit here would install a verified-looking wrong file.
	if _, ok := checksumFor(sums, "kiwi_linux_amd"); ok {
		t.Error("a partial filename must not match")
	}
	if _, ok := checksumFor(sums, "kiwi_freebsd_amd64.tar.gz"); ok {
		t.Error("an absent filename must not match")
	}
}

func TestArchiveNameMatchesGoreleaser(t *testing.T) {
	// The install scripts and .goreleaser.yaml agree on this shape; if it
	// changes in one place it has to change in all of them.
	got := ArchiveName()
	want := "kiwi_" + runtime.GOOS + "_" + runtime.GOARCH
	if runtime.GOOS == "windows" {
		want += ".zip"
	} else {
		want += ".tar.gz"
	}
	if got != want {
		t.Errorf("ArchiveName() = %q, want %q", got, want)
	}
}
