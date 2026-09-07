package skills

import (
	"crypto/sha256"
	"embed"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// builtinFS carries the skills that ship with Kiwi.
//
// Embedding them is what makes "prefabricated" mean anything. Skills live in
// the user's config directory, deliberately outside any repository, so a fresh
// `go install` would otherwise leave that directory empty and the feature
// would exist only for whoever wrote a skill by hand.
//
//go:embed builtin/*.md
var builtinFS embed.FS

// seedRecord is where the hashes of previously seeded skills are kept, so an
// upgrade can tell a file the user edited from one that is simply the old
// version of a built-in.
const seedRecord = ".builtin-seeded"

// Seed writes the built-in skills into the skills directory, and returns the
// names it created or updated.
//
// The rule that matters: **a file the user has touched is never overwritten.**
// Upgrading Kiwi must not silently undo somebody's edit, and there is no way
// to ask about it at the moment this runs. A built-in is refreshed only when
// what is on disk is byte-for-byte the version Kiwi last wrote there, which is
// exactly the case where nothing can be lost.
func Seed() ([]string, error) {
	dir, err := Dir()
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}

	seeded := readRecord(filepath.Join(dir, seedRecord))
	entries, err := builtinFS.ReadDir("builtin")
	if err != nil {
		return nil, err
	}

	var written []string
	changed := false
	for _, e := range entries {
		data, err := builtinFS.ReadFile("builtin/" + e.Name())
		if err != nil {
			continue
		}
		path := filepath.Join(dir, e.Name())
		sum := hash(data)

		switch existing, err := os.ReadFile(path); {
		case os.IsNotExist(err):
			// Not there: either a first run, or the user deleted it. A
			// deletion Kiwi already knows about is respected — putting a
			// skill back that somebody removed on purpose would be the same
			// mistake as overwriting an edit.
			if _, known := seeded[e.Name()]; known {
				continue
			}
		case err != nil:
			continue
		case hash(existing) == sum:
			// Already current. Record it anyway, so a version installed
			// before this record existed is adopted rather than treated as
			// user-written forever.
			if seeded[e.Name()] != sum {
				seeded[e.Name()], changed = sum, true
			}
			continue
		case seeded[e.Name()] != hash(existing):
			// On disk is not what Kiwi last wrote: the user edited it. Theirs
			// wins, and it stops being a built-in.
			continue
		}

		if err := os.WriteFile(path, data, 0o644); err != nil {
			continue
		}
		seeded[e.Name()], changed = sum, true
		written = append(written, strings.TrimSuffix(e.Name(), ".md"))
	}

	if changed {
		writeRecord(filepath.Join(dir, seedRecord), seeded)
	}
	sort.Strings(written)
	return written, nil
}

// BuiltinNames lists the skills that ship with Kiwi, so a loaded skill can be
// shown as built-in rather than user-written.
func BuiltinNames() map[string]bool {
	out := map[string]bool{}
	entries, err := builtinFS.ReadDir("builtin")
	if err != nil {
		return out
	}
	for _, e := range entries {
		out[strings.TrimSuffix(e.Name(), ".md")] = true
	}
	return out
}

func hash(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// readRecord loads the "filename hash" table. A missing or corrupt record is
// an empty one: the worst it costs is that an edited skill is left alone,
// which is the safe direction to fail in.
func readRecord(path string) map[string]string {
	out := map[string]string{}
	data, err := os.ReadFile(path)
	if err != nil {
		return out
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if name, sum, ok := strings.Cut(line, " "); ok && name != "" && sum != "" {
			out[name] = sum
		}
	}
	return out
}

func writeRecord(path string, record map[string]string) {
	names := make([]string, 0, len(record))
	for n := range record {
		names = append(names, n)
	}
	sort.Strings(names)

	var b strings.Builder
	b.WriteString("# written by kiwi — which built-in skills were installed, and at what version\n")
	for _, n := range names {
		b.WriteString(n + " " + record[n] + "\n")
	}
	_ = os.WriteFile(path, []byte(b.String()), 0o644)
}
