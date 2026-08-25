// Package skills loads reusable instructions the model pulls into context on
// demand, rather than paying for them in every system prompt.
//
// A skill is a Markdown file with a YAML-ish frontmatter header — the same
// shape the open Agent Skills convention and the Python prototype both used —
// stored outside any project repository, in Kiwi's own config directory, so
// a skill written while working on one project is never accidentally
// committed to it.
package skills

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/oscar1223/kiwi/internal/config"
)

// Skill is one loaded skill.
type Skill struct {
	Name        string
	Description string
	Body        string
}

// ErrNotFound is returned when a named skill does not exist.
var ErrNotFound = errors.New("skills: not found")

// Dir is where skill files live.
func Dir() (string, error) {
	dir, err := config.Dir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "skills"), nil
}

// Load reads every .md file in Dir(), keyed by skill name. A file without a
// valid frontmatter header is skipped rather than failing the whole load —
// one malformed skill should not take every other skill down with it. A
// missing directory is not an error: it just means no skills exist yet.
func Load() (map[string]Skill, error) {
	dir, err := Dir()
	if err != nil {
		return nil, err
	}
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return map[string]Skill{}, nil
	}
	if err != nil {
		return nil, err
	}

	out := map[string]Skill{}
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".md") {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		fallback := strings.TrimSuffix(e.Name(), ".md")
		if sk, ok := parse(string(data), fallback); ok {
			out[sk.Name] = sk
		}
	}
	return out, nil
}

// parse extracts a Skill from `---\nname: x\ndescription: y\n---\nbody`.
// fallbackName is used when the frontmatter omits name, matching the
// prototype's own behaviour of falling back to the filename.
func parse(raw, fallbackName string) (Skill, bool) {
	if !strings.HasPrefix(raw, "---") {
		return Skill{}, false
	}
	rest := raw[3:]
	end := strings.Index(rest, "---")
	if end < 0 {
		return Skill{}, false
	}

	sk := Skill{Name: fallbackName, Body: strings.TrimSpace(rest[end+3:])}
	for _, line := range strings.Split(rest[:end], "\n") {
		key, value, ok := strings.Cut(strings.TrimSpace(line), ":")
		if !ok {
			continue
		}
		key, value = strings.TrimSpace(key), strings.TrimSpace(value)
		switch key {
		case "name":
			if value != "" {
				sk.Name = value
			}
		case "description":
			sk.Description = value
		}
	}
	return sk, true
}

// Save creates or overwrites a skill file.
func Save(name, description, body string) (string, error) {
	dir, err := Dir()
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, name+".md")
	content := fmt.Sprintf("---\nname: %s\ndescription: %s\n---\n%s\n", name, description, strings.TrimSpace(body))
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// Delete removes a skill file.
func Delete(name string) error {
	dir, err := Dir()
	if err != nil {
		return err
	}
	path := filepath.Join(dir, name+".md")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("%w: %q", ErrNotFound, name)
	}
	return os.Remove(path)
}

// Summary renders the compact system-prompt blurb listing available skills —
// enough for the model to judge fit against the task at hand, without
// spending tokens on full bodies until load_skill actually pulls one in.
func Summary(sk map[string]Skill) string {
	if len(sk) == 0 {
		return ""
	}
	names := make([]string, 0, len(sk))
	for n := range sk {
		names = append(names, n)
	}
	sort.Strings(names)

	var b strings.Builder
	b.WriteString("Available skills (use the load_skill tool with a skill's name to load its full instructions, only when the task actually matches its description):\n")
	for _, n := range names {
		fmt.Fprintf(&b, "- %s: %s\n", n, sk[n].Description)
	}
	return b.String()
}
