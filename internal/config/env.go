package config

import (
	"os"
	"path/filepath"
	"strings"
)

// EnvFile is the .env file the /config command manages.
//
// It preserves the file's line order — comments and blank lines included —
// so editing one variable never reshuffles or strips the rest of it. Parsing
// mirrors LoadDotEnv's own KEY=VALUE rules, so what /config shows always
// matches what actually gets loaded at startup.
type EnvFile struct {
	path  string
	lines []string
}

// OpenEnvFile reads the .env file in Dir(), or starts an empty one if it does
// not exist yet — the first Set call then creates it.
func OpenEnvFile() (*EnvFile, error) {
	dir, err := Dir()
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, ".env")

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return &EnvFile{path: path}, nil
	}
	if err != nil {
		return nil, err
	}

	lines := strings.Split(string(data), "\n")
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1] // drop the split artifact from the trailing newline
	}
	return &EnvFile{path: path, lines: lines}, nil
}

// Keys returns the variable names defined in the file, in file order.
func (f *EnvFile) Keys() []string {
	var keys []string
	for _, line := range f.lines {
		if k, _, ok := parseEnvLine(line); ok {
			keys = append(keys, k)
		}
	}
	return keys
}

// Get returns a variable's value and whether it is set.
func (f *EnvFile) Get(key string) (string, bool) {
	for _, line := range f.lines {
		if k, v, ok := parseEnvLine(line); ok && k == key {
			return v, true
		}
	}
	return "", false
}

// Set adds or updates a variable and saves immediately.
func (f *EnvFile) Set(key, value string) error {
	line := key + "=" + value
	for i, l := range f.lines {
		if k, _, ok := parseEnvLine(l); ok && k == key {
			f.lines[i] = line
			return f.save()
		}
	}
	f.lines = append(f.lines, line)
	return f.save()
}

// Unset removes a variable and saves immediately. Removing a variable that
// was never set is not an error.
func (f *EnvFile) Unset(key string) error {
	out := f.lines[:0]
	for _, l := range f.lines {
		if k, _, ok := parseEnvLine(l); ok && k == key {
			continue
		}
		out = append(out, l)
	}
	f.lines = out
	return f.save()
}

func (f *EnvFile) save() error {
	if err := os.MkdirAll(filepath.Dir(f.path), 0o755); err != nil {
		return err
	}
	content := strings.Join(f.lines, "\n")
	if content != "" {
		content += "\n"
	}
	// 0600: this file holds API keys.
	return os.WriteFile(f.path, []byte(content), 0o600)
}

func parseEnvLine(line string) (key, value string, ok bool) {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" || strings.HasPrefix(trimmed, "#") {
		return "", "", false
	}
	trimmed = strings.TrimPrefix(trimmed, "export ")

	k, v, found := strings.Cut(trimmed, "=")
	if !found {
		return "", "", false
	}
	k = strings.TrimSpace(k)
	if k == "" {
		return "", "", false
	}
	v = strings.Trim(strings.TrimSpace(v), `"'`)
	return k, v, true
}
