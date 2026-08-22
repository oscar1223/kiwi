package config

import (
	"bufio"
	"os"
	"strings"
)

// LoadDotEnv loads KEY=VALUE pairs from the .env file in Dir() (by default
// ~/.config/kiwi/.env) into the process environment.
//
// The key never lives inside a project's own repository — the file sits in
// Kiwi's config directory, the same place opencode and OpenClaw keep
// credentials, so it cannot be committed by accident. An already-set
// environment variable is never overridden: the shell's environment always
// wins over the file, matching the precedence documented by other agents
// (OpenClaw: process environment, then .env).
//
// A missing file is not an error: plenty of setups only ever export the key
// from the shell.
func LoadDotEnv() error {
	dir, err := Dir()
	if err != nil {
		return err
	}
	path := dir + "/.env"

	f, err := os.Open(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")

		key, value, ok := strings.Cut(line, "=")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		value = strings.Trim(value, `"'`)

		if key == "" {
			continue
		}
		if _, alreadySet := os.LookupEnv(key); alreadySet {
			continue
		}
		os.Setenv(key, value)
	}
	return scanner.Err()
}
