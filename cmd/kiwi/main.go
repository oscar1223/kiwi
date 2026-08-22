// Command kiwi is a local-first coding agent for the terminal.
package main

import (
	"fmt"
	"os"

	"github.com/oscar1223/kiwi/internal/config"
)

func main() {
	// Loaded before anything else so every command — the TUI, `ask`, future
	// subcommands — sees the same environment regardless of how it builds its
	// provider.
	if err := config.LoadDotEnv(); err != nil {
		fmt.Fprintln(os.Stderr, "kiwi: loading .env:", err)
		os.Exit(1)
	}

	if err := newRootCmd().Execute(); err != nil {
		fmt.Fprintln(os.Stderr, "kiwi:", err)
		os.Exit(1)
	}
}
