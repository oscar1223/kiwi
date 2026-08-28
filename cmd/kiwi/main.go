// Command kiwi is a local-first coding agent for the terminal.
package main

import (
	"context"
	"fmt"
	"os"
	"os/user"
	"path/filepath"

	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/telemetry"
)

func main() {
	// run's deferred cleanups (including the telemetry flush) must complete
	// before the process exits, so os.Exit is called out here rather than
	// from inside run — os.Exit skips deferred functions.
	os.Exit(run())
}

func run() int {
	// Loaded before anything else so every command — the TUI, `ask`, future
	// subcommands — sees the same environment regardless of how it builds its
	// provider.
	if err := config.LoadDotEnv(); err != nil {
		fmt.Fprintln(os.Stderr, "kiwi: loading .env:", err)
		return 1
	}

	ctx := context.Background()

	// Export failures (backend unreachable, bad credentials) go to a log
	// file, never to the terminal: with the TUI running, a stray write to
	// stderr from the exporter's background retry goroutine corrupts the
	// display rather than merely looking untidy. openTelemetryLog never fails
	// main() over this — a nil writer just means those lines are dropped.
	telemetryLog := openTelemetryLog()
	if telemetryLog != nil {
		defer telemetryLog.Close()
	}

	// A telemetry backend is entirely optional (see internal/telemetry), so
	// a failure to reach one is a warning, never a reason to refuse to run.
	shutdown, err := telemetry.Configure(ctx, telemetryLog, telemetryVersionAttr())
	if err != nil {
		fmt.Fprintln(os.Stderr, "kiwi: warning: telemetry disabled:", err)
	}
	defer shutdown(ctx)

	// Carried on ctx (not read fresh per command) so every trace this process
	// produces is attributed to the same OS user, and so ExecuteContext is
	// what makes it reach cmd.Context() inside every subcommand's RunE.
	ctx = telemetry.WithUserID(ctx, currentOSUser())

	if err := newRootCmd().ExecuteContext(ctx); err != nil {
		fmt.Fprintln(os.Stderr, "kiwi:", err)
		return 1
	}
	return 0
}

// openTelemetryLog opens the file OTel export failures are written to,
// instead of the terminal. Any failure along the way (no data directory, no
// permission) just means telemetry errors are dropped instead of logged —
// never a reason to refuse to start kiwi.
func openTelemetryLog() *os.File {
	dataDir, err := config.DataDir()
	if err != nil {
		return nil
	}
	if err := os.MkdirAll(dataDir, 0o755); err != nil {
		return nil
	}
	f, err := os.OpenFile(filepath.Join(dataDir, "telemetry.log"), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil
	}
	return f
}

// currentOSUser identifies who is running kiwi for telemetry attribution.
// Kiwi is a single-user local tool with no login of its own, so the OS
// account is the only stable identity available — good enough to tell one
// person's traces apart from another's on a shared self-hosted backend.
func currentOSUser() string {
	if u, err := user.Current(); err == nil && u.Username != "" {
		return u.Username
	}
	return "unknown"
}
