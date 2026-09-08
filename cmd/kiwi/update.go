package main

import (
	"errors"
	"fmt"
	"io"

	"github.com/oscar1223/kiwi/internal/update"
	"github.com/spf13/cobra"
)

func newUpdateCmd() *cobra.Command {
	var checkOnly bool

	cmd := &cobra.Command{
		Use:   "update",
		Short: "Update kiwi to the latest release",
		Long: `Update kiwi to the latest release.

Downloads the build for this platform, verifies it against the checksums
published with the release, and replaces the running binary.

Where a package manager installed kiwi, it owns the binary and this command
prints its upgrade command instead of overwriting the file behind its back.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runUpdate(cmd, checkOnly)
		},
	}

	cmd.Flags().BoolVar(&checkOnly, "check", false,
		"only report whether a newer version exists")
	return cmd
}

func runUpdate(cmd *cobra.Command, checkOnly bool) error {
	out := cmd.OutOrStdout()

	if version == update.DevVersion {
		fmt.Fprintln(out, "This is a development build; there is nothing to compare it against.")
		return nil
	}

	fmt.Fprintln(out, "Checking for updates…")
	latest, err := update.Latest(cmd.Context())
	if err != nil {
		if errors.Is(err, update.ErrNoReleases) {
			fmt.Fprintln(out, "No release has been published yet; there is nothing to update to.")
			return nil
		}
		return err
	}
	if !update.IsNewer(version, latest) {
		fmt.Fprintf(out, "kiwi %s is already the latest release.\n", version)
		return nil
	}

	fmt.Fprintf(out, "\nA newer version is available: %s (you have %s)\n", latest, version)

	method := update.Detect()
	if !method.SelfManaged() {
		// Overwriting the binary here would leave the package manager's
		// records pointing at a version that is no longer on disk.
		fmt.Fprintf(out, "\nkiwi was installed through %s. Update it with:\n\n  %s\n\n",
			method.Label(), method.Command())
		return nil
	}

	if checkOnly {
		fmt.Fprintf(out, "\nRun %q to install it.\n", "kiwi update")
		return nil
	}

	fmt.Fprintln(out)
	log := func(step string) { fmt.Fprintf(out, "  %s\n", step) }
	if err := update.Apply(cmd.Context(), latest, log); err != nil {
		return err
	}

	fmt.Fprintf(out, "\nUpdated to %s. Restart kiwi to use it.\n", latest)
	return nil
}

// printUpdateNotice writes the one-line "there is a newer version" hint used by
// non-interactive commands. The TUI shows its own in the status line.
func printUpdateNotice(w io.Writer, latest string) {
	fmt.Fprintf(w, "kiwi %s is available (you have %s) — run `kiwi update`\n", latest, version)
}
