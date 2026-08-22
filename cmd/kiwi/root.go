package main

import (
	"fmt"

	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/spf13/cobra"
)

// version is overridden at build time via -ldflags.
var version = "dev"

// globalFlags are shared by every subcommand.
type globalFlags struct {
	model string
	cwd   string

	continueLast bool
	resumeID     string
}

func newRootCmd() *cobra.Command {
	var (
		g    globalFlags
		mode string
	)

	cmd := &cobra.Command{
		Use:   "kiwi",
		Short: "A local-first coding agent for your terminal",
		Long: `Kiwi is a local-first coding agent for your terminal.

Run ` + "`kiwi`" + ` with no arguments to start the interactive interface, or
` + "`kiwi ask`" + ` for a single non-interactive turn that composes with pipes.`,
		Version:       version,
		SilenceUsage:  true,
		SilenceErrors: true,
		Args:          cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			m := permission.Mode(mode)
			if !m.Valid() {
				return fmt.Errorf("unknown mode %q (want ask, plan or work)", mode)
			}
			return runTUI(cmd.Context(), &g, m)
		},
	}

	cmd.PersistentFlags().StringVarP(&g.model, "model", "m", "", "model profile to use (default: the configured one)")
	cmd.PersistentFlags().StringVar(&g.cwd, "cwd", "", "working directory (default: the current one)")
	cmd.PersistentFlags().BoolVarP(&g.continueLast, "continue", "c", false,
		"continue the most recent session for this project")
	cmd.PersistentFlags().StringVar(&g.resumeID, "resume", "",
		"resume a specific session by id (see kiwi session list)")
	cmd.Flags().StringVar(&mode, "mode", string(permission.ModeAsk),
		"starting permission mode: ask, plan or work")

	cmd.AddCommand(newAskCmd(&g))
	cmd.AddCommand(newSessionCmd(&g))
	return cmd
}
