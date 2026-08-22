package main

import (
	"github.com/spf13/cobra"
)

// version is overridden at build time via -ldflags.
var version = "dev"

// globalFlags are shared by every subcommand.
type globalFlags struct {
	model string
	cwd   string
}

func newRootCmd() *cobra.Command {
	var g globalFlags

	cmd := &cobra.Command{
		Use:           "kiwi",
		Short:         "A local-first coding agent for your terminal",
		Version:       version,
		SilenceUsage:  true,
		SilenceErrors: true,
	}

	cmd.PersistentFlags().StringVarP(&g.model, "model", "m", "", "model profile to use (default: the configured one)")
	cmd.PersistentFlags().StringVar(&g.cwd, "cwd", "", "working directory (default: the current one)")

	cmd.AddCommand(newAskCmd(&g))
	return cmd
}
