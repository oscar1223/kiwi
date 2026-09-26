package main

import (
	"fmt"
	"sort"

	"github.com/oscar1223/kiwi/internal/mcp"
	"github.com/oscar1223/kiwi/internal/reach"
	"github.com/spf13/cobra"
)

func newDoctorCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "doctor",
		Short: "Show which ways of reaching the internet work here",
		Long: `Show which ways of reaching the internet work on this machine.

For each channel — web pages, search, YouTube, GitHub, RSS, MCP servers — it
says whether the agent can use it, what it uses, and how to enable or improve
it. It only looks; it installs nothing.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			channels := reach.Check(cmd.Context(), reach.SystemProbe(mcpServerNames))
			fmt.Fprint(cmd.OutOrStdout(), reach.Render(channels))
			return nil
		},
	}
}

// mcpServerNames lists the configured MCP servers, sorted.
func mcpServerNames() ([]string, error) {
	cfg, err := mcp.LoadConfig()
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(cfg))
	for n := range cfg {
		names = append(names, n)
	}
	sort.Strings(names)
	return names, nil
}
