package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"time"

	sessionstore "github.com/oscar1223/kiwi/internal/session"
	"github.com/spf13/cobra"
)

func newSessionCmd(g *globalFlags) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "session",
		Short: "Inspect saved conversations",
	}
	cmd.AddCommand(newSessionListCmd(g))
	return cmd
}

func newSessionListCmd(g *globalFlags) *cobra.Command {
	return &cobra.Command{
		Use:   "list",
		Short: "List sessions saved for the current project",
		Long: `List sessions saved for the current project, most recently used first.

The id shown here (or an unambiguous prefix of it) is what --resume expects:

  kiwi --resume a1b2c3
`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			cwd := g.cwd
			if cwd == "" {
				var err error
				if cwd, err = os.Getwd(); err != nil {
					return err
				}
			}

			store, err := openSessionStore()
			if err != nil {
				return err
			}
			defer store.Close()

			return listSessions(cmd.Context(), store, cwd, cmd.OutOrStdout())
		},
	}
}

// listSessions is the testable core of `kiwi session list`: given an already
// open store, it just formats and prints.
func listSessions(ctx context.Context, store *sessionstore.Store, projectDir string, out io.Writer) error {
	sessions, err := store.List(ctx, projectDir, 50)
	if err != nil {
		return err
	}
	if len(sessions) == 0 {
		fmt.Fprintln(out, "no sessions saved for this project yet")
		return nil
	}

	for _, s := range sessions {
		title := s.Title
		if title == "" {
			title = "(untitled)"
		}
		fmt.Fprintf(out, "%s  %-10s  %s\n", s.ID, relativeTime(s.UpdatedAt), title)
	}
	return nil
}

// relativeTime renders a timestamp the way a human reads a session list:
// coarse buckets, not a precise duration nobody needs.
func relativeTime(t time.Time) string {
	d := time.Since(t)
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		return fmt.Sprintf("%dm ago", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh ago", int(d.Hours()))
	default:
		return fmt.Sprintf("%dd ago", int(d.Hours()/24))
	}
}
