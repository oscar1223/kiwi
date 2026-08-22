package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/session"
	"github.com/spf13/cobra"
)

func newAskCmd(g *globalFlags) *cobra.Command {
	var (
		quiet bool
		mode  string
		yolo  bool
	)

	cmd := &cobra.Command{
		Use:   "ask [prompt]",
		Short: "Run a single non-interactive turn and print the answer",
		Long: `Run one turn without the TUI and print the result to stdout.

The answer goes to stdout and progress to stderr, so redirecting stdout
captures only the answer.

The prompt comes from the arguments, from stdin, or from both. Stdin is read
when there are no arguments, or when one of them is "-":

  kiwi ask "what does this repo do?"
  git diff | kiwi ask
  git diff | kiwi ask "review this patch" -

Runs are read-only by default. Use --mode work to let it edit files, and
--yolo to also let it run commands unsupervised.`,
		Args: cobra.ArbitraryArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			input, err := readPrompt(cmd.InOrStdin(), args)
			if err != nil {
				return err
			}
			if strings.TrimSpace(input) == "" {
				return errors.New("no prompt given; pass it as an argument or pipe it on stdin")
			}
			m := permission.Mode(mode)
			if !m.Valid() {
				return fmt.Errorf("unknown mode %q (want ask, plan or work)", mode)
			}
			return runAsk(cmd.Context(), g, askOptions{
				input: input,
				quiet: quiet,
				mode:  m,
				yolo:  yolo,
			}, cmd.OutOrStdout(), cmd.ErrOrStderr())
		},
	}

	cmd.Flags().BoolVarP(&quiet, "quiet", "q", false, "suppress tool-call progress on stderr")
	cmd.Flags().StringVar(&mode, "mode", string(permission.ModePlan),
		"permission mode: ask, plan (read-only) or work")
	cmd.Flags().BoolVar(&yolo, "yolo", false,
		"approve every action without asking (dangerous: this run is unsupervised)")
	return cmd
}

// readPrompt resolves the prompt from arguments and/or stdin.
//
// Stdin is only consumed when the user asked for it: either no arguments were
// given, or one of them is "-". Sniffing whether stdin "looks piped" is not
// good enough — any non-interactive parent (CI, cron, Docker, a script) hands
// us an inherited pipe that never closes, and kiwi would block forever with no
// output. Explicit beats clever here.
//
//	kiwi ask "what does this do?"      → args only, never touches stdin
//	git diff | kiwi ask                → stdin only
//	git diff | kiwi ask "review this" - → both, joined
func readPrompt(stdin io.Reader, args []string) (string, error) {
	var parts []string
	wantStdin := len(args) == 0

	for _, a := range args {
		if a == "-" {
			wantStdin = true
			continue
		}
		parts = append(parts, a)
	}
	argPart := strings.Join(parts, " ")

	if !wantStdin {
		return argPart, nil
	}

	data, err := io.ReadAll(stdin)
	if err != nil {
		return "", fmt.Errorf("reading stdin: %w", err)
	}
	piped := strings.TrimRight(string(data), "\n")

	switch {
	case piped != "" && argPart != "":
		return argPart + "\n\n" + piped, nil
	case piped != "":
		return piped, nil
	default:
		return argPart, nil
	}
}

type askOptions struct {
	input string
	quiet bool
	mode  permission.Mode
	yolo  bool
}

func runAsk(ctx context.Context, g *globalFlags, opts askOptions, stdout, stderr io.Writer) error {
	// Ctrl-C cancels the turn: the context reaches the model stream and every
	// child process a tool started.
	ctx, stop := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Nobody is watching a headless run, so anything the mode policy does not
	// settle is refused — unless the user opted out with --yolo.
	var decider permission.Decider = permission.NonInteractive{}
	if opts.yolo {
		decider = permission.AllowAll{}
	}

	sess, err := newSession(ctx, g, opts.mode, decider)
	if err != nil {
		return err
	}
	defer sess.Close()

	if !opts.quiet {
		sess.broker.OnAutoDecision(func(req *permission.Request, allowed bool) {
			verb := "blocked"
			if allowed {
				verb = "auto-approved"
			}
			fmt.Fprintf(stderr, "· %s (%s mode): %s\n", verb, req.Mode.Label(), truncate(req.Detail, 100))
		})
	}

	obs := &cliObserver{out: stdout, progress: stderr, quiet: opts.quiet}
	res, err := sess.agent.Run(ctx, opts.input, sess.history, obs)
	if err != nil {
		if errors.Is(err, context.Canceled) {
			fmt.Fprintln(stderr, "\ncancelled")
			return nil
		}
		return err
	}

	// The answer streamed to stdout already; just terminate the line.
	if res.Text != "" && !strings.HasSuffix(res.Text, "\n") {
		fmt.Fprintln(stdout)
	}

	// A single headless run has no interactivity to protect, so persisting
	// synchronously before exiting is simplest — there is no next prompt it
	// could be blocking.
	if _, err := session.Persist(ctx, sess.store, sess.meta.ID, sess.agent.Provider, res.Messages); err != nil && !opts.quiet {
		fmt.Fprintln(stderr, "kiwi: warning: could not save this session:", err)
	}
	return nil
}

// cliObserver streams the answer to stdout and progress to stderr, so that
// redirecting stdout captures only the answer.
type cliObserver struct {
	out      io.Writer
	progress io.Writer
	quiet    bool
}

func (o *cliObserver) OnText(delta string) { fmt.Fprint(o.out, delta) }

func (o *cliObserver) OnToolCall(call llm.ToolCall) {
	if o.quiet {
		return
	}
	fmt.Fprintf(o.progress, "· %s %s\n", call.Name, truncate(string(call.Input), 120))
}

func (o *cliObserver) OnToolResult(_ llm.ToolCall, output string, isErr bool) {
	if o.quiet {
		return
	}
	marker := "  ↳"
	if isErr {
		marker = "  ✗"
	}
	fmt.Fprintf(o.progress, "%s %s\n", marker, truncate(output, 120))
}

func (o *cliObserver) OnUsage(llm.Usage) {}

func truncate(s string, n int) string {
	s = strings.ReplaceAll(strings.TrimSpace(s), "\n", " ")
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
