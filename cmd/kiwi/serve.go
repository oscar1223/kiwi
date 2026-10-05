package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/oscar1223/kiwi/internal/llm"
	"github.com/oscar1223/kiwi/internal/permission"
	"github.com/oscar1223/kiwi/internal/remote"
	"github.com/oscar1223/kiwi/internal/session"
	"github.com/spf13/cobra"
)

// The bot's settings come from the environment, or from Kiwi's .env file
// (loaded in main), so the token never lands in kiwi.json or a repository.
const (
	envTelegramToken   = "KIWI_TELEGRAM_TOKEN"
	envTelegramAllowed = "KIWI_TELEGRAM_ALLOWED_USERS"
)

func newServeCmd(g *globalFlags) *cobra.Command {
	var mode string

	cmd := &cobra.Command{
		Use:   "serve",
		Short: "Run Kiwi as a Telegram bot",
		Long: `Run Kiwi as a Telegram bot, so it can be reached from your phone.

It needs two settings, in the environment or in Kiwi's .env file:

  ` + envTelegramToken + `           the token @BotFather gave you
  ` + envTelegramAllowed + `   your Telegram user ID (comma-separated for several)

Only the allowed users get an answer, and only in a private chat: anyone else
is ignored without a reply. The bot polls Telegram, so no port is opened.

Each message is a task for the agent, working in the current directory (or
--cwd) and answering when it is done. Meanwhile one message is kept up to date
with the tools it runs. It runs in work mode by default: it edits
files and runs commands without asking, and anything the mode would ask about,
such as a dangerous command, is refused. Use --mode plan to keep it read-only.

It carries on the most recent conversation for the directory, so a restart
does not lose the thread. Send /new to start over.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			m := permission.Mode(mode)
			if !m.Valid() {
				return fmt.Errorf("unknown mode %q (want ask, plan or work)", mode)
			}
			return runServe(cmd.Context(), g, m, os.Getenv, cmd.ErrOrStderr())
		},
	}
	cmd.Flags().StringVar(&mode, "mode", string(permission.ModeWork),
		"permission mode: ask, plan (read-only) or work")
	return cmd
}

func runServe(ctx context.Context, g *globalFlags, mode permission.Mode, getenv func(string) string, out io.Writer) error {
	token := getenv(envTelegramToken)
	if token == "" {
		return fmt.Errorf("%s is not set; create a bot with @BotFather and put its token there", envTelegramToken)
	}
	allowed, err := remote.ParseUserIDs(getenv(envTelegramAllowed))
	if err != nil {
		return fmt.Errorf("%s: %w", envTelegramAllowed, err)
	}
	if len(allowed) == 0 {
		// A bot nobody may use is pointless, and one everybody may use is a
		// remote shell. Refusing to start is the only safe default.
		return fmt.Errorf("%s is not set; refusing to run a bot anyone can talk to (get your ID from @userinfobot)", envTelegramAllowed)
	}

	ctx, stop := signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)
	defer stop()

	// The bot and every running turn write here from their own goroutines.
	var logMu sync.Mutex
	logf := func(s string) {
		logMu.Lock()
		defer logMu.Unlock()
		fmt.Fprintf(out, "%s  %s\n", time.Now().Format("15:04:05"), s)
	}

	client := remote.NewClient(token)

	checkCtx, cancel := context.WithTimeout(ctx, 15*time.Second)
	me, err := client.GetMe(checkCtx)
	cancel()
	if err != nil {
		if ctx.Err() != nil {
			return nil
		}
		var apiErr *remote.APIError
		if errors.As(err, &apiErr) && apiErr.Code == 401 {
			return fmt.Errorf("telegram rejected %s; check the token", envTelegramToken)
		}
		return err
	}

	rs, err := newServeSession(ctx, g, mode, logf)
	if err != nil {
		return err
	}
	defer rs.Close()

	fmt.Fprintf(out, "kiwi serve: @%s is listening for %d allowed user(s), in %s mode, working in %s. Ctrl+C to stop.\n",
		me.Username, len(allowed), mode, rs.WorkDir)

	bot := remote.NewBot(client, allowed, rs.Handle)
	bot.Log = logf
	if err := bot.Run(ctx); err != nil {
		return err
	}
	fmt.Fprintln(out, "kiwi serve: stopped")
	return nil
}

// serveSession is the remote.Session the bot drives plus the runSession that
// owns its resources.
type serveSession struct {
	*remote.Session
	run *runSession
}

func (s *serveSession) Close() error { return s.run.Close() }

// newServeSession builds the agent the bot talks to. It continues the most
// recent conversation for the directory unless --resume names another, so a
// restart picks up where it left off.
func newServeSession(ctx context.Context, g *globalFlags, mode permission.Mode, logf func(string)) (*serveSession, error) {
	flags := *g
	if flags.resumeID == "" {
		flags.continueLast = true
	}

	// Nobody can answer a question from here yet (buttons are #11), so
	// whatever the mode does not settle on its own is refused. In work mode
	// that is only the dangerous commands.
	rs, err := newSession(ctx, &flags, mode, permission.NonInteractive{})
	if err != nil {
		return nil, err
	}
	if rs.needsOnboarding {
		rs.Close()
		return nil, errors.New("no model provider is configured yet — run `kiwi` once to set one up")
	}

	rs.broker.OnAutoDecision(func(req *permission.Request, allowed bool) {
		if !allowed {
			logf(fmt.Sprintf("blocked (%s mode): %s", req.Mode.Label(), truncate(req.Detail, 100)))
		}
	})

	s := &remote.Session{
		Agent:   rs.agent,
		WorkDir: rs.workDir,
		History: rs.history,
		Log:     logf,
		Save: func(ctx context.Context, turn []llm.Message) ([]llm.Message, error) {
			return session.Persist(ctx, rs.store, rs.meta.ID, rs.agent.Provider, turn)
		},
		Reset: func(ctx context.Context) error {
			meta, err := rs.store.Create(ctx, rs.workDir)
			if err != nil {
				return err
			}
			rs.meta = meta
			return nil
		},
	}
	if len(rs.history) > 0 {
		logf(fmt.Sprintf("continuing session %s (%d messages)", rs.meta.ID, len(rs.history)))
	}
	return &serveSession{Session: s, run: rs}, nil
}
