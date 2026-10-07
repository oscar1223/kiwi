package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"path/filepath"
	"sync"
	"syscall"
	"time"
	// Embedded so KIWI_TZ works on minimal servers without zoneinfo.
	_ "time/tzdata"

	"github.com/oscar1223/kiwi/internal/config"
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
	// envTimezone is the zone /cron schedules are read in. A server usually
	// runs in UTC, which is not what "every day at 9" means to its owner.
	envTimezone     = "KIWI_TZ"
	defaultTimezone = "Europe/Madrid"
	// Photos, audios, videos and PDFs are turned into text by a multimodal
	// model on OpenRouter, so any main model can work with them.
	envMediaModel = "KIWI_MEDIA_MODEL"
	envMediaKey   = "OPENROUTER_API_KEY"
)

func newServeCmd(g *globalFlags) *cobra.Command {
	var (
		mode            string
		approvalTimeout time.Duration
	)

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
files and runs commands without asking. Anything the mode would still ask
about, such as a dangerous command, arrives as a message with Allow and Deny
buttons; with no answer before --approval-timeout it is refused. Use --mode
plan to keep it read-only.

It carries on the most recent conversation for the directory, so a restart
does not lose the thread. Send /new to start over.

/cron schedules tasks that run on their own and report to the chat, read in
the time zone in KIWI_TZ (default Europe/Madrid). Send /cron for the details.

Photos, voice notes, audios, videos and documents are saved under Kiwi's data
directory (inbox/). With OPENROUTER_API_KEY set, photos, audio, video and PDFs
are also described or transcribed by ` + remote.DefaultMediaModel + `
(change it with KIWI_MEDIA_MODEL), and the agent gets that text plus the path:
the main model never sees the file itself. Those files are sent to OpenRouter.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			m := permission.Mode(mode)
			if !m.Valid() {
				return fmt.Errorf("unknown mode %q (want ask, plan or work)", mode)
			}
			return runServe(cmd.Context(), g, m, approvalTimeout, os.Getenv, cmd.ErrOrStderr())
		},
	}
	cmd.Flags().StringVar(&mode, "mode", string(permission.ModeWork),
		"permission mode: ask, plan (read-only) or work")
	cmd.Flags().DurationVar(&approvalTimeout, "approval-timeout", remote.DefaultApprovalTimeout,
		"how long to wait for an Allow/Deny answer before refusing")
	return cmd
}

func runServe(ctx context.Context, g *globalFlags, mode permission.Mode, approvalTimeout time.Duration, getenv func(string) string, out io.Writer) error {
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
	tzName := getenv(envTimezone)
	if tzName == "" {
		tzName = defaultTimezone
	}
	loc, err := time.LoadLocation(tzName)
	if err != nil {
		return fmt.Errorf("%s: %w", envTimezone, err)
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

	approver := &remote.Approver{Timeout: approvalTimeout}
	rs, err := newServeSession(ctx, g, mode, approver, logf)
	if err != nil {
		return err
	}
	defer rs.Close()

	bot := remote.NewBot(client, allowed, rs.Handle)
	bot.Log = logf
	bot.OnCallback = approver.HandleCallback
	dataDir, err := config.DataDir()
	if err != nil {
		return err
	}
	bot.InboxDir = filepath.Join(dataDir, "inbox")
	if key := getenv(envMediaKey); key != "" {
		rs.Translator = &remote.OpenRouterTranslator{APIKey: key, Model: getenv(envMediaModel)}
	} else {
		logf(fmt.Sprintf("%s is not set: photos, audio and video are saved but not described", envMediaKey))
	}

	cron, err := newServeScheduler(rs, bot, allowed, loc, logf)
	if err != nil {
		return err
	}
	defer cron.Store.Close()
	rs.Cron = cron
	cronDone := make(chan struct{})
	go func() { cron.Start(ctx); close(cronDone) }()

	fmt.Fprintf(out, "kiwi serve: @%s is listening for %d allowed user(s), in %s mode, working in %s, scheduling in %s. Ctrl+C to stop.\n",
		me.Username, len(allowed), mode, rs.WorkDir, loc)

	err = bot.Run(ctx)
	stop() // a fatal bot error also stops the scheduler
	<-cronDone
	if err != nil {
		return err
	}
	fmt.Fprintln(out, "kiwi serve: stopped")
	return nil
}

// newServeScheduler runs /cron jobs as turns of the bot's session, reporting
// to the chat that created each one.
func newServeScheduler(rs *serveSession, bot *remote.Bot, allowed []int64, loc *time.Location, logf func(string)) (*remote.Scheduler, error) {
	dataDir, err := config.DataDir()
	if err != nil {
		return nil, err
	}
	store, err := remote.OpenCronStore(filepath.Join(dataDir, "cron.db"))
	if err != nil {
		return nil, err
	}

	isAllowed := map[int64]bool{}
	for _, id := range allowed {
		isAllowed[id] = true
	}

	sc := remote.NewScheduler(store, loc, func(ctx context.Context, j remote.Job) {
		// Jobs live in private chats, whose ID is the user's. Someone taken
		// off the allowed list stops getting their jobs run, and messages.
		if !isAllowed[j.ChatID] {
			logf(fmt.Sprintf("cron: skipping job #%d: chat %d is no longer allowed", j.ID, j.ChatID))
			return
		}
		conv := bot.Conversation(j.ChatID)
		if _, err := conv.Send(ctx, fmt.Sprintf("⏰ Tarea #%d: %s", j.ID, truncate(j.Prompt, 300))); err != nil {
			logf(fmt.Sprintf("cron: job #%d: %v", j.ID, err))
		}
		bot.Reply(ctx, j.ChatID, rs.RunScheduled(ctx, j.Prompt, conv))
	})
	sc.Log = logf
	return sc, nil
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
func newServeSession(ctx context.Context, g *globalFlags, mode permission.Mode, approver *remote.Approver, logf func(string)) (*serveSession, error) {
	flags := *g
	if flags.resumeID == "" {
		flags.continueLast = true
	}

	// Whatever the mode does not settle on its own is asked on Telegram, with
	// buttons. In work mode that is only the dangerous commands.
	rs, err := newSession(ctx, &flags, mode, approver)
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
		Agent:    rs.agent,
		WorkDir:  rs.workDir,
		Approver: approver,
		History:  rs.history,
		Log:      logf,
		Save: func(ctx context.Context, turn []llm.Message) ([]llm.Message, error) {
			return session.Persist(ctx, rs.store, rs.meta.ID, rs.agent.Provider, turn)
		},
		NewRun: func(ctx context.Context) (func(context.Context, []llm.Message) error, error) {
			meta, err := rs.store.Create(ctx, rs.workDir)
			if err != nil {
				return nil, err
			}
			return func(ctx context.Context, turn []llm.Message) error {
				_, err := session.Persist(ctx, rs.store, meta.ID, rs.agent.Provider, turn)
				return err
			}, nil
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
