package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/oscar1223/kiwi/internal/remote"
	"github.com/spf13/cobra"
)

// The bot's settings come from the environment, or from Kiwi's .env file
// (loaded in main), so the token never lands in kiwi.json or a repository.
const (
	envTelegramToken   = "KIWI_TELEGRAM_TOKEN"
	envTelegramAllowed = "KIWI_TELEGRAM_ALLOWED_USERS"
)

func newServeCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "serve",
		Short: "Run Kiwi as a Telegram bot",
		Long: `Run Kiwi as a Telegram bot, so it can be reached from your phone.

It needs two settings, in the environment or in Kiwi's .env file:

  ` + envTelegramToken + `           the token @BotFather gave you
  ` + envTelegramAllowed + `   your Telegram user ID (comma-separated for several)

Only the allowed users get an answer, and only in a private chat: anyone else
is ignored without a reply. The bot polls Telegram, so no port is opened.

For now the bot only echoes what it receives.`,
		Args: cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			return runServe(cmd.Context(), os.Getenv, cmd.ErrOrStderr())
		},
	}
}

func runServe(ctx context.Context, getenv func(string) string, log io.Writer) error {
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

	fmt.Fprintf(log, "kiwi serve: @%s is listening for %d allowed user(s). Ctrl+C to stop.\n", me.Username, len(allowed))

	bot := remote.NewBot(client, allowed, remote.Echo)
	bot.Log = func(s string) {
		fmt.Fprintf(log, "%s  %s\n", time.Now().Format("15:04:05"), s)
	}
	if err := bot.Run(ctx); err != nil {
		return err
	}
	fmt.Fprintln(log, "kiwi serve: stopped")
	return nil
}
