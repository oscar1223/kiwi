package main

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/permission"
)

func TestServeRefusesIncompleteConfig(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		wantErr string
	}{
		{"no token", map[string]string{envTelegramAllowed: "42"}, envTelegramToken},
		{"no allowed users", map[string]string{envTelegramToken: "1:x"}, "refusing"},
		{"bad user id", map[string]string{envTelegramToken: "1:x", envTelegramAllowed: "@me"}, "not a Telegram user ID"},
		{"bad time zone", map[string]string{envTelegramToken: "1:x", envTelegramAllowed: "42", envTimezone: "Mars/Olympus"}, envTimezone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			getenv := func(k string) string { return tt.env[k] }
			err := runServe(context.Background(), &globalFlags{}, permission.ModeWork, time.Minute, getenv, io.Discard)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("runServe() = %v, want an error mentioning %q", err, tt.wantErr)
			}
		})
	}
}
