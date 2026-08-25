package tui

import (
	"context"
	"testing"
	"time"

	"github.com/oscar1223/kiwi/internal/config"
	"github.com/oscar1223/kiwi/internal/permission"
)

// ApplyTheme mutates package-level vars, so every test that calls it must
// restore the default afterward — otherwise it leaks into unrelated tests
// that run later in the same binary.
func withRestoredTheme(t *testing.T) {
	t.Helper()
	t.Cleanup(func() { applyTheme(themeKiwiDark) })
}

func TestThemeByNameDefaultsOnEmptyOrUnknown(t *testing.T) {
	got, ok := themeByName("")
	if !ok || got.Name != DefaultThemeName {
		t.Errorf("themeByName(\"\") = (%+v, %v), want the default theme", got, ok)
	}
	if _, ok := themeByName("no-such-theme"); ok {
		t.Error("themeByName should report false for an unknown name")
	}
}

func TestApplyThemeChangesPackageColorsAndStyles(t *testing.T) {
	withRestoredTheme(t)

	applyTheme(themeKiwiLight)
	if colKiwi != themeKiwiLight.Kiwi {
		t.Errorf("colKiwi = %v, want the light theme's Kiwi colour", colKiwi)
	}
	if colDim != themeKiwiLight.Dim {
		t.Errorf("colDim = %v, want the light theme's Dim colour", colDim)
	}
	// Styles are rebuilt from the colour vars, not left pointing at the old
	// palette — this is the bug applyTheme exists to avoid.
	if styleKiwi.GetForeground() != themeKiwiLight.Text {
		t.Errorf("styleKiwi foreground did not follow the new theme")
	}

	applyTheme(themeKiwiDark)
	if colKiwi != themeKiwiDark.Kiwi {
		t.Error("applyTheme(kiwi-dark) did not restore the dark palette")
	}
}

// The theme picker previews live — every highlight move applies the theme
// immediately, before the user confirms anything — and cancelling must put
// the previous theme back exactly and leave nothing persisted to disk.
func TestThemeFlowLivePreviewAndEscRestores(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	withRestoredTheme(t)
	applyTheme(themeKiwiDark) // known starting point regardless of test order

	m, _ := newTestModel(t, permission.ModeAsk)

	done := make(chan struct{})
	go func() {
		m.themeFlow(context.Background())
		close(done)
	}()

	req := waitForPickRequest(t, m)
	m = update(t, m, req)
	if m.activePick == nil {
		t.Fatal("theme picker did not open")
	}

	// Highlighting kiwi-light must apply it right away — this is the whole
	// point of onHighlight, verified synchronously since onPickKey runs
	// inline inside update().
	m = update(t, m, key("down"))
	if colKiwi != themeKiwiLight.Kiwi {
		t.Errorf("colKiwi after highlighting kiwi-light = %v, want the light theme's colour", colKiwi)
	}

	m = update(t, m, key("esc"))
	if colKiwi != themeKiwiDark.Kiwi {
		t.Errorf("colKiwi after esc = %v, want the previous theme restored", colKiwi)
	}

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("themeFlow did not return after esc")
	}

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Theme != "" {
		t.Errorf("cfg.Theme = %q, want unset — a cancelled preview must not persist", cfg.Theme)
	}
}

// Confirming a highlighted theme both keeps it applied and writes it to
// kiwi.json, so it survives a restart.
func TestThemeFlowEnterPersistsChoice(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	withRestoredTheme(t)
	applyTheme(themeKiwiDark)

	m, _ := newTestModel(t, permission.ModeAsk)

	done := make(chan struct{})
	go func() {
		m.themeFlow(context.Background())
		close(done)
	}()

	req := waitForPickRequest(t, m)
	m = update(t, m, req)
	m = update(t, m, key("down"))
	m = update(t, m, key("enter"))

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("themeFlow did not return after enter")
	}

	if colKiwi != themeKiwiLight.Kiwi {
		t.Errorf("colKiwi after confirming = %v, want the light theme's colour", colKiwi)
	}

	cfg, err := config.Load()
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if cfg.Theme != "kiwi-light" {
		t.Errorf("cfg.Theme = %q, want %q persisted", cfg.Theme, "kiwi-light")
	}
}
