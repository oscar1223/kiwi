package config

import (
	"path/filepath"
	"testing"
)

func TestDirsFollowAppName(t *testing.T) {
	configHome, dataHome := t.TempDir(), t.TempDir()
	t.Setenv("XDG_CONFIG_HOME", configHome)
	t.Setenv("XDG_DATA_HOME", dataHome)

	for _, tc := range []struct {
		name string
		dev  bool
	}{
		{"kiwi", false},
		{"kiwi-dev", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			old := appName
			appName = tc.name
			t.Cleanup(func() { appName = old })

			dir, err := Dir()
			if err != nil {
				t.Fatal(err)
			}
			if want := filepath.Join(configHome, tc.name); dir != want {
				t.Errorf("Dir() = %q, want %q", dir, want)
			}

			data, err := DataDir()
			if err != nil {
				t.Fatal(err)
			}
			if want := filepath.Join(dataHome, tc.name); data != want {
				t.Errorf("DataDir() = %q, want %q", data, want)
			}

			if IsDev() != tc.dev {
				t.Errorf("IsDev() = %v, want %v", IsDev(), tc.dev)
			}
		})
	}
}
