package update

import (
	"os"
	"path/filepath"
	"strings"
)

// Method says who owns this binary, worked out from where it lives.
//
// It matters because overwriting a binary that Homebrew or Scoop installed
// breaks their bookkeeping: the formula would still claim the old version, and
// the next upgrade would fight whatever we wrote. Where another tool is in
// charge, kiwi names the command instead of acting.
type Method int

const (
	// Standalone came from install.sh or a manual download. It is ours to
	// replace.
	Standalone Method = iota
	Homebrew
	Scoop
	GoInstall
)

// Detect works out how the running binary was installed.
func Detect() Method {
	exe, err := os.Executable()
	if err != nil {
		return Standalone
	}
	// Homebrew links from bin/ into the Cellar or Caskroom; without resolving
	// the symlink we would only ever see /opt/homebrew/bin and miss it.
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		exe = resolved
	}
	return methodForPath(exe)
}

// methodForPath is the testable half of Detect.
func methodForPath(path string) Method {
	p := filepath.ToSlash(path)
	switch {
	case strings.Contains(p, "/Cellar/"),
		strings.Contains(p, "/Caskroom/"),
		strings.Contains(p, "/linuxbrew/"):
		return Homebrew
	case strings.Contains(p, "/scoop/"):
		return Scoop
	case strings.Contains(p, "/go/bin/"),
		strings.Contains(p, "/go/pkg/mod/"):
		return GoInstall
	}
	return Standalone
}

// SelfManaged reports whether kiwi may replace its own binary.
func (m Method) SelfManaged() bool { return m == Standalone }

// Command is what the user has to run to upgrade through this channel.
func (m Method) Command() string {
	switch m {
	case Homebrew:
		return "brew upgrade --cask kiwi"
	case Scoop:
		return "scoop update kiwi"
	case GoInstall:
		return "go install github.com/" + repo + "/cmd/kiwi@latest"
	default:
		return "kiwi update"
	}
}

// Label names the channel in a sentence.
func (m Method) Label() string {
	switch m {
	case Homebrew:
		return "Homebrew"
	case Scoop:
		return "Scoop"
	case GoInstall:
		return "go install"
	default:
		return "a standalone install"
	}
}
