// Package permission decides which tool actions may proceed.
//
// Two things live here. The *policy* is a pure function of (mode, action): it
// answers allow / deny / "ask the human" with no I/O, which makes it trivially
// testable. The *broker* is the channel-based plumbing that carries the
// questions the policy could not answer to whatever UI is attached.
package permission

import (
	"regexp"
	"strings"
)

// Mode is the permission stance Kiwi operates under. It is cycled with
// Shift+Tab in the TUI, from most to least conservative.
type Mode string

const (
	// ModeAsk confirms every mutating action. The default.
	ModeAsk Mode = "ask"
	// ModePlan is read-only: edits are refused outright and only read-only
	// commands run. Intended for investigating before touching anything.
	ModePlan Mode = "plan"
	// ModeWork applies edits without confirmation. Commands still ask.
	ModeWork Mode = "work"
)

// Order is the Shift+Tab cycle.
var Order = []Mode{ModeAsk, ModePlan, ModeWork}

func (m Mode) Label() string {
	switch m {
	case ModeAsk:
		return "Ask"
	case ModePlan:
		return "Plan"
	case ModeWork:
		return "Work"
	default:
		return string(m)
	}
}

func (m Mode) Valid() bool {
	switch m {
	case ModeAsk, ModePlan, ModeWork:
		return true
	}
	return false
}

// Next returns the mode after m in the cycle.
func (m Mode) Next() Mode {
	for i, mode := range Order {
		if mode == m {
			return Order[(i+1)%len(Order)]
		}
	}
	return ModeAsk
}

// Instructions is the system-prompt fragment telling the model what the
// current mode allows. Keeping the model informed matters: a model that knows
// edits are blocked proposes a plan instead of burning turns on refusals.
func (m Mode) Instructions() string {
	switch m {
	case ModePlan:
		return `[CURRENT MODE: Plan]
You are in read-only mode. write_file and edit_file are blocked, and bash only
runs read-only commands (ls, cat, grep, find, git status/log/diff, …) — anything
else is denied automatically without asking the user.

Investigate as much as you need, then answer with a clear numbered plan instead
of applying it. If a tool is blocked, do not retry it: note in the plan that the
step needs the user to leave Plan mode.`

	case ModeWork:
		return `[CURRENT MODE: Work]
write_file and edit_file apply without asking the user, so be deliberate about
what you write. bash and MCP tools still ask for confirmation.`

	default:
		return ""
	}
}

// Standard action names. Tools report themselves under these so policy does
// not have to know about every tool implementation.
const (
	ActionRead  = "read_file"
	ActionWrite = "write_file"
	ActionEdit  = "edit_file"
	ActionBash  = "bash"
	// MCPPrefix marks actions coming from MCP servers: "mcp:server/tool".
	MCPPrefix = "mcp:"
)

// Action describes what a tool wants to do.
type Action struct {
	// Name is one of the Action* constants, or an MCPPrefix-prefixed name.
	Name string
	// Detail is the command line, path, or other one-line summary shown to
	// the user.
	Detail string
	// Diff, when set, is a unified diff previewed before approval.
	Diff string
	// Dangerous marks actions that warrant a louder prompt.
	Dangerous bool
}

// Resolve applies the mode policy to an action.
//
// It returns (decision, true) when the mode decides on its own, and
// (false, false) when the human must be asked.
func Resolve(mode Mode, a Action) (allow, decided bool) {
	switch mode {
	case ModePlan:
		switch {
		case a.Name == ActionWrite, a.Name == ActionEdit:
			return false, true
		case a.Name == ActionBash:
			return IsReadOnlyCommand(a.Detail), true
		case strings.HasPrefix(a.Name, MCPPrefix):
			// MCP tools are opaque; assume they can mutate.
			return false, true
		}
		return false, false

	case ModeWork:
		if a.Name == ActionWrite || a.Name == ActionEdit {
			return true, true
		}
		return false, false

	default: // ModeAsk
		return false, false
	}
}

// readOnlyVerbs are commands that inspect without mutating.
var readOnlyVerbs = map[string]bool{
	"ls": true, "cat": true, "grep": true, "rg": true, "find": true,
	"head": true, "tail": true, "wc": true, "pwd": true, "tree": true,
	"file": true, "which": true, "echo": true, "du": true, "df": true,
	"ps": true, "whoami": true, "date": true, "diff": true, "stat": true,
	"basename": true, "dirname": true, "realpath": true, "sort": true,
	"uniq": true, "column": true, "jq": true, "env": true, "printenv": true,
}

var readOnlyGitSubcommands = map[string]bool{
	"status": true, "log": true, "diff": true, "show": true, "branch": true,
	"remote": true, "blame": true, "shortlog": true, "describe": true,
	"rev-parse": true, "ls-files": true, "stash": false,
}

// mutationMarkers appearing anywhere in a segment disqualify it, even as an
// argument: `find . -exec rm {} \;` must not pass as read-only.
var mutationMarkers = map[string]bool{
	"rm": true, "mv": true, "cp": true, "touch": true, "mkdir": true,
	"rmdir": true, "chmod": true, "chown": true, "ln": true, "sudo": true,
	"kill": true, "dd": true, "install": true, "tee": true, "truncate": true,
	"-exec": true, "-delete": true, "-execdir": true,
}

var dangerousPatterns = []*regexp.Regexp{
	regexp.MustCompile(`\brm\s+(-[a-zA-Z]*\s+)*-[a-zA-Z]*[rf]`),
	regexp.MustCompile(`\bsudo\b`),
	regexp.MustCompile(`\bdd\b.*\bof=`),
	regexp.MustCompile(`\bmkfs\b`),
	regexp.MustCompile(`>\s*/dev/(sd|nvme|disk)`),
	regexp.MustCompile(`\bgit\s+push\b.*(--force|-f)\b`),
	regexp.MustCompile(`\bgit\s+reset\s+--hard\b`),
	regexp.MustCompile(`\bgit\s+clean\b.*-[a-zA-Z]*f`),
	regexp.MustCompile(`\bchmod\s+(-[a-zA-Z]+\s+)*777\b`),
	regexp.MustCompile(`\bcurl\b[^|]*\|\s*(ba)?sh\b`),
	regexp.MustCompile(`\bwget\b[^|]*\|\s*(ba)?sh\b`),
	regexp.MustCompile(`:\(\)\s*\{.*\|.*&.*\}`), // fork bomb
}

// IsDangerous reports whether a command deserves a louder confirmation prompt.
func IsDangerous(command string) bool {
	for _, re := range dangerousPatterns {
		if re.MatchString(command) {
			return true
		}
	}
	return false
}

// segmentSplitter splits a command line on shell operators.
var segmentSplitter = regexp.MustCompile(`&&|\|\||;|\|`)

// IsReadOnlyCommand reports whether a command only inspects the system.
//
// This is a conservative heuristic, not a sandbox: it does not understand
// subshells, aliases, or variable expansion. Anything it cannot confidently
// classify is treated as unsafe, so the cost of being wrong is an extra
// confirmation prompt rather than an unexpected mutation.
func IsReadOnlyCommand(command string) bool {
	if IsDangerous(command) {
		return false
	}
	// Command substitution can hide anything.
	if strings.Contains(command, "$(") || strings.Contains(command, "`") {
		return false
	}

	sawSegment := false
	for _, segment := range segmentSplitter.Split(command, -1) {
		segment = strings.TrimSpace(segment)
		if segment == "" {
			continue
		}
		sawSegment = true

		// Any redirection can write.
		if strings.Contains(segment, ">") {
			return false
		}

		words := strings.Fields(segment)
		if len(words) == 0 {
			continue
		}
		for _, w := range words {
			if mutationMarkers[w] {
				return false
			}
		}

		verb := words[0]
		if verb == "git" {
			if len(words) < 2 || !readOnlyGitSubcommands[words[1]] {
				return false
			}
			continue
		}
		if !readOnlyVerbs[verb] {
			return false
		}
	}
	return sawSegment
}
