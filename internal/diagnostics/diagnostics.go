// Package diagnostics runs the project's own checker over the files a tool
// just edited, and hands what it finds back to the model in the same
// observation.
//
// This is the cheap half of language-server support: a fraction of the code,
// and it catches the mistake that matters most — an edit that does not
// compile. The model finds out in the observation it is already reading rather
// than three tool calls later, when it has built more on top of the break.
//
// Two rules keep it from becoming noise, because a diagnostic channel that
// cries wolf gets ignored wholesale:
//
//   - Only problems in the files just edited are reported. A package with
//     forty pre-existing errors elsewhere is not this edit's business.
//   - The same findings are not repeated. If a file's problems are what they
//     were last time, the model already knows.
package diagnostics

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

// EnvCommand overrides the detected checker. It is split on spaces and run in
// the working directory, with the edited file appended when it contains no
// placeholder.
const EnvCommand = "KIWI_DIAGNOSTICS_CMD"

// timeout is how long a check may take before it is abandoned.
//
// Abandoned silently, and that is the point: a checker that doubles the
// latency of every edit gets switched off, and then it is worth nothing at
// all. Fast and occasionally absent beats thorough and resented.
const timeout = 5 * time.Second

// maxReported caps how many problems one observation carries. Past a handful
// the model should go and read the file, and a wall of errors costs the
// context window more than it informs.
const maxReported = 10

// Runner checks edited files against the project's own tooling.
//
// The zero value is not usable; New decides whether there is anything to run
// at all and returns nil when there is not, so the caller's nil check doubles
// as the feature switch.
type Runner struct {
	workDir string
	// override is EnvCommand split into argv, or nil when the checker is
	// detected per language instead.
	override []string

	mu sync.Mutex
	// reported remembers what was last said about each file, so unchanged
	// findings stay quiet.
	reported map[string]string
}

// New returns a Runner for workDir, or nil when nothing can be checked.
func New(workDir string) *Runner {
	r := &Runner{workDir: workDir, reported: map[string]string{}}
	if cmd := strings.Fields(os.Getenv(EnvCommand)); len(cmd) > 0 {
		r.override = cmd
	}
	return r
}

// Check runs the project's checker over the given absolute paths and returns
// a short report, or "" when there is nothing worth saying.
//
// It never returns an error. A checker that is missing, broken or slow is not
// the model's problem to solve mid-edit, and turning one into a failed tool
// call would make edits fail for reasons that have nothing to do with the
// edit.
func (r *Runner) Check(ctx context.Context, paths ...string) string {
	if r == nil || len(paths) == 0 {
		return ""
	}
	argv, ok := r.command(paths[0])
	if !ok {
		return ""
	}

	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	cmd.Dir = r.workDir
	// Checkers split their output between the two streams inconsistently —
	// go vet writes to stderr, tsc to stdout — so both are read as one.
	out, _ := cmd.CombinedOutput()
	if ctx.Err() != nil {
		return ""
	}

	found := parse(string(out))
	if len(found) == 0 {
		// Nothing to report, but the file's slate is now clean: forgetting
		// the old findings is what lets them be reported again if they come
		// back.
		r.forget(paths)
		return ""
	}

	wanted := map[string]bool{}
	for _, p := range paths {
		wanted[p] = true
	}

	var mine []Problem
	for _, p := range found {
		abs := p.File
		if !filepath.IsAbs(abs) {
			abs = filepath.Join(r.workDir, abs)
		}
		if wanted[filepath.Clean(abs)] {
			mine = append(mine, p)
		}
	}
	if len(mine) == 0 {
		r.forget(paths)
		return ""
	}
	sort.SliceStable(mine, func(i, j int) bool {
		if mine[i].File != mine[j].File {
			return mine[i].File < mine[j].File
		}
		return mine[i].Line < mine[j].Line
	})

	report := render(mine, r.rel)
	if r.seen(paths, report) {
		return ""
	}
	return report
}

// seen records the report against every path it covers and reports whether it
// is the same one those paths already produced.
func (r *Runner) seen(paths []string, report string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	repeat := true
	for _, p := range paths {
		if r.reported[p] != report {
			repeat = false
		}
		r.reported[p] = report
	}
	return repeat
}

func (r *Runner) forget(paths []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, p := range paths {
		delete(r.reported, p)
	}
}

// rel shortens a path for the report.
func (r *Runner) rel(p string) string {
	if !filepath.IsAbs(p) {
		return p
	}
	if rel, err := filepath.Rel(r.workDir, p); err == nil && !strings.HasPrefix(rel, "..") {
		return rel
	}
	return p
}

// command decides what to run for a given edited file, and whether there is
// anything to run at all.
//
// The check is scoped to the edited file's own package rather than the whole
// project. On a large repository that is the difference between a check that
// runs inside the timeout and one that never finishes.
func (r *Runner) command(path string) ([]string, bool) {
	if len(r.override) > 0 {
		argv := append([]string(nil), r.override...)
		// A command that names the file itself keeps control of where the
		// path goes; one that does not gets it appended, which is what most
		// linters expect.
		for i, a := range argv {
			if strings.Contains(a, "{}") {
				argv[i] = strings.ReplaceAll(a, "{}", path)
				return argv, true
			}
		}
		return append(argv, path), true
	}

	pkg := "./" + filepath.ToSlash(r.rel(filepath.Dir(path)))
	switch strings.ToLower(filepath.Ext(path)) {
	case ".go":
		if r.has("go.mod") && onPath("go") {
			// vet compiles the package first, so it reports build failures
			// as well as the suspicious constructs it is named for. That is
			// the whole reason it is preferred over a linter here.
			return []string{"go", "vet", pkg}, true
		}
	case ".py":
		if onPath("ruff") && (r.has("pyproject.toml") || r.has("ruff.toml") || r.has(".ruff.toml")) {
			return []string{"ruff", "check", "--quiet", path}, true
		}
	case ".ts", ".tsx", ".js", ".jsx", ".mts", ".cts":
		if !r.has("tsconfig.json") {
			return nil, false
		}
		// The project's own binary, never a downloaded one: reaching for the
		// network in the middle of an edit is not a trade this should make
		// on the user's behalf.
		local := filepath.Join(r.workDir, "node_modules", ".bin", "tsc")
		if _, err := os.Stat(local); err == nil {
			return []string{local, "--noEmit", "--pretty", "false"}, true
		}
		if onPath("tsc") {
			return []string{"tsc", "--noEmit", "--pretty", "false"}, true
		}
	}
	return nil, false
}

func (r *Runner) has(name string) bool {
	_, err := os.Stat(filepath.Join(r.workDir, name))
	return err == nil
}

func onPath(bin string) bool {
	_, err := exec.LookPath(bin)
	return err == nil
}

// Problem is one reported issue.
type Problem struct {
	File string
	Line int
	Msg  string
}

// Two shapes cover every checker worth detecting: "file:line:col: message",
// which go vet, ruff, gcc and most Unix tools emit, and "file(line,col):
// message", which is TypeScript's.
//
// The optional leading word is what go vet puts in front of a build failure
// ("vet: ./main.go:4:2: undefined: f"). Without it the tool's own name is read
// as the start of the path, and the problem is then attributed to a file that
// does not exist — which looks exactly like no problem at all, since only the
// edited file's problems are reported.
var (
	colonForm = regexp.MustCompile(`^(?:[a-zA-Z]+:\s+)?(.+?):(\d+)(?::(\d+))?:\s+(.+)$`)
	parenForm = regexp.MustCompile(`^(.+?)\((\d+),(\d+)\):\s+(.+)$`)
)

// parse pulls problems out of a checker's output, ignoring everything it does
// not recognise — progress lines, summaries, and the noise a build tool emits
// on its way to the answer.
func parse(out string) []Problem {
	var found []Problem
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimRight(line, "\r")
		if strings.TrimSpace(line) == "" || strings.HasPrefix(line, "\t") || strings.HasPrefix(line, " ") {
			continue
		}
		for _, re := range []*regexp.Regexp{parenForm, colonForm} {
			m := re.FindStringSubmatch(line)
			if m == nil {
				continue
			}
			n, err := strconv.Atoi(m[2])
			if err != nil {
				break
			}
			msg := strings.TrimSpace(m[4])
			if msg == "" {
				break
			}
			found = append(found, Problem{File: filepath.Clean(m[1]), Line: n, Msg: msg})
			break
		}
	}
	return found
}

// render turns problems into the block appended to a tool observation.
func render(problems []Problem, short func(string) string) string {
	var b strings.Builder
	b.WriteString("\n\nThe project's checker reports problems in what you just edited:\n")
	for i, p := range problems {
		if i == maxReported {
			b.WriteString("… and " + strconv.Itoa(len(problems)-maxReported) + " more\n")
			break
		}
		b.WriteString("  " + short(p.File) + ":" + strconv.Itoa(p.Line) + ": " + p.Msg + "\n")
	}
	b.WriteString("Fix these before moving on, unless they were already there and are unrelated to your change.")
	return b.String()
}
