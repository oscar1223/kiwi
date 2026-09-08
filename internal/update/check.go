// Package update discovers newer releases and, when kiwi owns the binary,
// replaces it in place.
//
// Three rules keep the check out of the way: it never blocks startup, it never
// reports an error — a version check that complains about the network is worse
// than no check at all — and KIWI_NO_UPDATE_CHECK=1 turns it off entirely.
package update

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/oscar1223/kiwi/internal/config"
)

const (
	repo = "oscar1223/kiwi"

	latestURL = "https://api.github.com/repos/" + repo + "/releases/latest"

	// interval is how long a lookup stays good for. GitHub allows 60
	// unauthenticated requests an hour per IP; caching for a day means a user
	// who opens kiwi fifty times still spends one of them.
	interval = 24 * time.Hour

	// checkTimeout bounds the background lookup. Startup does not wait on it,
	// but a request left hanging would keep the goroutine alive for the whole
	// session.
	checkTimeout = 3 * time.Second
)

// DevVersion is the value of the version string in a build that did not go
// through the release pipeline. There is nothing to compare it against.
const DevVersion = "dev"

// Check reports the latest published version when it is newer than current,
// and "" when there is nothing worth saying. It deliberately returns no error:
// when in doubt, it stays quiet.
func Check(ctx context.Context, current string) string {
	if os.Getenv("KIWI_NO_UPDATE_CHECK") != "" || current == DevVersion {
		return ""
	}

	path := cachePath()
	latest, fresh := readCache(path)
	if !fresh {
		var err error
		if latest, err = fetchLatest(ctx); err != nil || latest == "" {
			// Deliberately silent: this runs unasked, in the background.
			return ""
		}
		writeCache(path, latest)
	}

	if newer(current, latest) {
		return latest
	}
	return ""
}

// ErrNoReleases means the repository has published no release yet. It is a
// different problem from an unreachable network and deserves a different
// message, so `kiwi update` can say which one happened.
var ErrNoReleases = errors.New("no release has been published yet")

// Latest is Check without the cache and without the silence, for `kiwi update`:
// someone who typed the command is asking now, so a cache written yesterday
// would answer a different question, and a failure is worth reporting.
func Latest(ctx context.Context) (string, error) {
	v, err := fetchLatest(ctx)
	if err != nil {
		return "", err
	}
	writeCache(cachePath(), v)
	return v, nil
}

// IsNewer reports whether latest is a higher version than current.
func IsNewer(current, latest string) bool { return newer(current, latest) }

type cacheFile struct {
	Latest    string    `json:"latest"`
	CheckedAt time.Time `json:"checked_at"`
}

func cachePath() string {
	// DataDir, not Dir: this is state kiwi keeps for itself, not configuration
	// anyone is meant to open.
	dir, err := config.DataDir()
	if err != nil {
		return ""
	}
	return filepath.Join(dir, "update-check.json")
}

func readCache(path string) (latest string, fresh bool) {
	if path == "" {
		return "", false
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return "", false
	}
	var c cacheFile
	if err := json.Unmarshal(b, &c); err != nil {
		return "", false
	}
	// A timestamp in the future means a clock that moved; treat it as stale
	// rather than trusting it until the clock catches up.
	age := time.Since(c.CheckedAt)
	return c.Latest, age >= 0 && age < interval
}

func writeCache(path, latest string) {
	if path == "" {
		return
	}
	b, err := json.Marshal(cacheFile{Latest: latest, CheckedAt: time.Now()})
	if err != nil {
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return
	}
	_ = os.WriteFile(path, b, 0o644)
}

func fetchLatest(ctx context.Context) (string, error) {
	ctx, cancel := context.WithTimeout(ctx, checkTimeout)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, latestURL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "application/vnd.github+json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("reaching GitHub: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		// The repository exists but has no releases — the state kiwi itself
		// is in before its first tag.
		return "", ErrNoReleases
	case http.StatusForbidden, http.StatusTooManyRequests:
		// 60 requests an hour per IP without a token. The daily cache means a
		// normal user never sees this; a shared NAT might.
		return "", errors.New("GitHub is rate-limiting this network; try again later")
	default:
		return "", fmt.Errorf("GitHub returned %s", resp.Status)
	}

	var body struct {
		TagName    string `json:"tag_name"`
		Draft      bool   `json:"draft"`
		Prerelease bool   `json:"prerelease"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return "", fmt.Errorf("reading the release feed: %w", err)
	}
	if body.Draft || body.Prerelease {
		return "", ErrNoReleases
	}
	if body.TagName == "" {
		return "", ErrNoReleases
	}
	return body.TagName, nil
}

// newer compares two vX.Y.Z versions. Anything with a suffix (-rc1, +meta) has
// it trimmed, and a version that does not parse means false: on that ground we
// would rather say nothing than guess half right.
func newer(current, latest string) bool {
	c, okC := parse(current)
	l, okL := parse(latest)
	if !okC || !okL {
		return false
	}
	for i := range c {
		if l[i] != c[i] {
			return l[i] > c[i]
		}
	}
	return false
}

func parse(v string) ([3]int, bool) {
	var out [3]int
	v = strings.TrimPrefix(strings.TrimSpace(v), "v")
	if i := strings.IndexAny(v, "-+"); i >= 0 {
		v = v[:i]
	}
	parts := strings.Split(v, ".")
	if len(parts) != 3 {
		return out, false
	}
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil || n < 0 {
			return out, false
		}
		out[i] = n
	}
	return out, true
}
