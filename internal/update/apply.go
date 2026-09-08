package update

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// maxBinarySize bounds what we are willing to write out of an archive. The
// stripped binary is well under 100 MB; the cap is what keeps a malformed or
// hostile archive from filling the disk.
const maxBinarySize = 300 << 20

// downloadTimeout covers the whole transfer, not a single read.
const downloadTimeout = 5 * time.Minute

// Apply downloads the release for this platform, verifies it against the
// published checksums, and replaces the running binary with it.
//
// It reports progress through log, which may be nil.
func Apply(ctx context.Context, version string, log func(string)) error {
	if log == nil {
		log = func(string) {}
	}

	if m := Detect(); !m.SelfManaged() {
		return fmt.Errorf(
			"kiwi was installed through %s; update it with:\n\n  %s",
			m.Label(), m.Command())
	}

	target, err := targetPath()
	if err != nil {
		return err
	}
	dir := filepath.Dir(target)
	if err := writable(dir); err != nil {
		return fmt.Errorf("cannot write to %s: %w", dir, err)
	}

	ctx, cancel := context.WithTimeout(ctx, downloadTimeout)
	defer cancel()

	archive := ArchiveName()
	base := "https://github.com/" + repo + "/releases/download/" + version

	// A scratch directory for the download. The extracted binary does not go
	// here: it has to land next to the target so the final rename is atomic.
	tmp, err := os.MkdirTemp("", "kiwi-update-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp)

	log("downloading " + archive + "…")
	archivePath := filepath.Join(tmp, archive)
	if err := download(ctx, base+"/"+archive, archivePath); err != nil {
		return fmt.Errorf("downloading %s: %w", archive, err)
	}

	log("verifying checksum…")
	sums, err := fetch(ctx, base+"/checksums.txt")
	if err != nil {
		return fmt.Errorf("downloading checksums.txt: %w", err)
	}
	want, ok := checksumFor(sums, archive)
	if !ok {
		return fmt.Errorf("%s is not listed in checksums.txt", archive)
	}
	got, err := sha256File(archivePath)
	if err != nil {
		return err
	}
	if got != want {
		// Never install something we cannot vouch for.
		return fmt.Errorf("checksum mismatch for %s:\n  want %s\n  got  %s", archive, want, got)
	}

	log("unpacking…")
	staged, err := extract(archivePath, dir)
	if err != nil {
		return err
	}
	// Until the swap succeeds this is our mess to clean up.
	defer os.Remove(staged)

	mode := os.FileMode(0o755)
	if fi, err := os.Stat(target); err == nil {
		mode = fi.Mode().Perm()
	}
	if err := os.Chmod(staged, mode); err != nil {
		return err
	}

	// Run the new binary before trusting it. A correct checksum proves we got
	// the file the release published, not that it runs here — a release built
	// for the wrong architecture would pass the hash and fail on exec, and
	// finding that out after the swap means finding out with no kiwi left.
	log("checking the new binary…")
	if err := verifyRuns(ctx, staged); err != nil {
		return fmt.Errorf("the downloaded binary does not run: %w", err)
	}

	log("installing…")
	if err := swap(staged, target); err != nil {
		return fmt.Errorf("replacing %s: %w", target, err)
	}
	return nil
}

// ArchiveName is the release asset for the running platform. It has to match
// the name_template in .goreleaser.yaml.
func ArchiveName() string {
	ext := ".tar.gz"
	if runtime.GOOS == "windows" {
		ext = ".zip"
	}
	return fmt.Sprintf("kiwi_%s_%s%s", runtime.GOOS, runtime.GOARCH, ext)
}

// binaryName is what the binary is called inside the archive.
func binaryName() string {
	if runtime.GOOS == "windows" {
		return "kiwi.exe"
	}
	return "kiwi"
}

// targetPath is the binary to replace, with symlinks resolved so we write the
// real file rather than clobbering someone's link.
func targetPath() (string, error) {
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if resolved, err := filepath.EvalSymlinks(exe); err == nil {
		return resolved, nil
	}
	return exe, nil
}

// writable checks the directory can be written to, so the failure lands here
// with a clear message rather than after a full download.
func writable(dir string) error {
	f, err := os.CreateTemp(dir, ".kiwi-write-check-*")
	if err != nil {
		return err
	}
	name := f.Name()
	_ = f.Close()
	return os.Remove(name)
}

func download(ctx context.Context, url, dst string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%s: %s", url, resp.Status)
	}

	f, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer f.Close()

	if _, err := io.Copy(f, io.LimitReader(resp.Body, maxBinarySize)); err != nil {
		return err
	}
	return f.Sync()
}

func fetch(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: %s", url, resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 1<<20))
}

// checksumFor pulls the hash for name out of a checksums.txt, whose lines are
// "<sha256>  <filename>".
func checksumFor(sums []byte, name string) (string, bool) {
	for line := range strings.Lines(string(sums)) {
		fields := strings.Fields(line)
		if len(fields) == 2 && fields[1] == name {
			return strings.ToLower(fields[0]), true
		}
	}
	return "", false
}

func sha256File(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// extract writes the kiwi binary out of the archive into destDir, returning
// the staged path. destDir is the directory holding the binary being replaced,
// so that the later rename stays on one filesystem and therefore atomic.
func extract(archivePath, destDir string) (string, error) {
	if strings.HasSuffix(archivePath, ".zip") {
		return extractZip(archivePath, destDir)
	}
	return extractTarGz(archivePath, destDir)
}

func extractTarGz(archivePath, destDir string) (string, error) {
	f, err := os.Open(archivePath)
	if err != nil {
		return "", err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return "", err
	}
	defer gz.Close()

	want := binaryName()
	tr := tar.NewReader(gz)
	for {
		h, err := tr.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return "", err
		}
		if h.Typeflag != tar.TypeReg || filepath.Base(h.Name) != want {
			continue
		}
		return stage(tr, destDir)
	}
	return "", fmt.Errorf("no %s inside %s", want, filepath.Base(archivePath))
}

func extractZip(archivePath, destDir string) (string, error) {
	zr, err := zip.OpenReader(archivePath)
	if err != nil {
		return "", err
	}
	defer zr.Close()

	want := binaryName()
	for _, f := range zr.File {
		if f.FileInfo().IsDir() || filepath.Base(f.Name) != want {
			continue
		}
		rc, err := f.Open()
		if err != nil {
			return "", err
		}
		defer rc.Close()
		return stage(rc, destDir)
	}
	return "", fmt.Errorf("no %s inside %s", want, filepath.Base(archivePath))
}

// stage writes r into a temporary file alongside the target binary.
func stage(r io.Reader, destDir string) (string, error) {
	out, err := os.CreateTemp(destDir, ".kiwi-update-*")
	if err != nil {
		return "", err
	}
	path := out.Name()

	n, err := io.Copy(out, io.LimitReader(r, maxBinarySize))
	if err == nil && n == maxBinarySize {
		err = fmt.Errorf("binary is larger than the %d byte limit", int64(maxBinarySize))
	}
	if err == nil {
		err = out.Sync()
	}
	if closeErr := out.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		_ = os.Remove(path)
		return "", err
	}
	return path, nil
}

// verifyRuns executes the staged binary with --version.
func verifyRuns(ctx context.Context, path string) error {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	out, err := exec.CommandContext(ctx, path, "--version").CombinedOutput()
	if err != nil {
		if trimmed := strings.TrimSpace(string(out)); trimmed != "" {
			return fmt.Errorf("%w: %s", err, trimmed)
		}
		return err
	}
	return nil
}
