package tools

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/permission"
)

// fetchTool returns a WebFetch that talks to test servers.
//
// Work mode so no decider is needed, and the real client so the SSRF guard is
// exercised rather than stubbed — the guard is the part worth testing.
func fetchTool() WebFetch {
	return WebFetch{
		Perms:  permission.NewBroker(permission.ModeWork, nil),
		Client: fetchClient(),
	}
}

// fetchAllowingLoopback is the same tool with the address guard removed, for
// the tests that need to reach httptest at all. Anything about the guard
// itself uses fetchTool instead.
func fetchAllowingLoopback() WebFetch {
	return WebFetch{
		Perms:  permission.NewBroker(permission.ModeWork, nil),
		Client: &http.Client{Transport: http.DefaultTransport},
	}
}

func fetch(t *testing.T, tool WebFetch, url string) (string, error) {
	t.Helper()
	input, err := json.Marshal(map[string]any{"url": url})
	if err != nil {
		t.Fatal(err)
	}
	return tool.Run(context.Background(), input)
}

func TestWebFetchReturnsPlainText(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte("hola desde el servidor"))
	}))
	defer srv.Close()

	out, err := fetch(t, fetchAllowingLoopback(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "hola desde el servidor") {
		t.Errorf("the body is missing:\n%s", out)
	}
	if !strings.Contains(out, srv.URL) {
		t.Errorf("the answer does not say where it came from:\n%s", out)
	}
}

func TestWebFetchStripsHTML(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(`<html><head><title>t</title>
			<style>body{color:red}</style></head>
			<body><h1>Título</h1><p>Un párrafo.</p>
			<script>alert('no')</script>
			<ul><li>uno</li><li>dos</li></ul></body></html>`))
	}))
	defer srv.Close()

	out, err := fetch(t, fetchAllowingLoopback(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"Título", "Un párrafo.", "uno", "dos"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	for _, unwanted := range []string{"<p>", "alert(", "color:red"} {
		if strings.Contains(out, unwanted) {
			t.Errorf("markup or script survived (%q):\n%s", unwanted, out)
		}
	}
}

// The guard that matters most. httptest listens on loopback, which is exactly
// what a fetch must refuse to reach.
func TestWebFetchRefusesLoopback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("secreto interno"))
	}))
	defer srv.Close()

	out, err := fetch(t, fetchTool(), srv.URL)
	if err == nil {
		t.Fatalf("a fetch to loopback succeeded: %s", out)
	}
	if strings.Contains(out, "secreto interno") {
		t.Error("the body leaked despite the error")
	}
}

// The case that URL inspection misses: the first hop is fine and the redirect
// points somewhere internal. Checking at dial time is what catches it.
func TestWebFetchRefusesARedirectToAPrivateAddress(t *testing.T) {
	internal := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("credenciales"))
	}))
	defer internal.Close()

	// A redirect straight at the internal server. The dialer sees the
	// resolved address of the second hop and refuses it.
	redirector := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, internal.URL, http.StatusFound)
	}))
	defer redirector.Close()

	tool := WebFetch{
		Perms: permission.NewBroker(permission.ModeWork, nil),
		// Loopback is allowed for the first hop only, so the test can reach
		// the redirector at all; the second hop goes through the real guard.
		Client: &http.Client{
			Transport: &http.Transport{DialContext: firstHopOnlyDialer(redirector.URL)},
		},
	}

	out, err := fetch(t, tool, redirector.URL)
	if err == nil {
		t.Fatalf("a redirect to a private address was followed: %s", out)
	}
	if strings.Contains(out, "credenciales") {
		t.Error("the internal body leaked through the redirect")
	}
}

// firstHopOnlyDialer allows exactly the given host and applies the real guard
// to everything else, which is how a redirect can be isolated in a test where
// every server is on loopback.
func firstHopOnlyDialer(allowedURL string) func(context.Context, string, string) (net.Conn, error) {
	allowed := strings.TrimPrefix(allowedURL, "http://")
	guarded := safeDialer()
	plain := &net.Dialer{}
	return func(ctx context.Context, network, address string) (net.Conn, error) {
		if address == allowed {
			return plain.DialContext(ctx, network, address)
		}
		return guarded.DialContext(ctx, network, address)
	}
}

func TestWebFetchRejectsNonHTTPSchemes(t *testing.T) {
	for _, target := range []string{"file:///etc/passwd", "ftp://example.com/x", "gopher://example.com"} {
		if _, err := fetch(t, fetchTool(), target); err == nil {
			t.Errorf("%s was accepted", target)
		}
	}
}

func TestWebFetchReportsAnErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "nope", http.StatusNotFound)
	}))
	defer srv.Close()

	_, err := fetch(t, fetchAllowingLoopback(), srv.URL)
	if err == nil {
		t.Fatal("a 404 was reported as success")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("the error does not name the status: %v", err)
	}
}

func TestWebFetchRefusesBinaryContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte{0x89, 'P', 'N', 'G'})
	}))
	defer srv.Close()

	if _, err := fetch(t, fetchAllowingLoopback(), srv.URL); err == nil {
		t.Error("an image was accepted as text")
	}
}

func TestWebFetchTruncatesLargeBodies(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		_, _ = w.Write([]byte(strings.Repeat("x", maxFetchBytes*2)))
	}))
	defer srv.Close()

	out, err := fetch(t, fetchAllowingLoopback(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "truncated") {
		t.Error("an oversized body was not reported as truncated")
	}
	if len(out) > maxFetchBytes+1024 {
		t.Errorf("the answer is %d bytes, past the cap", len(out))
	}
}

// Plan mode is read-only for the disk but deliberately open to the network:
// without this it would plan from memory.
func TestWebFetchIsAllowedInPlanMode(t *testing.T) {
	for _, mode := range []permission.Mode{permission.ModePlan, permission.ModeWork} {
		allow, decided := permission.Resolve(mode, permission.Action{Name: permission.ActionFetch})
		if !allow || !decided {
			t.Errorf("%s mode does not allow web_fetch outright", mode)
		}
	}
	if _, decided := permission.Resolve(permission.ModeAsk, permission.Action{Name: permission.ActionFetch}); decided {
		t.Error("ask mode should put a fetch to the user rather than deciding it")
	}
}

func TestBlockedIP(t *testing.T) {
	blocked := []string{
		"127.0.0.1", "::1", // loopback
		"10.0.0.1", "192.168.1.1", "172.16.0.1", // private
		"169.254.169.254",          // cloud metadata, the one that matters
		"fe80::1", "0.0.0.0", "::", // link-local and unspecified
	}
	for _, s := range blocked {
		if !blockedIP(net.ParseIP(s)) {
			t.Errorf("%s should be blocked", s)
		}
	}
	for _, s := range []string{"1.1.1.1", "93.184.216.34", "2606:4700:4700::1111"} {
		if blockedIP(net.ParseIP(s)) {
			t.Errorf("%s is a public address and should be allowed", s)
		}
	}
}
