package tools

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/oscar1223/kiwi/internal/permission"
)

func searchTool(env map[string]string, srv *httptest.Server) WebSearch {
	return WebSearch{
		Perms:   permission.NewBroker(permission.ModeWork, nil),
		Client:  &http.Client{Transport: http.DefaultTransport},
		Env:     func(k string) string { return env[k] },
		ExaURL:  srv.URL + "/exa",
		JinaURL: srv.URL + "/jina",
		DDGURL:  srv.URL + "/ddg",
	}
}

func search(t *testing.T, tool WebSearch, query string) string {
	t.Helper()
	input, _ := json.Marshal(map[string]any{"query": query})
	out, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	return out
}

const ddgPage = `<html><body>
<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgo.dev%2Fdoc%2F&rut=x">Go <b>docs</b></a>
<a class="result__snippet">The Go programming language.</a></div>
<div class="result"><a class="result__a" href="https://duckduckgo.com/y.js?ad=1">An ad</a></div>
<div class="result"><a class="result__a" href="https://pkg.go.dev/">Packages</a></div>
</body></html>`

// With no key it still works, through DuckDuckGo's HTML page, and unwraps its
// redirect links and drops its ads.
func TestWebSearchFallsBackToDuckDuckGo(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/ddg" || r.FormValue("q") != "go docs" {
			t.Errorf("unexpected request %s q=%q", r.URL.Path, r.FormValue("q"))
		}
		w.Write([]byte(ddgPage))
	}))
	defer srv.Close()

	out := search(t, searchTool(nil, srv), "go docs")
	for _, want := range []string{"(duckduckgo)", "1. Go docs", "https://go.dev/doc/", "The Go programming language.", "2. Packages"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in:\n%s", want, out)
		}
	}
	if strings.Contains(out, "An ad") {
		t.Errorf("an ad made it into the results:\n%s", out)
	}
}

func TestWebSearchUsesExaWithAKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/exa" || r.Header.Get("x-api-key") != "k" {
			t.Errorf("unexpected request %s key=%q", r.URL.Path, r.Header.Get("x-api-key"))
		}
		w.Write([]byte(`{"results":[{"title":"Exa hit","url":"https://a.example","text":"snippet"}]}`))
	}))
	defer srv.Close()

	out := search(t, searchTool(map[string]string{"EXA_API_KEY": "k", "JINA_API_KEY": "j"}, srv), "q")
	if !strings.Contains(out, "(exa)") || !strings.Contains(out, "Exa hit") {
		t.Errorf("exa was not used:\n%s", out)
	}
}

func TestWebSearchUsesJinaWithAKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/jina" || r.Header.Get("Authorization") != "Bearer j" || r.URL.Query().Get("q") != "q" {
			t.Errorf("unexpected request %s", r.URL)
		}
		w.Write([]byte(`{"data":[{"title":"Jina hit","url":"https://b.example","description":"d"}]}`))
	}))
	defer srv.Close()

	out := search(t, searchTool(map[string]string{"JINA_API_KEY": "j"}, srv), "q")
	if !strings.Contains(out, "(jina)") || !strings.Contains(out, "Jina hit") {
		t.Errorf("jina was not used:\n%s", out)
	}
}

// A provider's failure reaches the model as an error it can act on.
func TestWebSearchReportsAProviderError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	input, _ := json.Marshal(map[string]any{"query": "x"})
	_, err := searchTool(nil, srv).Run(context.Background(), input)
	if err == nil || !strings.Contains(err.Error(), "EXA_API_KEY") {
		t.Errorf("err = %v, want a hint about keyed providers", err)
	}
}

// Reader mode sends the page through the reader service and returns what it
// renders.
func TestWebFetchReaderMode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/https://spa.example/page" {
			t.Errorf("reader got %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/plain")
		w.Write([]byte("# Rendered page"))
	}))
	defer srv.Close()

	tool := fetchAllowingLoopback()
	tool.ReaderBase = srv.URL + "/"
	input, _ := json.Marshal(map[string]any{"url": "https://spa.example/page", "reader": true})
	out, err := tool.Run(context.Background(), input)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "# Rendered page") {
		t.Errorf("reader output missing:\n%s", out)
	}
}

// A page that comes back nearly empty suggests reader mode instead of
// leaving the model to guess why.
func TestWebFetchSuggestsReaderForAThinPage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		w.Write([]byte(`<html><body><div id="root"></div><script>render()</script></body></html>`))
	}))
	defer srv.Close()

	out, err := fetch(t, fetchAllowingLoopback(), srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out, "reader: true") {
		t.Errorf("no reader hint for an empty page:\n%s", out)
	}
}
