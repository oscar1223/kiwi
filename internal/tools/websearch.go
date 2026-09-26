package tools

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"golang.org/x/net/html"

	"github.com/oscar1223/kiwi/internal/permission"
)

const (
	defaultSearchResults = 8
	maxSearchResults     = 20

	exaSearchURL  = "https://api.exa.ai/search"
	jinaSearchURL = "https://s.jina.ai/"
	ddgSearchURL  = "https://html.duckduckgo.com/html/"
)

// WebSearch looks something up on the web and returns a list of results.
//
// It uses the first provider it has credentials for: Exa (EXA_API_KEY), then
// Jina Search (JINA_API_KEY), then DuckDuckGo's HTML page, which needs none.
// So it works out of the box, and a key buys better results.
type WebSearch struct {
	Perms  *permission.Broker
	Client *http.Client
	// Env reads a variable; nil means os.Getenv. The tests use it to pick a
	// provider without touching the process environment.
	Env func(string) string
	// The providers' endpoints. Empty means the real ones; the tests point
	// them at their own servers.
	ExaURL, JinaURL, DDGURL string
}

// SearchResult is one hit, whichever provider found it.
type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

func (WebSearch) Name() string { return "web_search" }

func (WebSearch) Description() string {
	return "Search the web and return a numbered list of results: title, URL and a " +
		"short snippet. Use it to find documentation, recent releases, error " +
		"messages or anything you do not already know the URL for, then read " +
		"the promising results with web_fetch."
}

func (WebSearch) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string", "description": "What to search for."},
			"max_results": map[string]any{
				"type":        "integer",
				"description": fmt.Sprintf("How many results to return (default %d, at most %d).", defaultSearchResults, maxSearchResults),
			},
		},
		"required": []string{"query"},
	}
}

// SearchProvider names the provider a search would use with the given
// environment, for kiwi doctor as much as for Run.
func SearchProvider(env func(string) string) string {
	if env == nil {
		env = os.Getenv
	}
	switch {
	case env("EXA_API_KEY") != "":
		return "exa"
	case env("JINA_API_KEY") != "":
		return "jina"
	default:
		return "duckduckgo"
	}
}

func (t WebSearch) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		Query      string `json:"query"`
		MaxResults int    `json:"max_results"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	query := strings.TrimSpace(in.Query)
	if query == "" {
		return "", fmt.Errorf("query is required")
	}
	n := in.MaxResults
	if n <= 0 {
		n = defaultSearchResults
	}
	n = min(n, maxSearchResults)

	env := t.Env
	if env == nil {
		env = os.Getenv
	}
	provider := SearchProvider(env)

	if t.Perms != nil {
		if err := t.Perms.Ask(ctx, permission.Action{
			Name:   permission.ActionFetch,
			Detail: fmt.Sprintf("search %q (%s)", query, provider),
		}); err != nil {
			return "", err
		}
	}

	var (
		results []SearchResult
		err     error
	)
	switch provider {
	case "exa":
		results, err = t.exa(ctx, env("EXA_API_KEY"), query, n)
	case "jina":
		results, err = t.jina(ctx, env("JINA_API_KEY"), query)
	default:
		results, err = t.duckduckgo(ctx, query)
	}
	if err != nil {
		return "", fmt.Errorf("%s search: %w", provider, err)
	}
	return formatResults(query, provider, results, n), nil
}

func formatResults(query, provider string, results []SearchResult, n int) string {
	if len(results) == 0 {
		return fmt.Sprintf("No results for %q (%s).", query, provider)
	}
	var b strings.Builder
	fmt.Fprintf(&b, "Results for %q (%s):\n", query, provider)
	for i, r := range results[:min(n, len(results))] {
		fmt.Fprintf(&b, "\n%d. %s\n   %s\n", i+1, oneLineText(r.Title), r.URL)
		if s := oneLineText(r.Snippet); s != "" {
			fmt.Fprintf(&b, "   %s\n", truncateText(s, 300))
		}
	}
	return b.String()
}

func (t WebSearch) client() *http.Client {
	if t.Client != nil {
		return t.Client
	}
	return fetchClient()
}

// doJSON sends req and decodes a JSON reply into out.
func (t WebSearch) doJSON(req *http.Request, out any) error {
	req.Header.Set("User-Agent", fetchUserAgent)
	req.Header.Set("Accept", "application/json")
	resp, err := t.client().Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		return err
	}
	if resp.StatusCode >= 400 {
		return fmt.Errorf("%s: %s", resp.Status, truncateText(oneLineText(string(body)), 200))
	}
	return json.Unmarshal(body, out)
}

func (t WebSearch) exa(ctx context.Context, key, query string, n int) ([]SearchResult, error) {
	payload, _ := json.Marshal(map[string]any{
		"query":      query,
		"numResults": n,
		"contents":   map[string]any{"text": map[string]any{"maxCharacters": 400}},
	})
	endpoint := firstNonBlank(t.ExaURL, exaSearchURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", key)

	var out struct {
		Results []struct {
			Title string `json:"title"`
			URL   string `json:"url"`
			Text  string `json:"text"`
		} `json:"results"`
	}
	if err := t.doJSON(req, &out); err != nil {
		return nil, err
	}
	results := make([]SearchResult, 0, len(out.Results))
	for _, r := range out.Results {
		results = append(results, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Text})
	}
	return results, nil
}

func (t WebSearch) jina(ctx context.Context, key, query string) ([]SearchResult, error) {
	endpoint := firstNonBlank(t.JinaURL, jinaSearchURL) + "?q=" + url.QueryEscape(query)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+key)
	// Just the result list: the pages themselves are web_fetch's job.
	req.Header.Set("X-Respond-With", "no-content")

	var out struct {
		Data []struct {
			Title       string `json:"title"`
			URL         string `json:"url"`
			Description string `json:"description"`
		} `json:"data"`
	}
	if err := t.doJSON(req, &out); err != nil {
		return nil, err
	}
	results := make([]SearchResult, 0, len(out.Data))
	for _, r := range out.Data {
		results = append(results, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Description})
	}
	return results, nil
}

func (t WebSearch) duckduckgo(ctx context.Context, query string) ([]SearchResult, error) {
	form := url.Values{"q": {query}}
	endpoint := firstNonBlank(t.DDGURL, ddgSearchURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("User-Agent", fetchUserAgent)
	req.Header.Set("Accept", "text/html")

	resp, err := t.client().Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	// DuckDuckGo answers a client it suspects of being a bot with a 202 and
	// a challenge page rather than an error.
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("DuckDuckGo returned %s; set EXA_API_KEY or JINA_API_KEY for a keyed provider", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes))
	if err != nil {
		return nil, err
	}
	return parseDuckDuckGo(string(body))
}

// parseDuckDuckGo reads the results out of DuckDuckGo's HTML page: each one
// is an a.result__a for the title and link, followed by a .result__snippet.
func parseDuckDuckGo(doc string) ([]SearchResult, error) {
	root, err := html.Parse(strings.NewReader(doc))
	if err != nil {
		return nil, err
	}
	var results []SearchResult
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.ElementNode {
			switch {
			case n.Data == "a" && hasClass(n, "result__a"):
				results = append(results, SearchResult{
					Title: nodeText(n),
					URL:   ddgTarget(attr(n, "href")),
				})
				return
			case hasClass(n, "result__snippet") && len(results) > 0:
				results[len(results)-1].Snippet = nodeText(n)
				return
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(root)

	// Ads link through DuckDuckGo's own click tracker; they are not results.
	out := results[:0]
	for _, r := range results {
		if r.URL != "" && !strings.Contains(r.URL, "duckduckgo.com/y.js") {
			out = append(out, r)
		}
	}
	return out, nil
}

// ddgTarget unwraps DuckDuckGo's redirect link (//duckduckgo.com/l/?uddg=…)
// into the address it points at.
func ddgTarget(href string) string {
	u, err := url.Parse(href)
	if err != nil {
		return href
	}
	if target := u.Query().Get("uddg"); target != "" {
		return target
	}
	if u.Scheme == "" && strings.HasPrefix(href, "//") {
		return "https:" + href
	}
	return href
}

func attr(n *html.Node, name string) string {
	for _, a := range n.Attr {
		if a.Key == name {
			return a.Val
		}
	}
	return ""
}

func hasClass(n *html.Node, class string) bool {
	for _, c := range strings.Fields(attr(n, "class")) {
		if c == class {
			return true
		}
	}
	return false
}

func nodeText(n *html.Node) string {
	var b strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		if n.Type == html.TextNode {
			b.WriteString(n.Data)
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
	}
	walk(n)
	return strings.TrimSpace(b.String())
}

func oneLineText(s string) string { return strings.Join(strings.Fields(s), " ") }

func truncateText(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}

func firstNonBlank(a, b string) string {
	if a != "" {
		return a
	}
	return b
}
