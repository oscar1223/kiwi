package tools

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strings"
	"syscall"
	"time"

	"golang.org/x/net/html"

	"github.com/oscar1223/kiwi/internal/permission"
)

const (
	fetchTimeout   = 30 * time.Second
	maxRedirects   = 5
	maxFetchBytes  = MaxFileBytes
	fetchUserAgent = "kiwi/1.0 (+https://github.com/oscar1223/kiwi)"
)

// blockedIP reports whether an address is one a fetch must never reach.
//
// The dangerous one is 169.254.169.254: on every major cloud that is the
// instance metadata endpoint, and it hands out credentials to anything that
// asks. The rest of the ranges are the same idea — a fetch is for reading
// public documentation, and anything it can reach on the local network is
// something the model was not asked to look at.
func blockedIP(ip net.IP) bool {
	return ip == nil ||
		ip.IsLoopback() ||
		ip.IsPrivate() ||
		ip.IsLinkLocalUnicast() ||
		ip.IsLinkLocalMulticast() ||
		ip.IsInterfaceLocalMulticast() ||
		ip.IsMulticast() ||
		ip.IsUnspecified()
}

// safeDialer refuses to open a connection to a blocked address.
//
// The check lives at dial time rather than on the URL on purpose. By the time
// Control runs, the name has already been resolved, so this covers the cases
// that URL inspection misses: a redirect to an internal host, a public name
// with a private A record, and DNS rebinding between the check and the
// connection. Every request and every redirect goes through here.
func safeDialer() *net.Dialer {
	return &net.Dialer{
		Timeout: 10 * time.Second,
		Control: func(network, address string, _ syscall.RawConn) error {
			host, _, err := net.SplitHostPort(address)
			if err != nil {
				return fmt.Errorf("web_fetch: cannot parse address %q", address)
			}
			if ip := net.ParseIP(host); blockedIP(ip) {
				return fmt.Errorf("web_fetch: refusing to connect to %s (private, loopback or link-local address)", host)
			}
			return nil
		},
	}
}

func fetchClient() *http.Client {
	return &http.Client{
		Timeout:   fetchTimeout,
		Transport: &http.Transport{DialContext: safeDialer().DialContext},
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxRedirects {
				return fmt.Errorf("web_fetch: stopped after %d redirects", maxRedirects)
			}
			// The dialer guards the address; this guards the scheme, which the
			// dialer never sees.
			if req.URL.Scheme != "http" && req.URL.Scheme != "https" {
				return fmt.Errorf("web_fetch: refusing to follow a redirect to %s", req.URL.Scheme)
			}
			return nil
		},
	}
}

// WebFetch reads a URL and returns it as text.
type WebFetch struct {
	Perms  *permission.Broker
	Client *http.Client
}

func (WebFetch) Name() string { return "web_fetch" }

func (WebFetch) Description() string {
	return "Fetch a URL and return its content as readable text, with HTML stripped. " +
		"Use it to read documentation, an API reference or a raw file from a " +
		"repository. It reaches public addresses only."
}

func (WebFetch) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"url": map[string]any{"type": "string", "description": "Absolute http:// or https:// URL."},
		},
		"required": []string{"url"},
	}
}

func (t WebFetch) Run(ctx context.Context, input json.RawMessage) (string, error) {
	var in struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal(input, &in); err != nil {
		return "", err
	}
	target := strings.TrimSpace(in.URL)
	if target == "" {
		return "", fmt.Errorf("url is required")
	}
	u, err := url.Parse(target)
	if err != nil {
		return "", fmt.Errorf("bad url %q: %w", target, err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return "", fmt.Errorf("web_fetch only speaks http and https, not %q", u.Scheme)
	}
	if u.Host == "" {
		return "", fmt.Errorf("url %q has no host", target)
	}

	if t.Perms != nil {
		if err := t.Perms.Ask(ctx, permission.Action{
			Name:   permission.ActionFetch,
			Detail: u.String(),
		}); err != nil {
			return "", err
		}
	}

	client := t.Client
	if client == nil {
		client = fetchClient()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", fetchUserAgent)
	req.Header.Set("Accept", "text/html,text/plain,application/json;q=0.9,*/*;q=0.5")

	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("fetching %s: %w", u, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return "", fmt.Errorf("%s returned %s", u, resp.Status)
	}

	ctype := resp.Header.Get("Content-Type")
	if !isTextual(ctype) {
		return "", fmt.Errorf("%s is %s, which is not text", u, firstToken(ctype))
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBytes+1))
	if err != nil {
		return "", fmt.Errorf("reading %s: %w", u, err)
	}
	truncated := len(body) > maxFetchBytes
	if truncated {
		body = body[:maxFetchBytes]
	}

	text := string(body)
	if strings.Contains(strings.ToLower(firstToken(ctype)), "html") {
		// Raw markup would eat the context window and read worse: the model
		// is here for the prose, not the div soup.
		text = htmlToText(text)
	}
	text = strings.TrimSpace(text)
	if text == "" {
		return fmt.Sprintf("(%s returned no readable text)", u), nil
	}

	var b strings.Builder
	fmt.Fprintf(&b, "%s\n\n", u)
	b.WriteString(text)
	if truncated {
		fmt.Fprintf(&b, "\n\n… (truncated at %d KB)", maxFetchBytes/1024)
	}
	return b.String(), nil
}

// isTextual reports whether a content type is worth reading as text.
func isTextual(ctype string) bool {
	t := strings.ToLower(firstToken(ctype))
	switch {
	case t == "":
		// Servers that say nothing are usually serving text; the size cap
		// bounds the damage if they are not.
		return true
	case strings.HasPrefix(t, "text/"):
		return true
	case strings.Contains(t, "json"), strings.Contains(t, "xml"),
		strings.Contains(t, "javascript"), strings.Contains(t, "yaml"):
		return true
	}
	return false
}

func firstToken(ctype string) string {
	return strings.TrimSpace(strings.SplitN(ctype, ";", 2)[0])
}

// blockTags start a new line in the extracted text, so paragraphs and list
// items do not run together into one wall.
var blockTags = map[string]bool{
	"p": true, "div": true, "br": true, "li": true, "tr": true,
	"h1": true, "h2": true, "h3": true, "h4": true, "h5": true, "h6": true,
	"section": true, "article": true, "header": true, "footer": true,
	"blockquote": true, "pre": true, "table": true, "ul": true, "ol": true,
}

// skipTags carry no prose worth keeping.
var skipTags = map[string]bool{
	"script": true, "style": true, "noscript": true, "svg": true,
	"head": true, "nav": true, "iframe": true, "template": true,
}

// htmlToText extracts readable text from a document.
//
// A real parser rather than a regexp over tags: markup that a regexp gets
// wrong turns into text that looks plausible and is missing half the page,
// which is worse than failing.
func htmlToText(doc string) string {
	root, err := html.Parse(strings.NewReader(doc))
	if err != nil {
		return doc
	}

	var b strings.Builder
	var walk func(*html.Node)
	walk = func(n *html.Node) {
		switch n.Type {
		case html.TextNode:
			if text := strings.TrimSpace(n.Data); text != "" {
				b.WriteString(text)
				b.WriteString(" ")
			}
		case html.ElementNode:
			if skipTags[n.Data] {
				return
			}
			if blockTags[n.Data] {
				b.WriteString("\n")
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			walk(c)
		}
		if n.Type == html.ElementNode && blockTags[n.Data] {
			b.WriteString("\n")
		}
	}
	walk(root)

	return collapseBlankLines(b.String())
}

// collapseBlankLines trims each line and squeezes runs of empty ones, which
// is most of what makes extracted HTML readable.
func collapseBlankLines(s string) string {
	var out []string
	blank := false
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			if blank {
				continue
			}
			blank = true
		} else {
			blank = false
		}
		out = append(out, line)
	}
	return strings.TrimSpace(strings.Join(out, "\n"))
}
