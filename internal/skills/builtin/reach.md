---
name: reach
description: Research on the internet beyond a single web page — search, JavaScript-heavy sites, YouTube videos, GitHub repos, issues and PRs, RSS feeds, Reddit and X threads. Load it when a task needs information from outside the repository.
user-invocable: true
---
# Reaching the internet

The job is to answer from sources, not from memory. Find the page, read it, and
say where the answer came from.

Run `kiwi doctor` through bash first if you are unsure what is installed. It
lists each channel below, whether it works here, and how to install what is
missing. Do not install anything yourself without asking the user.

## Search and read

1. `web_search` finds pages. Search with the words the answer would contain —
   an error message verbatim, a function name, a version number — not a
   question.
2. `web_fetch` reads a result. Read two or three before concluding anything; a
   single blog post is an opinion.
3. When a page comes back nearly empty, or is an app shell rather than content,
   fetch it again with `reader: true`. That renders it in a real browser
   through Jina Reader (r.jina.ai) and returns markdown. It sends the URL to a
   third party, so do not use it for private or internal URLs.

`web_search` uses Exa when `EXA_API_KEY` is set, Jina when `JINA_API_KEY` is,
and DuckDuckGo otherwise. If DuckDuckGo refuses (it answers suspected bots with
a challenge), say so and suggest setting one of the keys with `/config`.

## Channels that need a command-line tool

Run these through `bash`. Each needs its tool installed; `kiwi doctor` says
which are.

**YouTube** — `yt-dlp`. Never download the video itself.
```bash
yt-dlp --dump-json --skip-download URL | head -c 4000    # title, channel, description, duration
yt-dlp --skip-download --write-auto-subs --sub-langs 'en.*,es.*' \
  --sub-format vtt -o '/tmp/kiwi-yt/%(id)s' URL          # subtitles, to read the content
```
Read the `.vtt` file with `read_file`, skimming past the timestamps.

**GitHub** — `gh`. Better than fetching github.com pages: structured, and it
sees private repositories the user has access to.
```bash
gh repo view OWNER/REPO                       # README and description
gh issue list -R OWNER/REPO --search "words"  # find an issue
gh issue view N -R OWNER/REPO --comments
gh pr view N -R OWNER/REPO --comments
gh search code "symbol" --repo OWNER/REPO
gh release list -R OWNER/REPO --limit 5
gh api repos/OWNER/REPO/contents/PATH --jq .content | base64 -d
```

**RSS and Atom** — `web_fetch` the feed URL; it is XML and comes back as text.
Most blogs have one at `/feed`, `/rss.xml` or `/atom.xml`.

**Reddit** — append `.json` to a thread URL and `web_fetch` it
(`https://www.reddit.com/r/SUB/comments/ID/.json`). If Reddit blocks it, fetch
the thread with `reader: true`.

**X / Twitter** — no reliable anonymous API. Try `web_fetch` with
`reader: true` on the post URL; if that fails, say you could not read it rather
than guessing its contents.

## Reporting

Cite every claim with the URL it came from. When sources disagree, say so and
say which one is more authoritative — the project's own docs over a forum post,
a recent release note over an old answer.
