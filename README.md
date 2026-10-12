<p align="center"><img src="assets/kiwi.svg" width="116" height="116" alt="kiwi"></p>

# Kiwi

Kiwi is a local-first coding agent for the terminal: conversation memory, tools over
your file system, command execution and a TUI inspired by tools like Claude Code. It
runs on your machine, saves every session locally in SQLite and only acts on your
project with the permission you give it.

## Installation

**macOS and Linux**

```sh
curl -fsSL https://raw.githubusercontent.com/oscar1223/kiwi/main/install.sh | sh
```

**Windows** (PowerShell)

```powershell
irm https://raw.githubusercontent.com/oscar1223/kiwi/main/install.ps1 | iex
```

**Homebrew**

```sh
brew install oscar1223/tap/kiwi
```

**Scoop**

```powershell
scoop bucket add oscar1223 https://github.com/oscar1223/scoop-bucket
scoop install kiwi
```

You can also download the binary by hand from
[Releases](https://github.com/oscar1223/kiwi/releases).

> On macOS, a file downloaded **with the browser** is quarantined.
> If Gatekeeper blocks it: `xattr -d com.apple.quarantine ./kiwi`.
> Downloading it with `curl` or installing it with `brew` does not have that problem.

> On Windows, Kiwi's `bash` tool runs commands with `bash -c`, so you need
> **Git Bash or WSL** on your `PATH`. Without them the binary starts and the TUI
> works, but any command the agent tries to run will fail.

### From source

Requires Go 1.27+:

```sh
go install github.com/oscar1223/kiwi/cmd/kiwi@latest
```

Or building from the repo:

```sh
go build -o kiwi ./cmd/kiwi
```

## Updating

```sh
kiwi update
```

It downloads the build for your platform, **verifies the SHA256** against the checksums
published with the release and replaces the running binary. On Unix the replacement is
an atomic `rename`, so the session that launched the command stays alive on the old
binary; on Windows the old `.exe` is moved aside and deleted on the next start, because
Windows does not let you delete an executable that is in use.

If kiwi was installed by a package manager, **the binary is the manager's**: overwriting
it by hand would leave Homebrew or Scoop pointing at a version that is no longer on disk.
In that case `kiwi update` detects the channel and gives you the right command instead of
touching the file:

| Installed with | `kiwi update` tells you |
| --- | --- |
| script or manual download | *(updates itself)* |
| Homebrew | `brew upgrade --cask kiwi` |
| Scoop | `scoop update kiwi` |
| `go install` | `go install github.com/oscar1223/kiwi/cmd/kiwi@latest` |

`kiwi update --check` only looks for something new, without installing anything.

On startup, kiwi checks **at most once a day** whether there is a new version and
mentions it in one line. The check never blocks startup and never prints an error — one
that complains about the network is worse than not having it. To turn it off:
`KIWI_NO_UPDATE_CHECK=1`.

## First run

```sh
kiwi
```

With no provider configured, the first time you start `kiwi` an onboarding wizard asks
for the provider (Anthropic or any OpenAI-compatible API — OpenAI, Ollama, OpenRouter,
Groq...), the model and the API key, and leaves everything ready to chat. You can redo or
adjust it later with `/settings` → `Model profiles`, or directly with `/model`.

## Permission modes

Kiwi always runs in one of three modes, which decide what it can do without asking you:

| Mode   | What it allows                               |
|--------|-----------------------------------------------|
| `ask`  | confirms every action before running it (default) |
| `plan` | read-only — cannot edit or run commands |
| `work` | applies edits, commands and MCP tools automatically; only asks for dangerous commands (rm -rf, sudo, force push, ...) |

Pick one with `--mode` at startup, switch on the fly with `/ask` `/plan` `/work`, or
cycle with `shift+tab`.

## TUI commands

Typing `/` shows the available commands and filters them as you type (arrows to
navigate, `tab` or `enter` to complete).

| Command      | What it does                                         |
|--------------|------------------------------------------------------|
| `/init`      | writes or updates `KIWI.md` after reading the project |
| `/status`    | everything about this session on one screen          |
| `/context`   | what is filling the context window                   |
| `/tools`     | lists the tools, or turns one off for this session   |
| `/undo`      | undoes the last turn: files and conversation         |
| `/redo`      | redoes what `/undo` removed                          |
| `/diff`      | what changed in the session (`/diff turn`, the last one) |
| `/review`    | reviews the current changes with a separate agent    |
| `/settings`  | grouped menu with everything below                   |
| `/model`     | switches or manages model profiles                   |
| `/reasoning` | how much the model thinks: `off` … `max` (saved in the profile) |
| `/config`    | manages `.env` variables                             |
| `/mcp`       | manages MCP servers                                  |
| `/doctor`    | which ways of reaching the internet work here        |
| `/skill`     | manages skills                                       |
| `/theme`     | changes the color theme, with a live preview         |
| `/sessions`  | switches between the project's saved conversations   |
| `/memory`    | shows or edits what kiwi remembers                   |
| `/compact`   | summarizes the conversation to free up context       |
| `/clear`     | forgets the current conversation                     |
| `/help`      | lists every command and shortcut                     |
| `/quit`      | exits kiwi                                           |

## Agent tools

- `read_file` / `write_file` / `edit_file` — reading and editing the file system, with a
  diff before applying.
- `multi_edit` — several replacements in the same file in one call. Either all of them
  apply or the file is left alone: a batch that failed halfway would leave the file in a
  state nobody asked for.
- `grep` / `glob` / `ls` — search by content, by name, and directory listing. They do not
  ask for permission (like `read_file`) and skip `.git`, `node_modules` and friends on
  their own. Better than calling `grep` or `find` through `bash`: they do not depend on
  which binaries are installed.
- `todo_write` / `todo_read` — the task list for the work in progress. It is what keeps
  long tasks on course, and it also shows you where things are.
- `web_fetch` — reads a URL and returns the text, with the HTML already cleaned up. It
  rejects loopback, private and link-local addresses — including `169.254.169.254`, the
  cloud metadata one — and checks this when opening the connection, so a redirect
  inwards does not sneak through either. Free in `plan` and `work`, asks in `ask`.
  With `reader: true` the page is rendered by [Jina Reader](https://jina.ai/reader)
  (`r.jina.ai`) and comes back as markdown: useful for JavaScript-built sites. That URL
  does go through a third party, and the permission prompt says so.
- `web_search` — searches the web and returns title, URL and snippet. Uses Exa if
  `EXA_API_KEY` is set, Jina if `JINA_API_KEY` is set and, with no key, DuckDuckGo. Same
  permissions as `web_fetch`.
- `bash` — runs commands; in the background with `background_bash` /
  `background_output` / `kill_shell` for long-running processes.
- `task` — launches subagents for research or parallel work.
- `load_skill` — loads instructions on demand from installed skills.
- `remember` — saves a durable fact to memory (see below). It goes through the same
  permission as a file write: blocked in `plan`, asks in `ask`, saved automatically in
  `work`.
- `ask_questions` — the model asks you up to several multiple-choice questions (with
  room for a free answer and, where it fits, multiple selection) before giving you a
  final plan, just like Claude Code does. Available in any mode, but meant mostly for
  `plan`: better to ask once than to plan on an assumption. Only registered in the TUI —
  `kiwi ask` has nobody to ask, so it does not show up there.
- Any tool exposed by a configured MCP server is added automatically, behind the same
  permission as the native tools.

## Memory and context

Kiwi keeps apart two things that are often confused:

- **The conversation** — large, specific to the session, and compacted automatically as
  it grows. The threshold comes from the real context window of the active profile's
  model (half the window), so a 200k model is not compacted on the same terms as a 32k
  one. When it triggers, old messages are replaced by a summary and recent ones stay as
  they are. `/compact` forces that compaction whenever you want, without waiting for the
  threshold. The status line shows `ctx N%` of the window in use, and changes color at
  50% and 80%.
- **Saved notes** — a handful of lines that outlive the session and travel with every
  prompt. Two scopes: `global` (about you, across all projects) and `project` (only about
  this directory). The model writes them with the `remember` tool, or you do from
  `/memory`. Each scope has a character cap; when it is exceeded the oldest notes drop
  off and kiwi says so instead of forgetting silently.

Notes live in `~/.config/kiwi/memory/` (`global.md` and `projects/<project>.md`), not
inside the repository: what kiwi notes down for itself should never show up in your
`git status`. For shared, versioned instructions there is already `KIWI.md`/`AGENTS.md`,
which kiwi reads and never writes.

### Attaching files with `@`

In the TUI, typing `@path/to/file` puts its contents into the message the model gets,
without it having to call `read_file`:

```
› why does @internal/agent/agent.go fail in the test in @internal/agent/agent_test.go?
```

The transcript still shows what you typed; the contents are added underneath. If a path
does not exist, kiwi says so instead of asking about a file that never arrived.

## Sessions

Every conversation is persisted on the fly to SQLite, per project:

```sh
kiwi --continue          # resume the most recent session in this directory
kiwi --resume <id>       # resume a specific session
kiwi session list        # list the saved sessions for this project
```

Inside the TUI, `/sessions` does the same without leaving the conversation.

## `kiwi ask`

For a single non-interactive turn, meant to be composed with pipes:

```sh
kiwi ask "what does this repo do?"
git diff | kiwi ask "review this patch" -
```

It is read-only by default; `--mode work` allows edits, and `--yolo` also runs commands
without asking.

## `kiwi serve` (Telegram)

Kiwi as a Telegram bot, to send it tasks from your phone. Every message is a task for
the agent, which works in the directory you start it in (or `--cwd`) and replies when it
is done. Meanwhile, a message keeps updating with what it is doing
(`● bash: go test ./...`), and long answers arrive split over several messages. It also
understands photos, voice notes, audio, video and documents. This is epic
[#6](https://github.com/oscar1223/kiwi/issues/6).

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Get your user ID from [@userinfobot](https://t.me/userinfobot).
3. Put them in Kiwi's `.env` (`~/.config/kiwi/.env`), never in a repo:

   ```sh
   KIWI_TELEGRAM_TOKEN=123456:ABC...
   KIWI_TELEGRAM_ALLOWED_USERS=11111111
   ```

4. Run `kiwi serve` from the project directory.

It runs in `work` mode: it edits files and runs commands without asking. Whatever the
mode would still ask about (an `rm -rf`, a `sudo`, a force push…) reaches you as a
message with **Allow** and **Deny** buttons; if you do not answer within 5 minutes
(`--approval-timeout`), it is denied. `--mode plan` makes it read-only. It continues the
directory's last conversation, so restarting it does not lose the thread; `/new` starts
a new one. It only does one task at a time: if you write while it is working, it tells
you it is busy.

### Scheduled tasks

`/cron` schedules tasks that Kiwi does on its own and sends to the chat, with the same
live message and the same approval buttons:

```
/cron add "0 9 * * 1-5" review the open PRs and summarize them
/cron add @daily summarize yesterday's commits
/cron list
/cron pause 3 · /cron resume 3 · /cron rm 3 · /cron run 3
```

Times are in the `KIWI_TZ` zone (default `Europe/Madrid`; a server is usually on UTC).
Every run is a new conversation, separate from the chat's, and if the bot was off when a
task was due it is not caught up: it waits for the next one.

It only answers the IDs on the list and only in private chats; anyone else gets no reply
at all. It uses long polling, so it opens no port. Treat the token like a password: with
it, anyone can read the messages you send the bot and reply in its name. And **whoever
gets into your Telegram account can run code on the machine where it runs**.

To keep it always on in EC2, with no open ports, there is the CDK stack in
[`deploy/aws`](deploy/aws/README.md).

## Configuration

Everything lives under `~/.config/kiwi/` (or `$XDG_CONFIG_HOME/kiwi/`):

- `kiwi.json` — model profiles and the active profile. Example profiles included:
  - `sonnet` and `opus` (Anthropic, `ANTHROPIC_API_KEY`), `gpt` (OpenAI, `OPENAI_API_KEY`),
    `local` (any server compatible with the OpenAI API, such as Ollama, at
    `http://localhost:11434/v1`).
  - Any provider that speaks OpenAI's chat-completions format works just by changing
    `base_url` and the key's env var — ready to go are `openrouter`
    (`OPENROUTER_API_KEY`), `groq` (`GROQ_API_KEY`), `deepseek` (`DEEPSEEK_API_KEY`),
    `mistral` (`MISTRAL_API_KEY`), `xai` (`XAI_API_KEY`), `cerebras`
    (`CEREBRAS_API_KEY`), `perplexity` (`PERPLEXITY_API_KEY`), `together`
    (`TOGETHER_API_KEY`), `fireworks` (`FIREWORKS_API_KEY`), `deepinfra`
    (`DEEPINFRA_API_KEY`), `moonshot` (`MOONSHOT_API_KEY`), `zhipu` (`ZHIPU_API_KEY`) and
    `gemini` (`GEMINI_API_KEY`, through Gemini's OpenAI-compatible endpoint). The models
    in these profiles are examples — adjust them with `/model` to whatever each provider
    offers at the time.
  - Three more profiles need cloud credentials, not just an API key — they are not
    zero-config like the ones above:
    - `azure` — Azure OpenAI. Needs `AZURE_RESOURCE_NAME` and `AZURE_API_KEY`; the
      profile's `model` is the *deployment* name, not the model itself.
    - `vertex` — Claude models on Google Vertex AI. Needs `GOOGLE_VERTEX_PROJECT` and
      `GOOGLE_VERTEX_LOCATION`, plus Application Default Credentials (ADC) — usually via
      `GOOGLE_APPLICATION_CREDENTIALS` pointing at a service account, or `gcloud auth
      application-default login`.
    - `bedrock` — Claude models on AWS Bedrock. Uses the standard AWS credential chain
      (`AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`/`AWS_REGION`, a shared profile, or
      `AWS_BEARER_TOKEN_BEDROCK`).
- `.env` — API keys and other variables, managed with `/config`.
- `mcp.json` — MCP servers, over stdio or remote (HTTP/SSE).
- `skills/` — Markdown skills the model loads on demand.
- `memory/` — durable notes: `global.md` and one per project in `projects/`.

### Reasoning

Each profile can set how much the model thinks before answering with `reasoning`:
`off`, `low`, `medium`, `high`, `xhigh` or `max`. Without the field, the provider decides
(many models reason by default: GLM 5.2 through OpenRouter does it at `high`).

```json
"glm": {
  "provider": "openai",
  "model": "z-ai/glm-5.2",
  "base_url": "https://openrouter.ai/api/v1",
  "api_key_env": "OPENROUTER_API_KEY",
  "reasoning": "off"
}
```

`/reasoning` changes it live and saves it in the active profile (`/reasoning high`, or
with no argument to pick from a list). The status bar shows the level, and while the
model reasons the spinner line shows `thinking…` with the latest thing it is thinking.

How it reaches each provider:

| Provider | What is sent |
| --- | --- |
| OpenRouter | `reasoning: {effort}` or `{enabled: false}`; its `reasoning_details` are sent back on tool turns |
| Z.ai (GLM direct) | `thinking: {type: enabled/disabled}` — on or off only |
| OpenAI and compatibles | `reasoning_effort` (`off` is `none`) |
| Anthropic, Bedrock, Vertex | adaptive thinking + `effort`; signed blocks go back on every turn. On models that cannot turn it off (Opus 5.5, Fable), `off` is the minimum effort; Haiku 4.5 and earlier use a token budget |

## Telemetry (optional)

Kiwi traces each session through OpenTelemetry if it finds configuration:

- `OTEL_EXPORTER_OTLP_ENDPOINT` / `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` — any standard
  OTel backend.
- Or `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` (and optionally `LANGFUSE_HOST` for a
  self-hosted instance) for [Langfuse](https://langfuse.com).

With neither, Kiwi traces nothing.

## Undo

Before every turn that can write, kiwi takes a **snapshot of the working tree**. `/undo`
restores it — the files *and* the conversation, because restoring the files while
leaving the model convinced of an edit that no longer exists is worse than not undoing
anything — and `/redo` applies it again. `/diff` shows what has piled up since the
session started.

Snapshots live in a git repository **of kiwi's own**, under
`~/.config/kiwi/checkpoints/`, pointed at the project with `GIT_WORK_TREE`. Your index,
your stash, your hooks and your history are never touched: kiwi never names your
`.git`. The project does need to be a git repository — not for the shadow repo, but
because `.gitignore` is what stops a snapshot from swallowing `node_modules` —; where it
is not, kiwi says so once at startup and carries on without a safety net.

This matters most in **work mode**, which applies edits and commands without asking: it
is the way back that replaces the confirmation that is no longer there.

## Diagnostics and LSP

After every `write_file`, `edit_file` or `multi_edit`, kiwi runs **the project's own
checker** over the touched package — `go vet` with `go.mod`, `tsc --noEmit` with
`tsconfig.json`, `ruff` with `pyproject.toml` — and adds whatever it finds to the same
observation the model reads. That way it learns it broke the build right there, not
three calls later when it has already built on top. It can be replaced with
`KIWI_DIAGNOSTICS_CMD` (a `{}` in the command marks where the path goes; if there is
none, it is appended at the end).

Two rules keep it out of noise territory: only problems **in the files that were just
edited** are reported, and **what was already said is not repeated**. A diagnostics
channel that shouts on every edit ends up ignored entirely.

If a **language server** is installed and the project is in that language — `gopls`,
`typescript-language-server`, `pyright`, `rust-analyzer` — kiwi also registers the `lsp`
tool with three operations: `definition`, `references` and `diagnostics`. The server
starts the first time it is asked, not when kiwi opens, because indexing a whole module
is expensive and most sessions do not need it.

It earns its place on the one question `grep` cannot answer honestly: **which of these
matches is the same symbol**. `grep` finds a name; the server resolves it. Configure it
with `KIWI_LSP` (`.go=gopls;.rb=solargraph --stdio`), and if there is no server the tool
is not registered — a tool the model sees but that can only fail is worse than none.

## Skills

kiwi ships **seven built-in skills**, embedded in the binary and seeded into
`~/.config/kiwi/skills/` the first time it starts — without that, a fresh `go install`
would leave the directory empty and "built-in" would mean nothing.

| Skill | What for |
| --- | --- |
| `plan` | writes a `PLAN-*.md`: context, decisions with their reasoning, parts anchored to `file.go:line`, order and verification |
| `commit` | a message from the staged diff, checking first **which repository you are in** |
| `code-review` | reviews the change: correctness first, then reuse and simplification |
| `init` | writes `KIWI.md` after reading the project |
| `security-review` | injection, secrets, authorization, SSRF, path traversal, insecure defaults |
| `test` | finds *this* repo's real commands, runs them and reads the failures |
| `reach` | researches on the internet: search, JS sites, YouTube (`yt-dlp`), GitHub (`gh`), RSS, Reddit |

### Reaching the internet

The `reach` skill and `kiwi doctor` are inspired by
[Agent Reach](https://github.com/Panniantong/agent-reach): the agent uses ordinary
command-line tools for each platform, and `kiwi doctor` (or `/doctor` inside the TUI)
tells you which ones work on your machine and how to enable the missing ones. It only
looks; it installs nothing.

```text
$ kiwi doctor
✓ Web pages    web_fetch; reader: true for JavaScript pages (Jina Reader)
✓ Web search   web_search via DuckDuckGo (no key) — set EXA_API_KEY or JINA_API_KEY with /config for better results
✗ YouTube      brew install yt-dlp
✓ GitHub       gh (signed in)
…
```

**What you have touched is never overwritten.** A skill is only refreshed if what is on
disk is byte for byte the version kiwi last wrote, which is exactly the case where
nothing can be lost. And if you delete one, it does not come back: deleting it is a
decision too.

### Skills as commands

A skill with `user-invocable: true` in its frontmatter **becomes a slash command**: it
shows up in autocomplete and in `/help`, and `/plan whatever` starts a turn with its
instructions and your text appended. kiwi's native commands always win — a skill called
`clear` cannot hijack `/clear`.

## Keyboard

| Key | What it does |
| --- | --- |
| `↑` `↓` | recalls an earlier prompt (with the input empty or half-typed; a multi-line draft keeps its arrows) |
| `ctrl+r` | searches earlier prompts |
| `@` | fuzzy file picker — `@tumod` finds `internal/tui/model.go` |
| `!command` | runs in the shell without spending a turn |
| mouse drag | selects conversation text and copies it to the clipboard on release |
| `ctrl+t` | shows or hides the model's task list |
| `ctrl+e` | writes the prompt in `$EDITOR` |
| `?` | shortcuts panel, with the input empty |
| `shift+tab` | cycles the mode |
| `esc` | drops the queue, or cancels the turn |

**History is per working directory** (`~/.local/share/kiwi/history/`): one project's
prompts are noise in another.

**Typing during a turn queues** instead of getting lost. Everything queued is sent
together when the turn ends, as a single message: three things that occur to you while
watching the same work are one piece of feedback, not three turns. `esc` drops it. If
the turn fails, the queue is **not** sent — it was written about work that never
happened, and that decision is yours.

When a turn longer than 20 seconds finishes, **the terminal bell rings**. With
unattended work mode that stops being decoration: it is how you know you can come back.
`KIWI_NOTIFY=desktop` adds a desktop notification; `KIWI_NOTIFY=off` silences it
entirely.

`!` respects the mode: in plan mode a command that writes is rejected, just as the tool
would do. A read-only mode you can escape by typing `!` is not a promise worth making.

## Development

```sh
make test               # go vet + go test, same as CI
make dev                # installs ~/go/bin/kiwi-dev (isolated data, DEV badge in the TUI)
```

## License

MIT — see [LICENSE](LICENSE).
