---
name: security-review
description: Review the current changes for security problems — injection, secrets, authentication and authorization, unsafe deserialization, SSRF, path traversal, and unsafe defaults. Reports exploitable findings, not a checklist.
user-invocable: true
---
# Security review

Review the change, not the whole codebase: `git diff`, or `git diff main...HEAD`
for a branch. Read enough of the surrounding code to know whether an input is
actually attacker-controlled — that question decides almost every finding here,
and a diff alone cannot answer it.

## Where the real problems are

Trace untrusted input from where it enters to where it is used. Everything below
matters only if something outside the trust boundary can reach it.

- **Injection.** A string built into SQL, a shell command, an HTML page, a
  template, a log line that something else parses. Look for concatenation and
  interpolation where a parameterised API exists.
- **Path traversal.** A user-supplied path joined onto a base directory without
  resolving and re-checking that the result is still inside it. `..` and
  absolute paths both.
- **Server-side request forgery.** A fetched URL that is not checked, or checked
  before the connection rather than at it — a redirect to a private address
  defeats a check done on the original URL. Loopback, private ranges,
  link-local, and `169.254.169.254` in particular: in a cloud that address
  hands out credentials.
- **Secrets.** Keys, tokens or passwords in source, in test fixtures, in error
  messages, in logs, in a URL query string. Also: a secret that is read
  correctly and then printed.
- **Authentication and authorization.** A new endpoint or command with no check
  at all; a check that confirms *who* without confirming *what they may do*; an
  identifier taken from the request instead of the session.
- **Unsafe deserialization** and any parser handed untrusted bytes with no size
  limit.
- **Unsafe defaults.** Verification off, TLS skipped, a permissive CORS origin,
  a world-writable file, a debug flag that survives into production. Defaults
  are what almost everyone runs.
- **Missing bounds.** Anything unbounded that an attacker can grow: request
  size, result count, recursion, retries, allocation.

## Judgement

Say whether each finding is actually reachable, and by whom. A theoretical issue
in code only a developer can invoke is worth a sentence; the same issue behind
an unauthenticated endpoint is worth stopping for.

If a control is missing on purpose because something upstream provides it, name
what provides it. If nothing does, that is the finding.

## Reporting

Most severe first. Each one: `file:line`, what an attacker can do, and the
smallest change that closes it. Describe the class of problem and the fix —
there is no need to write a working exploit to make the point.

If nothing is wrong, say that plainly and say what you checked. A security
review that always finds something is a security review nobody believes.
