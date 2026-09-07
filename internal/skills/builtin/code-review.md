---
name: code-review
description: Review the current changes for correctness bugs and for reuse, simplification and efficiency. Reports findings with file:line and a concrete failure case; does not apply fixes.
user-invocable: true
---
# Reviewing a change

## Scope

Review what changed, not the whole codebase. `git diff` for uncommitted work,
`git diff main...HEAD` for a branch. Read enough of the surrounding code to
judge the change — a diff on its own hides almost everything that matters, and
most real bugs live in the interaction between new code and old.

## What to look for, in order

**Correctness first.** A bug that fires is worth more than ten style notes:

- The case the author did not have in mind: empty, nil, zero, one, the boundary,
  the second call, the concurrent call, the cancelled context.
- Errors swallowed, or returned but not acted on.
- State mutated from somewhere that does not own it.
- Anything unbounded — a loop, a buffer, a result set, a retry, a cache with no
  eviction. These pass every test and fail in production.
- Resources not released on the failing path. The happy path is usually fine.
- A comment or a name that no longer matches what the code does. It will
  mislead the next reader, and it costs nothing to fix now.

**Then reuse and simplification.** Code that duplicates something the project
already has, an abstraction with one caller, a special case that the general
path already covers, three parameters that are always passed together.

**Then efficiency**, but only where it is real: work repeated inside a loop that
could be hoisted, a file read twice, an O(n²) over something that grows.

## What not to say

- Anything the formatter or linter already enforces.
- Preferences dressed as findings. If it would be equally fine either way, it is
  not a finding.
- Praise padding. The author asked for a review.
- A finding you cannot demonstrate. If you cannot name the input that breaks it,
  say you are unsure and why, or leave it out.

## Reporting

Most severe first. Each finding gets: `file.go:line`, one sentence on what is
wrong, and a **concrete failure case** — the inputs or the state, and what
happens as a result. That last part is what separates a review from a hunch, and
it is what lets the author disagree with you on the merits.

Then, in a line or two: is this change safe to merge, and what would you fix
first?

Report findings. Do not apply them unless asked.
