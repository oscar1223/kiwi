---
name: test
description: Build the project and run its tests, then read the failures and fix them. Learns this repository's real commands once instead of guessing them every time.
user-invocable: true
---
# Building and testing

The value here is not knowing that Go projects run `go test`. It is finding out
what **this** repository actually does, and saying so, so nobody has to work it
out again.

## Find the real commands

In order of authority:

1. `KIWI.md` or `AGENTS.md`, if either exists. They were written for this.
2. The CI workflow — `.github/workflows/*.yml`. Whatever CI runs is what has to
   pass, by definition.
3. `Makefile`, `Taskfile`, `justfile`, or the `scripts` block of `package.json`.
4. Only then the language default: `go build ./... && go test ./...`,
   `npm test`, `pytest`, `cargo test`.

If the project has a lint or vet step, it is part of the job. `go vet` catches
things the compiler does not.

## Run it

Run the build first. A failing build makes every test failure noise, and reading
twenty test failures caused by one missing symbol wastes the turn.

Then the tests. If the suite is slow, run the package you touched first for a
fast answer, and the whole suite before reporting anything as done — a change
that passes its own package and breaks another is the most common way to be
wrong here.

## Read the failures properly

- Read the actual assertion, not just the test name.
- Ask whether the test is wrong before assuming the code is. A test that encodes
  the old behaviour will fail on a correct change, and the fix is the test — but
  say clearly that that is what you concluded and why, because "I changed the
  test until it passed" is the failure mode this warns against.
- One root cause usually produces several failures. Find it before fixing
  anything, or you will fix the same thing three times.
- A flaky failure is a finding, not an inconvenience. Say so rather than
  re-running until it passes.

## Report

What you ran, what passed, what failed and why. If anything is still failing,
say so plainly with the relevant output — a green summary over a red suite is
the one outcome that makes this skill worse than useless.
