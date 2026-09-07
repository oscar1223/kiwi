---
name: init
description: Write or refresh the project instructions file (KIWI.md / AGENTS.md) by reading the project first. Short, verified, and free of anything the file layout already says.
user-invocable: true
---
# Writing the project instructions

This file is loaded into **every** prompt for this project. That is the whole
constraint: every line costs tokens on every request, forever. A hundred useful
lines is a good file; four hundred padded ones make every future turn worse.

## Read first

Do not write a word until you have looked. Use `glob`, `grep` and `ls`:

- The build files — `go.mod`, `package.json`, `pyproject.toml`, `Makefile`,
  `Cargo.toml`. They name the language, the version and the real commands.
- The entry points, and the top two or three directories under them.
- The test files: how tests are named, how they are laid out, what they use.
- Any existing `README`, `CONTRIBUTING`, or CI workflow. The commands in CI are
  the ones that actually have to pass.
- If `KIWI.md` or `AGENTS.md` already exists, read it and improve it in place
  rather than starting over. Keep what is still true.

## What goes in

- **What this project is**, in one or two sentences.
- **Layout**: which directory holds what, and why, where that is not obvious
  from the name. Skip the parts a listing already tells you.
- **Commands**: build, test, run, lint — exactly as they are invoked here,
  copied from a file you read, not from habit.
- **Conventions a newcomer would get wrong**: naming, error handling, how tests
  are written, anything the code does consistently that is not the language's
  default. These are the highest-value lines in the file.
- **Anything genuinely surprising**: a directory that is not what it sounds
  like, a generated file that must not be edited, a test that needs a service
  running, a repository that is really several repositories.

## What stays out

- Generic advice. "Write clear code" helps nobody and costs every request.
- Anything you did not verify. An invented command is worse than a missing one:
  it will be run.
- A restatement of the directory tree.
- History, roadmaps, changelogs. They go stale and nobody updates them.

## Then

Write it to `KIWI.md` at the top of the working directory — unless an
`AGENTS.md` is already there and in use, in which case update that one, since
it is the convention shared with other tools.

Say what you wrote and what you deliberately left out.
