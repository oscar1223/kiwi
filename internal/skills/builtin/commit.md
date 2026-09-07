---
name: commit
description: Write a commit message from the staged changes and commit them. Checks which repository it is in first, reads the diff, and groups unrelated work into separate commits rather than one.
user-invocable: true
---
# Committing

## First: which repository?

Run `git rev-parse --show-toplevel` and say which repository you are in before
touching anything.

This matters more than it sounds. A working directory can be a parent folder
holding several independent repositories, each with its own `.git` and its own
remote — the folder looks like a project and is not one. Committing from the
parent, or from the wrong subdirectory, puts the change in a repository nobody
was looking at.

If the current directory is not inside a repository, stop and say so. Do not
run `git init`.

## Read the change before describing it

`git status --porcelain` and `git diff --staged`. If nothing is staged, look at
`git diff` and say what you would stage — do not stage everything on your own
initiative, since unrelated work in progress is the normal state of a working
tree.

Read the actual diff, not the file names. A message written from a file list
describes which files changed, which the reader can already see; the message has
to say what changed about the behaviour.

## More than one commit, when there is more than one change

If the staged work covers two unrelated things, say so and propose splitting it.
One commit per idea is what makes a history worth bisecting or reverting. It is
better to ask than to bundle.

## The message

- A subject line in the imperative, under ~70 characters, no trailing period.
  "Add the checkpoint store", not "Added" or "Adding".
- Then a blank line and a body, whenever the change is not self-evident. The
  body says **why**, not what: the diff already says what. The reason a choice
  was made, the alternative rejected, the constraint that forced it.
- No body at all is right for a genuinely trivial change. A body restating the
  subject is worse than none.
- Match the surrounding history. Run `git log --oneline -20` and follow whatever
  convention is already there — including its language. A repository whose
  history is in Spanish gets a message in Spanish.

Do not add trailers, co-author lines, or tool attribution unless the repository's
own history already has them.

## Then

Show the message and commit. Do not push, do not create a branch, do not amend
anything already pushed — none of those were asked for, and each is harder to
undo than a commit.
