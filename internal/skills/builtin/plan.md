---
name: plan
description: Write an implementation plan as a PLAN-*.md document — context, decisions with their reasoning, parts anchored to real file:line, order, and verification. Use before a change big enough that getting the approach wrong would be expensive.
user-invocable: true
---
# Writing a plan

A plan is not a task list. A task list says what to do; a plan says **what is
true, what was decided, and why** — so that someone reading it in a month,
including you, can tell a deliberate choice from an accident.

## Before writing anything

Go and read the code. Every claim in the plan has to be checkable, which means
every claim needs an anchor: `internal/tools/tools.go:100`, not "the registry".
A plan built on an assumption you could have verified is worth less than no
plan, because it will be trusted.

Use `grep`, `glob` and `ls` to find things, and the `lsp` tool when the question
is which of several same-named symbols is the real one. Read the files that
matter in full. If a design decision or an ambiguous requirement would
otherwise force you to guess, ask — everything you need in one go, not one
question at a time.

## The document

Write it to `PLAN-<topic>.md` at the top of the repository. Note in a blockquote
under the title that it is a local, gitignored note, with the date.

Then, in this order:

### Contexto

What is true today, with anchors. If the plan came out of comparing against
something else, put the comparison in a table — one row per capability, one
column per system, and a clearly marked column for this project.

Say what shape the gap has, not just its size. "Kiwi knows how to configure
itself and does not know how to look back" is worth more than a list of eight
missing commands, because it predicts the ninth.

If something recently built is what makes this work urgent, say so and say why.
Causes are the part people forget, and they are what stops the plan being
re-litigated later.

### Decisiones tomadas

Every decision that closes off an alternative, with the reason. Especially:

- Decisions that **reverse an earlier deliberate choice**. Say that it was
  deliberate, say what changed.
- Decisions with a **real cost**. State the cost plainly in the same sentence.
  A trade recorded as a pure win is a trade nobody can revisit honestly.
- What was **deferred rather than rejected**, so it is not lost.
- What was **rejected**, so it is not proposed again next quarter.

### Parte A, B, C…

One part per coherent piece of work. Each part explains what it does, where it
touches the code, and the reasoning behind any non-obvious choice. Prose, not
bullet soup: the reasoning is the content, and bullets flatten it.

Where a part has real subtlety, give it its own heading and spend the words.
Where it is routine, one paragraph is enough — length should track difficulty,
not importance.

Name the caps and limits explicitly. Anything unbounded — results, bytes, depth,
retries, history — is a bug that arrives three months in, not on day one.

### Orden

Numbered, with the reason each item is where it is. Dependencies, cheapest
valuable thing first, and anything that pays down debt the last change created.
When two orders are defensible, say which alternative you rejected.

### Verificación

The commands (`go build ./... && go vet ./... && go test ./...` or the
project's equivalent), then the specific cases worth pinning — especially the
one that is easy to forget, and any promise the design rests on. Finish with the
manual check that would show whether the goal was actually met, stated as an
acceptance criterion rather than a hope.

## Tone

Plain and direct. No filler, no "robust", no "seamless". Prefer the concrete
noun to the abstraction. It is fine — good, even — to say that something is the
cheapest fix with the biggest daily payoff, or that something else is expensive
and only worth it later. A plan that will not rank its own contents leaves the
ranking to whoever reads it last.

Write the whole plan before showing it. Then say, in two or three sentences,
what you would do first and why.
