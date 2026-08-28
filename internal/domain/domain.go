// Package domain is the opt-in subject-matter toggle: a small, dependency-free
// enum that says what kind of work Kiwi is being pointed at.
//
// It is deliberately orthogonal to everything else. A domain is not a
// permission stance (see internal/permission), not a model choice (see
// config.Profile), and not a tool: all it does is decide which system-prompt
// fragment and which bundled skills get folded in. The zero value is General,
// which contributes nothing at all — so a Kiwi that never opts in behaves
// exactly as it did before this package existed.
package domain

// Domain is the active subject-matter mode.
type Domain string

const (
	// General is the default: no domain-specific instructions or skills.
	General Domain = "general"
	// Science tunes Kiwi for numerical and scientific work — exact I/O
	// contracts, declared tolerances, reproducible runs.
	Science Domain = "science"
)

// Order is the set of domains a picker offers, in presentation order.
var Order = []Domain{General, Science}

// Valid reports whether d names a real domain. The empty string counts: it is
// how "the user never chose one" reaches us, from both an unset config field
// and an unpassed --domain flag, and it means General everywhere.
func (d Domain) Valid() bool {
	switch d {
	case "", General, Science:
		return true
	}
	return false
}

// Label is the human-facing name, used by the TUI picker.
func (d Domain) Label() string {
	switch d {
	case Science:
		return "Science"
	case "", General:
		return "General"
	default:
		return string(d)
	}
}

// Instructions is the system-prompt fragment for this domain, mirroring
// permission.Mode.Instructions: a pure function of the enum, empty when there
// is nothing to say.
//
// The Science text is deliberately about *rigor*, not about any one field.
// What separates a passing scientific task from a failing one is almost never
// domain knowledge the model lacks — it is following the stated I/O contract
// exactly and checking the numbers before declaring victory. Per-field
// expertise, if it is ever wanted, belongs in skills the user can edit, not
// in a constant compiled into the binary.
func (d Domain) Instructions() string {
	switch d {
	case Science:
		return `[DOMAIN: Science]
You are working on scientific and numerical tasks, where being approximately
right in the wrong format scores zero. Hold yourself to these:

- Follow the stated interface exactly. Flag names, argument order, JSON keys
  and nesting, CSV columns and header spelling, the output file's path and
  name, stdout vs stderr, the exit code — all of it is part of the spec, not a
  detail to improvise. Re-read the spec before you write the final version.
- Respect the declared tolerance. If one is given, compare against it
  (math.isclose/numpy.isclose with the stated atol/rtol) instead of testing
  exact equality; if none is given, say what precision you assumed.
- Check units and conventions before trusting a number: SI vs imperial,
  radians vs degrees, 0- vs 1-based indexing, time zones and epochs. A
  constant-factor error is the most common way a correct method gives a wrong
  answer.
- Make runs reproducible: seed every random number generator, sort anything
  whose order comes from a dict, set, or directory listing, and do not depend
  on library defaults that vary across versions.
- Verify before you finish. Run the code on whatever sample input exists, and
  construct a small case yourself if none does. Compare the actual output to
  the expected one field by field. Confirm the output landed at the exact path
  the spec named. Never report a task as done on the strength of code that
  merely ran without raising.

State your assumptions when the spec is genuinely silent; do not quietly pick
one and present the result as if it were unambiguous.`

	default:
		return ""
	}
}
