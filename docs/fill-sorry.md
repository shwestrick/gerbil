# Filling sorries with `--fill-sorry`

`--fill-sorry` turns "prove these `sorry`s" into a complete gerbil task
without writing a prompt, a termination check, or file-freezing rules by
hand. Point it at one or more positions and gerbil generates all three,
then runs a [ralph loop](ralph.md) until the goal check passes:

```console
$ gerbil run --fill-sorry MyProj/Basic.lean:42
$ gerbil run --fill-sorry MyProj/A.lean:42:5,MyProj/B.lean:107
$ gerbil run --fill-sorry task.toml
```

Positions are `FILE:LINE[:COL]` -- project-root-relative, 1-indexed, the
same format the [big-small mode](big-small.md) `zoom_in` tool uses. An
argument ending in `.toml` is read as a task spec instead (below).

The mode runs directly on the current repo, like any other gerbil session:
preflight requires a clean tree, the session starts from `HEAD`, and the
output is an ordinary patch in `.gerbil/` that `gerbil commit` applies.

Defaults with a bare position list: the whole repository is editable, the
finished proofs may use Lean's standard axioms (`propext`,
`Classical.choice`, `Quot.sound`), there is no computability restriction,
and the loop runs `--ralph 10` (an explicit `--ralph N` overrides).

## What gets generated

**The task prompt.** States the sorries (with each line's current text and
its enclosing declaration), the editable/off-limits split, the exact
conditions the goal check verifies, and the rules -- fill *only* the
designated sorries (others are to be left alone, and routing a proof
through one still fails the check), don't change a designated statement or
rename its declaration, and an agent that concludes a statement is
unprovable is told to argue that in the plan file and stop, not to edit
the question. With
`--prompt FILE`, the file's text is spliced in as approach notes (it does
*not* replace the generated prompt; the spec's `approach` key does the
same thing).

**The plan file.** Cross-session memory, at
`<project>/.gerbil/plans/<name>.md` -- the generated name is deterministic
from the sorry list, so re-running the same task later finds its own
notes. The agent is required to read it at session start and append what
it did, what failed, and what to try next at session end. gerbil itself
carries the file into and out of the container: it is untracked on both
sides and can never appear in a patch (see enforcement below). A
mid-session crash loses only that session's plan edits.

**The goal check.** A `--ralph_done`-style script (so `--ralph_done` is
rejected alongside `--fill-sorry`) that exits 0 only when:

1. every off-limits path is byte-identical to the commit the task started
   from (`.gerbil/` excepted);
2. `lake build` succeeds, as does building each target module by name;
3. each **designated declaration** -- the theorem/def each listed sorry
   lives in -- depends on no `sorryAx` (a surviving `sorry` in any
   disguise) and on no axiom beyond the allowed list (this is how
   `native_decide` is caught, too: its three axioms are simply not allowed
   unless the spec lists them);
4. the spec's computability restrictions hold for the designated
   declarations and everything they use within their modules (no
   `noncomputable`, no `partial`, per `forbid`).

The check is scoped to **exactly the designated sorries**, not their whole
files: other sorries in the same module are not part of the goal and never
block completion, so the loop stops the moment the listed ones are done.
This is sound because `collectAxioms` is transitive -- a designated
theorem whose proof routes through a sorried helper still shows `sorryAx`
and fails.

Positions drift as the agent edits, so the check works by *name*: at
preflight gerbil resolves each position to its enclosing declaration (a
syntactic scan tracking `namespace` blocks), and the check re-resolves
that name against the live Lean environment -- exact match in the module,
or a unique suffix match, so `private` name mangling and namespace
subtleties still land. A designated declaration that is renamed or deleted
is a **hard failure**, never a pass. Where the scan can't name the
declaration (an `example`, an anonymous `instance`), preflight stops and
asks you to name it in the spec:

```toml
sorries = [
  "MyProj/A.lean:42",                                  # auto-resolved
  { pos = "MyProj/B.lean:107:9", decl = "Ns.tricky" }, # named explicitly
]
```

## The task spec

```toml
# task.toml -- read by `gerbil run --fill-sorry task.toml`.
# Paths are project-root-relative; positions are 1-indexed.

# Required. A string is auto-resolved to its enclosing declaration; the
# table form names the declaration explicitly (needed for anonymous
# instances and other layouts the source scan cannot name).
sorries = [
  "MyProj/A.lean:42",
  { pos = "MyProj/B.lean:107:9", decl = "Ns.tricky" },
]

# Paths/globs that must not change (default: none -- full repo access).
# A bare path names a file or a whole directory; globs are fnmatch-style
# and `*` crosses `/`.
off_limits = ["MyProj/Spec.lean", "Contracts/", "lakefile.toml"]

# Axioms the finished proofs may rest on (default: the standard three).
# To permit native_decide, add its axioms: Lean.ofReduceBool,
# Lean.ofReduceNat, Lean.trustCompiler. sorryAx is never allowed.
axioms = ["propext", "Classical.choice", "Quot.sound"]

# Any of "noncomputable", "partial", "native_decide" (default: none).
forbid = ["noncomputable", "partial"]

# Approach notes spliced into the generated prompt. At most one of these;
# `gerbil run --prompt FILE` overrides both.
approach = "Try induction on the program; Wf carries the bound."
# approach_file = "notes/approach.md"

# plan = "my-task.md"     # plan-file name override (bare *.md filename)
# check_timeout = 1800    # seconds the goal check may run (default 1800)
# ralph = 10              # default session budget (CLI --ralph overrides)
```

Unknown keys are errors, not warnings -- a typo'd `off_limit` would
otherwise be an unenforced safety constraint. Preflight also validates
every position against the working tree (file tracked, line exists,
line actually contains `sorry` -- the last is a warning, since a `by`
block's sorry may sit on a later line) and resolves each position's
enclosing declaration, failing with a spec-entry hint when it cannot.

## How off-limits is enforced

Three layers, each catching what the previous one misses:

1. **The prompt** states the rules, so sessions aren't wasted violating
   them.
2. **The goal check** pins every off-limits path to the starting commit,
   so the ralph loop can never terminate "done" with a frozen file
   changed.
3. **The patch gate**: after every session, gerbil checks the session's
   patch against the off-limits list and refuses to continue if it
   touches one -- the run stops with an error, and the offending patch is
   kept in `.gerbil/` for inspection (do not `gerbil commit` it as-is).
   The patch is gerbil's output contract, so this is the actual
   guarantee.

The plan file gets the same enforcement treatment in reverse: it is reset
out of the commit index before every patch and snapshot (exactly like
submodule state), so no patch can ever carry it, even if the agent
`git add -f`s it.

## Resuming

A crashed `--fill-sorry` run resumes like any other: `gerbil resume LOG`.
The spec's enforcement data and the generated check ride in the session
log, the plan file is re-seeded from the host copy, and the patch gate
stays armed. No flags needed.
