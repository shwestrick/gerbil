# Filling sorries with `--fill-sorry`

`--fill-sorry` turns "prove these `sorry`s" into a complete gerbil task
without writing a prompt, a termination check, or file-freezing rules by
hand. Point it at one or more positions and gerbil generates all three,
then runs a [ralph loop](ralph.md) until the goal check passes:

```console
$ gerbil run --fill-sorry MyProj/Basic.lean:42
$ gerbil run --fill-sorry MyProj/A.lean:42:5,MyProj/B.lean:107
$ gerbil run --fill-sorry MyNs.foo               # by declaration name
$ gerbil run --fill-sorry MyNs.foo,MyProj/B.lean:107
$ gerbil run --fill-sorry task.toml
```

A sorry is designated by position -- `FILE:LINE[:COL]`, project-root-
relative, 1-indexed, the same format the [big-small mode](big-small.md)
`zoom_in` tool uses -- or by the name of the top-level declaration
holding it, optionally namespace-qualified (`foo`, `MyNs.foo`; the name
is located across the tracked `.lean` sources, and an ambiguous or
unknown name is a preflight error naming the candidates). Entries
containing `:`, `/`, or a `.lean` suffix are parsed as positions;
anything else is a name. An argument ending in `.toml` is read as a task
spec instead (below).

Prefer not to write the spec by hand? **`gerbil new-fill`** opens your
editor (`$VISUAL`, then `$EDITOR`, then `vi`) on a commented template of
every spec key, validates what you save, loops back into the editor on
problems, then shows a summary of the task -- resolved declarations,
off-limits paths, axiom policy, session budget -- and starts the run on
your confirmation, exactly as `gerbil run --fill-sorry <spec>` would. It
accepts all of `gerbil run`'s options, and the spec file is kept in
`.gerbil/tasks/` either way, so a declined task can be started later by
path.

The mode runs directly on the current repo, like any other gerbil session:
preflight requires a clean tree, the session starts from `HEAD`, and the
output is an ordinary patch in `.gerbil/patches/` that `gerbil commit`
applies.

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
`.gerbil/plans/<name>.md` -- gerbil's namespace in the repo, next to where
the folded session logs land. The name is deterministic from the sorry
list (`fill-<slug>-<hash>.md`; `plan` in the spec picks another bare
filename). gerbil seeds it into the container at exactly that path before
the first session, so the agent never decides where the plan lives -- it
only reads it at session start and appends what it did, what failed, and
what to try next at session end. (A continuing task keeps its existing
copy; the seed is written only when the file is absent.) Although
`.gerbil/` is conventionally gitignored, the plan still ships in every
patch: each session's commit force-includes it, exactly the way the
session log is force-added. Its updates are work product like the proofs themselves:
they **ship inside each patch** and reach your repo only when you
`gerbil commit` -- skip a patch and you skip its plan updates too. Within a ralph chain the memory carries forward
automatically (each session builds on the last commit); across separate
`gerbil run` invocations it carries through the patch chain, so commit the
earlier patches first -- preflight warns when uncommitted `.gerbil/`
patches would leave the task blind to its own history. The wip snapshot
covers the plan too, so even a mid-session crash keeps its notes. Delete
the file when the task is done.

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

The agent can run this exact check itself, via a **`check_goal` tool**
that gerbil adds whenever a termination check is installed: it returns
the check's verdict (`CHECK PASSED` / `CHECK NOT PASSED (exit N)`) and
its output, so an agent that believes it is finished can see precisely
which condition the harness still considers unmet instead of watching
the loop restart blind. The prompt tells it the check is the definition
of done and to use it at natural checkpoints.

The check is scoped to **exactly the designated sorries**, not their whole
files: other sorries in the same module are not part of the goal and never
block completion, so the loop stops the moment the listed ones are done.
This is sound because `collectAxioms` is transitive -- a designated
theorem whose proof routes through a sorried helper still shows `sorryAx`
and fails.

Positions drift as the agent edits, so the check works by *name*: at
preflight gerbil resolves each position to its enclosing declaration (a
syntactic scan tracking `namespace` blocks), and the check re-resolves
that name against the live Lean environment across **every module of the
declaration's library** -- enumerated from the tracked `.lean` files as
of each check, so a declaration legitimately moved to another file (even
one created mid-task) is found where it landed. Matching is exact on the
display name first, unique-suffix as fallback, so `private` name mangling
and namespace subtleties still land. A designated declaration that is
renamed or deleted is a **hard failure**, never a pass.

**Supported sorry sites**: the bodies of top-level `theorem` / `lemma` /
`def` / `abbrev` / named `instance` declarations -- anywhere inside them
(deep in a tactic proof, in a `where` clause, in a `mutual` block), since
the axiom sweep covers the whole elaborated term. Sorries in `example`s
(never enter the environment), anonymous `instance`s, and
`structure`/`inductive`/`class`/`opaque` declarations (their sorries hide
in auto-generated constants the axiom check cannot see -- a field-default
sorry would go undetected) are refused at preflight: move the sorry into
a named top-level declaration, or point preflight at one explicitly:

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

# Required. A position string is auto-resolved to its enclosing
# declaration; a name string is located across the tracked sources; the
# table form names the declaration explicitly (needed for anonymous
# instances and other layouts the source scan cannot name).
sorries = [
  "MyProj/A.lean:42",
  "MyNs.foo",
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

# plan = "my-task.md"     # plan-file name override (lives in .gerbil/plans/)
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
   them -- and the moment a turn's changes touch a frozen path anyway, a
   warning is delivered straight into the conversation (piggybacking on
   the per-turn wip snapshot, so it costs nothing extra): the warning
   names the paths and tells the agent to restore them byte for byte.
   It fires once per newly-violated path, not once per turn, and fixing
   the files is the agent's job.
2. **The goal check** pins every off-limits path to the starting commit,
   so the ralph loop can never terminate "done" with a frozen file
   changed.
3. **The patch gate**: after every session, gerbil checks the session's
   patch against the off-limits list and refuses to continue if it
   touches one -- the run stops with an error, and the offending patch is
   kept in `.gerbil/patches/` for inspection (do not `gerbil commit` it
   as-is).
   The patch is gerbil's output contract, so this is the actual
   guarantee.

Preflight also rules out the one plan-file state that would silently
break the memory chain: an existing-but-untracked host copy (invisible to
the upload, and colliding with `git am` later). A tracked plan file is
simply a continuing task.

## Resuming

A crashed `--fill-sorry` run resumes like any other: `gerbil resume LOG`.
The spec's enforcement data and the generated check ride in the session
log, the plan file is rebuilt with the rest of the tree (ancestor patches
plus the crashed session's wip snapshot), and the patch gate stays armed.
No flags needed.
