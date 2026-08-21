"""The --fill-sorry mode: an auto-generated sorry-filling task.

`gerbil run --fill-sorry FILE:LINE[:COL],...` (or `--fill-sorry spec.toml`)
turns "fill these sorries" into a full gerbil task without the user writing a
prompt, a --ralph_done check, or file-freezing conventions by hand. This module
owns everything specific to the mode:

  - parsing the CLI position list and the TOML task spec;
  - preflight validation against the host repo;
  - the generated task prompt (what the agent is told);
  - the generated check-goal script (how "done" is decided, run in-container
    as the --ralph_done check);
  - the off-limits glob matching shared by the check script and the host-side
    patch gate;
  - the plan-file naming that gives a task its cross-session memory.

Everything here is pure except `validate` (which reads the project's files and
`git ls-files` to fail before a container ever boots). Nothing here talks to a
sandbox or a provider; cli.py wires the results in.

Enforcement is layered, because each layer catches what the previous one
misses: the prompt states the rules (an agent that knows the check will fail
doesn't waste sessions violating it), the check script pins every off-limits
path to the base commit (gating the ralph loop), and cli.py's patch gate
refuses to let a finished session's patch touch an off-limits path at all
(the patch is the output contract, so this is the guarantee).

Position format is FILE:LINE[:COL], project-root-relative, 1-indexed -- the
same shape prompts.zoom_task_prompt and the zoom_in tool already use, so the
model sees one consistent convention everywhere.
"""

from __future__ import annotations

import fnmatch
import hashlib
import posixpath
import re
import shlex
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

# The axioms a finished Lean proof may always rest on. `sorryAx` is never
# allowed -- rejecting it is how the check detects a surviving sorry in any
# disguise. native_decide's axioms are deliberately NOT here: allowing
# native_decide is opting into trusting the compiler, which a spec must do
# explicitly by listing all three.
STANDARD_AXIOMS = ["propext", "Classical.choice", "Quot.sound"]
NATIVE_DECIDE_AXIOMS = ["Lean.ofReduceBool", "Lean.ofReduceNat", "Lean.trustCompiler"]

# What the `forbid` spec key may name.
FORBIDDABLE = ("noncomputable", "partial", "native_decide")

# --fill-sorry defaults to a ralph loop: the whole point of the generated
# check is to gate one. 10 sessions is enough to make real progress on a hard
# sorry while keeping the default spend bounded.
DEFAULT_RALPH = 10

# Timeout for the generated check, in seconds. run_script's 300s default is
# calibrated for user-supplied one-liners; this check runs a full `lake build`.
DEFAULT_CHECK_TIMEOUT = 1800


@dataclass(frozen=True)
class SorryPos:
    """One sorry's location: project-root-relative posix path, 1-indexed line,
    optional 1-indexed column. str() renders the FILE:LINE[:COL] form used
    everywhere the position is shown to the model or the user."""

    file: str
    line: int
    column: int | None = None

    def __str__(self) -> str:
        pos = f"{self.file}:{self.line}"
        if self.column is not None:
            pos += f":{self.column}"
        return pos


@dataclass
class FillSorrySpec:
    """The full task specification. A bare CLI position list is the spec with
    every default: full repo access, the standard axioms, no computability
    restriction."""

    sorries: list[SorryPos]
    off_limits: list[str] = field(default_factory=list)
    axioms: list[str] = field(default_factory=lambda: list(STANDARD_AXIOMS))
    forbid: list[str] = field(default_factory=list)
    approach: str = ""
    plan: str | None = None
    check_timeout: int = DEFAULT_CHECK_TIMEOUT
    ralph: int | None = None
    # str(pos) -> the fully-qualified declaration each sorry lives in. The
    # goal check is scoped to exactly these declarations: filled by a spec
    # entry's explicit `decl`, or by `validate` via resolve_decl. Kept beside
    # the positions (not on SorryPos) so position identity/dedup is untouched.
    decls: dict[str, str] = field(default_factory=dict)
    # Name-designated sorries ("Ns.foo" instead of a position), pending until
    # `validate` locates each in the tracked sources and converts it into a
    # SorryPos (the header line) + decls entry. Always [] after validate.
    names: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_positions(arg: str) -> list[SorryPos]:
    """Parse the CLI shorthand: comma-separated FILE:LINE[:COL] entries.
    Raises ValueError naming the offending entry. Duplicates are preserved
    here; `validate` dedups them with a warning (a duplicate is a user slip
    worth mentioning, not an error worth stopping for)."""
    positions = []
    for entry in arg.split(","):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
            file, line, col = ":".join(parts[:-2]), int(parts[-2]), int(parts[-1])
        elif len(parts) >= 2 and parts[-1].isdigit():
            file, line, col = ":".join(parts[:-1]), int(parts[-1]), None
        else:
            raise ValueError(
                f"bad sorry position {entry!r}: expected FILE:LINE[:COL] "
                "(project-root-relative, 1-indexed)"
            )
        _check_rel_path(file, f"sorry position {entry!r}")
        if line < 1 or (col is not None and col < 1):
            raise ValueError(
                f"bad sorry position {entry!r}: line and column are 1-indexed"
            )
        positions.append(SorryPos(posixpath.normpath(file), line, col))
    if not positions:
        raise ValueError("no sorry positions given")
    return positions


def _check_rel_path(path: str, what: str) -> None:
    """Reject paths that could escape the project root."""
    if not path:
        raise ValueError(f"{what}: empty file path")
    if path.startswith("/") or path.startswith("\\") or re.match(r"^[A-Za-z]:", path):
        raise ValueError(f"{what}: path must be project-root-relative, not absolute")
    if ".." in path.split("/"):
        raise ValueError(f"{what}: path must not contain `..`")


def parse_designations(arg: str) -> tuple[list[SorryPos], list[str]]:
    """Parse a comma-separated designation list, each entry a position
    (FILE:LINE[:COL]) or a declaration name (`foo`, `Ns.foo`). The rule: an
    entry containing `:`, a path separator, or a `.lean` suffix is meant as
    a position (and gets position errors -- a misremembered `Foo.lean` must
    not silently become a doomed name lookup); anything else is a name,
    checked against the tracked sources by `validate`."""
    positions: list[SorryPos] = []
    names: list[str] = []
    for entry in arg.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if (":" in entry or "/" in entry or "\\" in entry
                or entry.endswith(".lean")):
            positions.extend(parse_positions(entry))
        else:
            names.append(entry)
    if not positions and not names:
        raise ValueError("no sorry designations given")
    return positions, names


def spec_from_arg(arg: str) -> FillSorrySpec:
    """The all-defaults spec for the CLI shorthand (positions and/or
    declaration names)."""
    positions, names = parse_designations(arg)
    return FillSorrySpec(sorries=positions, names=names)


def spec_from_positions(positions: list[SorryPos]) -> FillSorrySpec:
    """The all-defaults spec for a bare position list."""
    return FillSorrySpec(sorries=list(positions))


_SPEC_KEYS = {
    "sorries", "off_limits", "axioms", "forbid",
    "approach", "approach_file", "plan", "check_timeout", "ralph",
}


def load_spec(path: Path) -> FillSorrySpec:
    """Parse a TOML task spec. Raises ValueError with every problem found,
    collected like sandbox._image_problems (a hand-written spec is usually
    wrong in several ways at once).

    Deliberately strict about unknown keys, unlike cli._project_config: that
    file's unknown keys are future features, but here a typo'd `off_limit`
    silently ignored is an *unenforced safety constraint*."""
    try:
        data = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"could not read {path}:\n  {exc}")

    problems: list[str] = []
    for key in sorted(set(data) - _SPEC_KEYS):
        problems.append(f"unknown key `{key}` (valid: {', '.join(sorted(_SPEC_KEYS))})")

    def str_list(key: str) -> list[str]:
        val = data.get(key)
        if val is None:
            return []
        if not isinstance(val, list) or not all(isinstance(s, str) for s in val):
            problems.append(f"`{key}` must be a list of strings")
            return []
        return val

    # A sorries entry is a position string (FILE:LINE[:COL]), a declaration
    # name ("Ns.foo" -- located in the sources by validate), or an inline
    # table pinning the declaration explicitly --
    # { pos = "Foo.lean:42", decl = "MyNs.foo" } -- for the cases source
    # scanning cannot resolve (anonymous instances, unusual layouts).
    sorries: list[SorryPos] = []
    names: list[str] = []
    decls: dict[str, str] = {}
    raw_sorries = data.get("sorries")
    if not isinstance(raw_sorries, list) or not raw_sorries:
        problems.append("`sorries` is required and must be a non-empty list")
        raw_sorries = []
    for entry in raw_sorries:
        try:
            if isinstance(entry, str):
                p, n = parse_designations(entry)
                sorries.extend(p)
                names.extend(n)
            elif isinstance(entry, dict):
                if set(entry) != {"pos", "decl"} or not all(
                    isinstance(v, str) for v in entry.values()
                ):
                    raise ValueError(
                        f"bad sorries entry {entry!r}: a table entry is "
                        '{ pos = "FILE:LINE[:COL]", decl = "Full.Name" }'
                    )
                positions = parse_positions(entry["pos"])
                if len(positions) != 1:
                    raise ValueError(
                        f"bad sorries entry {entry!r}: `pos` names one position"
                    )
                sorries.extend(positions)
                decls[str(positions[0])] = entry["decl"]
            else:
                raise ValueError(f"bad sorries entry {entry!r}")
        except ValueError as exc:
            problems.append(str(exc))

    off_limits = str_list("off_limits")

    axioms = str_list("axioms") if "axioms" in data else list(STANDARD_AXIOMS)
    if "sorryAx" in axioms:
        problems.append("`axioms` must not contain `sorryAx` -- a filled sorry "
                        "is the whole point")

    forbid = str_list("forbid")
    for f in forbid:
        if f not in FORBIDDABLE:
            problems.append(f"`forbid` entry {f!r} is not one of: "
                            + ", ".join(FORBIDDABLE))
    if "native_decide" in forbid and any(a in axioms for a in NATIVE_DECIDE_AXIOMS):
        problems.append(
            "`forbid` includes native_decide but `axioms` allows its axioms "
            f"({', '.join(a for a in NATIVE_DECIDE_AXIOMS if a in axioms)}) -- "
            "pick one"
        )

    approach = data.get("approach", "")
    approach_file = data.get("approach_file")
    if approach and approach_file:
        problems.append("give `approach` or `approach_file`, not both")
    if not isinstance(approach, str):
        problems.append("`approach` must be a string")
        approach = ""
    if approach_file is not None:
        if not isinstance(approach_file, str):
            problems.append("`approach_file` must be a string")
        else:
            ap = path.parent / approach_file
            try:
                approach = ap.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                problems.append(f"could not read approach_file {ap}:\n  {exc}")

    plan = data.get("plan")
    if plan is not None:
        if not isinstance(plan, str) or not plan.strip():
            problems.append("`plan` must be a non-empty string")
            plan = None
        else:
            plan = plan.strip().removeprefix(".gerbil/plans/")
            if not plan.endswith(".md"):
                plan += ".md"
            if "/" in plan or "\\" in plan or ".." in plan:
                problems.append(
                    "`plan` must be a bare *.md filename -- the plan always "
                    "lives at .gerbil/plans/<name>.md"
                )
                plan = None

    check_timeout = data.get("check_timeout", DEFAULT_CHECK_TIMEOUT)
    if not isinstance(check_timeout, int) or isinstance(check_timeout, bool) \
            or check_timeout < 1:
        problems.append("`check_timeout` must be a positive integer (seconds)")
        check_timeout = DEFAULT_CHECK_TIMEOUT

    ralph = data.get("ralph")
    if ralph is not None and (not isinstance(ralph, int) or isinstance(ralph, bool)
                             or ralph < 1):
        problems.append("`ralph` must be an integer >= 1")
        ralph = None

    if problems:
        raise ValueError(f"problems in {path}:\n"
                         + "".join(f"  - {p}\n" for p in problems).rstrip("\n"))

    return FillSorrySpec(
        sorries=sorries, off_limits=off_limits, axioms=axioms, forbid=forbid,
        approach=approach.strip(), plan=plan, check_timeout=check_timeout,
        ralph=ralph, decls=decls, names=names,
    )


# ---------------------------------------------------------------------------
# Derived names
# ---------------------------------------------------------------------------


def plan_name(spec: FillSorrySpec) -> str:
    """The plan file's project-root-relative path, always under
    `.gerbil/plans/` -- gerbil's namespace in the repo, next to where the
    folded session logs land: `.gerbil/plans/<name>.md`, with the name taken
    from the spec's override or derived deterministically from the sorry
    list. The plan is still patch-native: `.gerbil/` is conventionally
    gitignored, so the squash/wip force-include it exactly the way the
    session log is force-added, and its updates ship inside each session's
    patch. Deterministic on purpose: a second run of the same task names the
    same file and (once the earlier patches are committed) inherits its
    memory, while a different sorry list gets a fresh one.

    Tolerates a spec.plan that is already fully qualified -- that is how the
    recorded metadata round-trips through spec_from_meta."""
    if spec.plan:
        name = spec.plan
    else:
        slug = re.sub(r"[^a-z0-9]+", "-",
                      Path(spec.sorries[0].file).stem.lower())
        slug = slug.strip("-") or "task"
        h = hashlib.sha256(
            "\n".join(sorted(str(p) for p in spec.sorries)).encode()
        ).hexdigest()[:8]
        name = f"fill-{slug}-{h}.md"
    if not name.startswith(".gerbil/plans/"):
        name = f".gerbil/plans/{name}"
    return name


def module_of(path: str) -> str:
    """The Lean module a project-relative .lean path elaborates as, assuming
    the standard Lake layout (no custom srcDir or globs): Foo/Bar.lean ->
    Foo.Bar. Good enough for the files that hold the sorries; a nonstandard
    layout fails loudly in the check ("target module is not in the
    environment") rather than silently passing."""
    return path.removesuffix(".lean").replace("/", ".")


def target_modules(spec: FillSorrySpec) -> list[str]:
    """The modules the check sweeps, sorted and deduplicated."""
    return sorted({module_of(p.file) for p in spec.sorries})


# ---------------------------------------------------------------------------
# Lexing: enough of Lean's surface syntax to tell code from prose
#
# Both the declaration scan and the sorry census must not be fooled by text
# that merely *reads* like Lean. A module docstring saying "the proofs are
# intentionally left as `sorry`" is the common case -- counting it made
# --fill-sorry warn about undesignated sorries in a file that had none, which
# is worse than no warning: it tells the user their designation list is
# incomplete when it is exactly right.
# ---------------------------------------------------------------------------

# A char literal, matched only to step over it: `'` is also an ordinary
# identifier character in Lean (h', k'), so it can never be treated as the
# start of a quoted span -- but `'"'` must not be read as opening a string.
_CHAR_LIT_RE = re.compile(r"'(?:\\.|[^'\\])'")


def code_only(lines: list[str]) -> list[str]:
    """`lines` with every comment and string literal blanked to spaces.

    Same number of lines, each the same length as its original, so 1-indexed
    line and column numbers still address the same characters -- callers can
    scan the result and report positions in the real file.

    Handles what actually shows up in Lean sources: `--` line comments,
    `/- -/` block comments (which *nest*, and cover `/-- -/` docstrings), and
    string literals with backslash escapes. Multi-line strings keep their
    state across lines like block comments do; the char-literal step-over
    above is what keeps a stray `'"'` from swallowing the rest of the file."""
    out: list[str] = []
    depth = 0        # block-comment nesting
    in_string = False
    for text in lines:
        buf = []
        i, n = 0, len(text)
        while i < n:
            two = text[i:i + 2]
            if depth:
                if two == "/-":
                    depth += 1
                    buf.append("  ")
                    i += 2
                    continue
                if two == "-/":
                    depth -= 1
                    buf.append("  ")
                    i += 2
                    continue
                buf.append(" ")
                i += 1
                continue
            if in_string:
                if text[i] == "\\" and i + 1 < n:
                    buf.append("  ")
                    i += 2
                    continue
                if text[i] == '"':
                    in_string = False
                buf.append(" ")
                i += 1
                continue
            if two == "--":
                buf.append(" " * (n - i))
                break
            if two == "/-":
                depth = 1
                buf.append("  ")
                i += 2
                continue
            if text[i] == '"':
                in_string = True
                buf.append(" ")
                i += 1
                continue
            if text[i] == "'" and (m := _CHAR_LIT_RE.match(text, i)):
                buf.append(m.group(0))  # code, and holds no `sorry`
                i = m.end()
                continue
            buf.append(text[i])
            i += 1
        out.append("".join(buf))
    return out


# `sorry` as a term, not as part of a longer name: sorryAx, foo_sorry and
# `Foo.sorry` (dot-notation) are all something else. Lean identifiers admit
# unicode letters and digits plus _ ' ! ?, which is exactly what \w and the
# extras here cover.
_SORRY_RE = re.compile(r"(?<![\w'!?.])sorry(?![\w'!?])")


def find_sorries(lines: list[str]) -> list[tuple[int, int]]:
    """Every real `sorry` term in a file, as 1-indexed (line, column) --
    comments and strings excluded (see code_only)."""
    return [
        (i, m.start() + 1)
        for i, text in enumerate(code_only(lines), start=1)
        for m in _SORRY_RE.finditer(text)
    ]


# ---------------------------------------------------------------------------
# Position -> declaration resolution
#
# The goal check is scoped to the declaration each designated sorry lives in
# (collectAxioms is transitive, so "this theorem carries no sorryAx" already
# covers every helper its proof routes through). Line numbers drift under the
# agent's edits, so the check works by *name*: this resolver derives the name
# once, at preflight, from the source as the task was created.
# ---------------------------------------------------------------------------

_DECL_RE = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|scoped|local|nonrec)\s+)*"
    r"(theorem|lemma|def|abbrev|instance|structure|inductive|class|opaque|example)\b"
    r"(?:\s+(_root_\.)?([^\s:(\[{⟨|)\]]+))?"
)
_NS_RE = re.compile(r"^\s*namespace\s+([\w.«»']+)")
_SCOPE_RE = re.compile(r"^\s*(?:(?:noncomputable|private)\s+)?(?:section\b|mutual\b)")
_END_RE = re.compile(r"^\s*end\b")


def _scan_decls(lines: list[str]) -> list[tuple[int, str, str | None]]:
    """Every declaration header in a file, as (1-indexed line, kind,
    fully-qualified name or None when anonymous), with names qualified by
    the `namespace` blocks open at each point. A best-effort *syntactic*
    scan on purpose: elaborating the project host-side just to name a
    declaration would cost a full build at preflight, and the check
    re-resolves names against the live environment anyway (suffix match
    within the module), so a plausible mis-qualification still lands. The scan
    runs over code_only(lines), so a header quoted in a docstring or commented
    out is not picked up."""
    out: list[tuple[int, str, str | None]] = []
    stack: list[tuple[str, str]] = []  # ("namespace", name) | ("scope", "")
    for i, text in enumerate(code_only(lines), start=1):
        if m := _NS_RE.match(text):
            stack.append(("namespace", m.group(1)))
        elif _SCOPE_RE.match(text):
            stack.append(("scope", ""))
        elif _END_RE.match(text):
            if stack:
                stack.pop()
        elif m := _DECL_RE.match(text):
            if m.group(3):
                ns = [] if m.group(2) else [
                    n for kind, n in stack if kind == "namespace"
                ]
                fqn = ".".join(ns + [m.group(3)])
            else:
                fqn = None
            out.append((i, m.group(1), fqn))
    return out


def _enclosing_header(headers: list[int], line: int) -> int | None:
    """The header line of the declaration containing 1-indexed `line` -- the
    last header at or above it -- or None when nothing is (a sorry above the
    file's first declaration). The same "last header wins" rule resolve_decl
    uses, over a header list already scanned."""
    found = None
    for h in headers:
        if h > line:
            break
        found = h
    return found


def resolve_decl(lines: list[str], line: int) -> tuple[str | None, str]:
    """The fully-qualified name of the declaration containing 1-indexed
    `line`: the last declaration header at or above it. Returns (name, "")
    on success or (None, why) when the source defeats the scan or the site
    is one --fill-sorry cannot soundly check (see the messages)."""
    found: tuple[str, str | None] | None = None  # kind, fqn
    for decl_line, kind, fqn in _scan_decls(lines):
        if decl_line > line:
            break
        found = (kind, fqn)
    if found is None:
        return None, "no declaration header found above it"
    kind, name = found
    if kind == "example":
        return None, ("it sits in an `example`, which never enters the "
                      "environment -- name the statement (e.g. make it a "
                      "theorem) so the goal can be checked")
    if kind in ("structure", "inductive", "class", "opaque"):
        # A sorry in e.g. a structure field default elaborates into an
        # auto-generated constant (Cfg.x._default) that collectAxioms on the
        # type itself never visits -- the check would pass with the sorry
        # still present. Refusing here is what keeps the goal sound.
        return None, (
            f"it sits in a `{kind}` declaration, whose sorries live in "
            "auto-generated constants the declaration's axiom check cannot "
            "see. --fill-sorry supports sorries only in the bodies of "
            "top-level theorem/lemma/def/abbrev/instance declarations -- "
            "move the sorry into one (e.g. a plain def used as the default)"
        )
    if not name:
        return None, (f"the enclosing `{kind}` is anonymous -- give the "
                      "declaration a name, or state it explicitly in a spec "
                      "entry: { pos = \"...\", decl = \"TheName\" }")
    return name, ""


def seed_plan(spec: FillSorrySpec) -> str:
    """The plan file's initial content. gerbil seeds it into the container at
    the exact path the generated prompt names, so the agent never decides
    where the plan lives -- it only ever reads and appends. The seed is
    ordinary working-tree content: the session squash commits it, so it
    enters the repository through the session's own patch like every other
    new file, and it reaches the host only via `gerbil commit`."""
    lines = "".join(
        f"  - `{p}` (in `{spec.decls.get(str(p), '?')}`)\n"
        for p in spec.sorries
    )
    return (
        "# Plan: fill the designated sorries\n"
        "\n"
        "The task: resolve these sorries, designated as\n"
        "FILE:LINE[:COLUMN] (as of the task's starting commit):\n"
        "\n"
        f"{lines}"
        "\n"
        "This file is the task's only cross-session memory, maintained by\n"
        "the agent. Append a session entry at the end of every session --\n"
        "what was done, what worked, what failed and why, and what to do\n"
        "next. Never delete this file or earlier entries. If there are no\n"
        "session entries below, no session has run yet.\n"
    )


# ---------------------------------------------------------------------------
# Off-limits matching (shared semantics: host patch gate + generated check)
# ---------------------------------------------------------------------------


def normalize_off_limits(patterns: list[str]) -> list[str]:
    """Expand the spec's off_limits into concrete fnmatch patterns. A pattern
    with no glob character names a file *or* a directory, so it becomes both
    `pat` and `pat/*`; explicit globs pass through. fnmatch's `*` crosses `/`,
    and so does bash `[[ $path == $pat ]]` -- the generated check script and
    the host-side gate agree by construction."""
    out = []
    for pat in patterns:
        pat = pat.rstrip("/")
        if not pat:
            continue
        out.append(pat)
        if not any(c in pat for c in "*?["):
            out.append(pat + "/*")
    return out


def is_off_limits(path: str, normalized: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pat) for pat in normalized)


def off_limits_monitor(off_limits: list[str], subdir: str):
    """A per-session hook for the agent loop: fed each turn's wip-patch
    text, returns the warning the model must see when its changes have newly
    touched an off-limits path, else None. None (no hook at all) when the
    task has no off-limits paths.

    Stateful on purpose: the wip patch is cumulative since the session base,
    so an unfixed violation would otherwise re-warn every turn and burn the
    very context the model needs to fix it. The warning fires when the
    violation set GROWS relative to the previous turn -- once per offense --
    and a path that is fixed and then broken again warns again. Warning
    only: reverting is the model's job (the patch gate and the termination
    check remain the enforcement)."""
    if not off_limits:
        return None
    prev: set[str] = set()

    def check(patch_text: str) -> str | None:
        # Local import: view is display machinery, and this module stays
        # otherwise free of gerbil-internal imports.
        from .view import patch_file_stats
        nonlocal prev
        violations = set(gate_violations(
            list(patch_file_stats(patch_text)), off_limits, subdir))
        grew = violations - prev
        prev = violations
        if not grew:
            return None
        listing = "".join(f"  {p}\n" for p in sorted(violations))
        return (
            "[OFF-LIMITS] Your changes currently modify path(s) that are "
            "off-limits for this task:\n"
            f"{listing}"
            "These paths are frozen: the termination check pins them to the "
            "starting commit, and gerbil will refuse to emit a patch that "
            "touches them -- the session's work cannot land while they "
            "differ. Restore each one to its exact original content before "
            "continuing with anything else (the original is available via "
            "`git show <base-commit>:<path>`, with the base commit named in "
            "your instructions; rewrite the file to match it byte for byte). "
            "Do not use git checkout/reset/restore."
        )

    return check


def gate_violations(
    changed: list[str], off_limits: list[str], subdir: str
) -> list[str]:
    """The changed paths (repo-root-relative, from sandbox.changed_paths) that
    violate the spec's off_limits (project-root-relative). Paths under
    .gerbil/ are exempt -- the folded session log lives there legitimately,
    and it is where gerbil's own bookkeeping goes."""
    prefix = subdir + "/" if subdir else ""
    normalized = [prefix + pat for pat in normalize_off_limits(off_limits)]
    gerbil_dirs = {".gerbil"} | ({posixpath.join(subdir, ".gerbil")} if subdir else set())
    violations = []
    for path in changed:
        if any(path == g or path.startswith(g + "/") for g in gerbil_dirs):
            continue
        if is_off_limits(path, normalized):
            violations.append(path)
    return violations


# ---------------------------------------------------------------------------
# Preflight validation
# ---------------------------------------------------------------------------


def validate(
    spec: FillSorrySpec, *, project_dir: Path, repo_root: Path
) -> tuple[list[str], list[str], dict[SorryPos, str]]:
    """Check the spec against the actual repo, before any container boots.
    Returns (errors, warnings, excerpts): errors abort the run, warnings are
    printed and ignored, and excerpts maps each position to its current
    source line for the generated prompt.

    The impure function of this module: reads the sorry files and one
    `git ls-files`."""
    errors: list[str] = []
    warnings: list[str] = []
    excerpts: dict[SorryPos, str] = {}

    subdir = project_subdir(project_dir, repo_root)
    tracked = set(
        subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "-z"],
            capture_output=True, text=True,
        ).stdout.split("\0")
    )

    normalized = normalize_off_limits(spec.off_limits)

    # Name designations: locate each declaration across the tracked .lean
    # sources (the reverse of resolve_decl) and fold it into the position
    # list as its header line. from_name suppresses the "line does not
    # contain sorry" warning below -- a header like `theorem foo : P := by`
    # is the expected shape there.
    from_name: set[SorryPos] = set()
    if spec.names:
        prefix = subdir + "/" if subdir else ""
        scans: dict[str, list] = {}
        for t in sorted(tracked):
            if not t.endswith(".lean") or (prefix and not t.startswith(prefix)):
                continue
            f = t[len(prefix):]
            try:
                scans[f] = _scan_decls(
                    (project_dir / f).read_text().splitlines())
            except (OSError, UnicodeDecodeError):
                continue
        for name in spec.names:
            # A designation matches a declaration when it names it by its
            # (possibly namespace-suffixed) Lean name -- and also when it is
            # MODULE-qualified: users naturally write Hello.Basic.foo for a
            # bare `foo` in Hello/Basic.lean, since module path and namespace
            # coincide by convention. Per file, a leading module prefix is
            # stripped before the name match.
            def hits(f: str, fqn: str) -> bool:
                if fqn == name or fqn.endswith("." + name):
                    return True
                mod_prefix = module_of(f) + "."
                if name.startswith(mod_prefix):
                    rest = name[len(mod_prefix):]
                    return fqn == rest or fqn.endswith("." + rest)
                return False

            matches = [
                (f, line, kind, fqn)
                for f, headers in scans.items()
                for line, kind, fqn in headers
                if fqn and hits(f, fqn)
            ]
            if not matches:
                errors.append(
                    f"`{name}`: no declaration with this name found in the "
                    "project's tracked .lean sources (a name is matched by "
                    "its namespace-qualified form, optionally prefixed by "
                    "its file's module path)"
                )
            elif len(matches) > 1:
                cand = ", ".join(
                    f"{fqn} ({f}:{line})" for f, line, _, fqn in matches)
                errors.append(
                    f"`{name}` is ambiguous: {cand} -- qualify the name")
            elif matches[0][2] in ("structure", "inductive", "class",
                                   "opaque", "example"):
                f, line, kind, _ = matches[0]
                errors.append(
                    f"`{name}` is a `{kind}` ({f}:{line}) -- --fill-sorry "
                    "supports only top-level theorem/lemma/def/abbrev/"
                    "instance declarations"
                )
            else:
                f, line, _, fqn = matches[0]
                pos = SorryPos(f, line)
                spec.sorries.append(pos)
                spec.decls.setdefault(str(pos), fqn)
                from_name.add(pos)
        spec.names = []

    seen: set[SorryPos] = set()
    deduped: list[SorryPos] = []
    file_lines: dict[str, list[str]] = {}   # for the undesignated-sorry note
    for pos in spec.sorries:
        if pos in seen:
            warnings.append(f"duplicate sorry position {pos} (ignored)")
            continue
        seen.add(pos)
        deduped.append(pos)

        repo_rel = posixpath.join(subdir, pos.file) if subdir else pos.file
        host = project_dir / pos.file
        if not host.is_file():
            errors.append(f"{pos}: no such file under {project_dir}")
            continue
        if repo_rel not in tracked:
            errors.append(
                f"{pos}: {pos.file} is not tracked by git -- gerbil uploads "
                "only tracked files, so the agent would never see it. "
                "Commit it first."
            )
            continue
        try:
            lines = host.read_text().splitlines()
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{pos}: could not read: {exc}")
            continue
        if pos.line > len(lines):
            errors.append(f"{pos}: file has only {len(lines)} lines")
            continue
        file_lines[pos.file] = lines
        text = lines[pos.line - 1]
        if (pos.line not in {l for l, _ in find_sorries(lines)}
                and pos not in from_name):
            warnings.append(f"{pos}: the line does not contain `sorry` "
                            f"(it reads: {text.strip()[:60]!r})")
        if pos.column is not None and pos.column > len(text) + 1:
            warnings.append(f"{pos}: column {pos.column} is past the end of "
                            f"the line ({len(text)} chars)")
        excerpts[pos] = text.strip()[:80]

        if is_off_limits(pos.file, normalized):
            errors.append(
                f"{pos}: {pos.file} matches off_limits -- an off-limits file "
                "cannot be edited, so its sorry cannot be filled"
            )

        # The goal check is scoped to each sorry's enclosing declaration:
        # resolve it now (unless the spec named it explicitly), and fail
        # loudly when the source defeats the scan -- a goal that cannot be
        # checked must not silently become "the build passes".
        if str(pos) not in spec.decls:
            decl, why = resolve_decl(lines, pos.line)
            if decl is None:
                errors.append(
                    f"{pos}: cannot determine the enclosing declaration -- "
                    f"{why}. A spec-file entry can name it explicitly: "
                    f'{{ pos = "{pos}", decl = "TheName" }}'
                )
            else:
                spec.decls[str(pos)] = decl

    # Sorries the goal does not cover, noted so nobody expects the check to
    # demand their removal. The scope is the *declaration*, not the file: the
    # check runs collectAxioms per designated declaration, so a second sorry
    # inside one of them is part of the goal after all, and only sorries
    # elsewhere in the file are not. Counting real `sorry` terms (find_sorries,
    # not the word anywhere in the text) is the other half of getting this
    # right -- a docstring that merely mentions `sorry` used to make a file
    # whose designation list was exactly complete report otherwise.
    for f, lines in file_lines.items():
        headers = [line for line, _kind, _fqn in _scan_decls(lines)]
        covered = {_enclosing_header(headers, p.line)
                   for p in spec.sorries if p.file == f}
        extra = [line for line, _col in find_sorries(lines)
                 if _enclosing_header(headers, line) not in covered]
        if extra:
            shown = ", ".join(str(line) for line in extra[:6])
            if len(extra) > 6:
                shown += ", ..."
            warnings.append(
                f"note: {f} has {len(extra)} `sorry`(s) outside the "
                f"designated declarations (line{'s' if len(extra) > 1 else ''} "
                f"{shown}) -- they are not part of the goal; the agent is "
                "told to leave them alone"
            )

    if not spec.sorries:
        # Every designation failed to resolve (errors above say why);
        # nothing is left to derive a plan name from.
        return errors, warnings, excerpts

    # The plan file lives at .gerbil/plans/<name>.md and its updates ship
    # inside each session's patch (force-included past the conventional
    # .gerbil gitignore, exactly like the folded session log). The one state
    # that would silently break the memory chain: an existing host copy the
    # patch chain does not know about -- invisible to the upload (untracked
    # files never enter the sandbox), and colliding with `git am` when a
    # patch later creates its own. A *tracked* plan file is simply a
    # continuing task.
    plan = plan_name(spec)
    plan_repo = posixpath.join(subdir, plan) if subdir else plan
    if plan_repo not in tracked and (project_dir / plan).exists():
        errors.append(
            f"the plan file {plan} exists but is not tracked by git -- the "
            "agent would never see it, and the patch that creates its own "
            "copy would collide with it at `gerbil commit`. Commit it or "
            "remove it (or set `plan` in the spec to another name)."
        )

    spec.sorries = deduped
    return errors, warnings, excerpts


def project_subdir(project_dir: Path, repo_root: Path) -> str:
    """The project's path inside the repo, '' when the project is the root."""
    rel = project_dir.resolve().relative_to(repo_root.resolve()).as_posix()
    return "" if rel == "." else rel


# ---------------------------------------------------------------------------
# The generated prompt
# ---------------------------------------------------------------------------


def build_prompt(
    spec: FillSorrySpec, *, plan_rel: str, excerpts: dict[SorryPos, str]
) -> str:
    """The task prompt: the generalized form of what absriscv's mkprompt.sh
    hand-rolled for one project. Structure: the sorries, the file rules, a
    concrete GOAL (exactly what the check verifies), RULES (including the
    plan-file protocol that gives a ralph chain memory), optional APPROACH,
    and the per-session task framing."""
    axioms = ", ".join(f"`{a}`" for a in spec.axioms)
    decl_names = sorted({spec.decls[str(p)] for p in spec.sorries
                         if str(p) in spec.decls})
    decl_list = ", ".join(f"`{d}`" for d in decl_names)

    def entry(pos: SorryPos) -> str:
        decl = spec.decls.get(str(pos))
        where = f" (in `{decl}`)" if decl else ""
        return f"  * `{pos}`{where} -- `{excerpts.get(pos, '?')}`\n"

    sorry_lines = "".join(entry(pos) for pos in spec.sorries)

    n = len(spec.sorries)
    out = [
        "Your task is to fill in specific, designated Lean `sorry`s in this "
        "project, listed below -- those and no others. This is a "
        "multi-session effort: the same task may run again in a fresh "
        "session that builds on what you leave behind now.\n"
        "\n"
        "## THE SORRIES\n"
        "\n"
        f"Exactly {'this location' if n == 1 else f'these {n} locations'}, "
        "given as FILE:LINE[:COLUMN] (project-root-relative, 1-indexed) plus "
        "the enclosing declaration, each with the text of its line as the "
        "task was created (your edits may have moved it since -- earlier "
        "sessions of this same task may already have progressed or even "
        "filled some):\n"
        "\n"
        f"{sorry_lines}"
        "\n"
        "## THE FILES\n"
        "\n"
    ]

    if spec.off_limits:
        off_lines = "".join(f"  * `{pat}`\n" for pat in spec.off_limits)
        out.append(
            "Read, do not change -- these paths are OFF-LIMITS:\n"
            "\n"
            f"{off_lines}"
            "\n"
            "They are the specification, not the workspace: changing them "
            "would not make the proofs correct, it would change the question. "
            "This is enforced, not merely asked: the termination check pins "
            "every off-limits path to the commit this task started from, and "
            "gerbil rejects any session whose final patch touches one.\n"
            "\n"
            "Everything else in the repository is yours to edit: "
        )
    else:
        out.append("The whole repository is yours to edit: ")

    out.append(
        "fill the sorries in place, add helper lemmas or new files, refactor "
        "a helper an earlier session left behind -- whatever the proofs need. "
        "A new file must be imported (directly or transitively) by the "
        "project's build targets, or the build will not see it. Anything you "
        "find beyond the original code was written by an earlier session of "
        "this same task: it is your own work handed forward, not part of the "
        "specification -- edit it, restate it, or delete it freely.\n"
        "\n"
        f"  * `{plan_rel}` -- the running plan (see RULES), already present "
        "in the working tree at exactly that path. An ordinary repository "
        "file: it is committed with your work, ships in your patch, and is "
        "how this task remembers anything between sessions. Do not move it "
        "or start another plan file elsewhere.\n"
        "\n"
        "## GOAL\n"
        "\n"
        "Every designated sorry is resolved. Concretely, the termination "
        "check verifies:\n"
        "\n"
        "  * `lake build` succeeds with no errors;\n"
        f"  * each designated declaration -- {decl_list} -- still exists "
        "under its original name (anywhere in the library: moving it to "
        "another file is fine), and depends on no `sorryAx` (a surviving "
        "`sorry` in any disguise, anywhere in its proof or in anything it "
        f"uses) and on no axiom beyond: {axioms};\n"
    )
    if "noncomputable" in spec.forbid or "partial" in spec.forbid:
        words = " or ".join(
            f"`{w}`" for w in ("noncomputable", "partial") if w in spec.forbid
        )
        out.append(
            f"  * neither the designated declarations nor anything they use "
            f"within their module(s) is {words};\n"
        )
    if spec.off_limits:
        out.append("  * no off-limits path differs from the starting commit.\n")
    out.append(
        "\n"
        "The check is scoped to exactly the designated declarations (and, "
        "through their dependencies, whatever they use). Other `sorry`s in "
        "the same files are NOT part of this task and do not block "
        "completion.\n"
        "\n"
        "## RULES\n"
        "\n"
        "  * **Fill only the designated sorries.** Any other `sorry` you "
        "encounter -- in the same file or elsewhere -- is not yours: leave "
        "it exactly as it is, and do not route your proofs through it (a "
        "designated declaration that depends on a sorried helper still "
        "fails the check).\n"
        "  * **Do not change the statement a sorry lives in**, and do not "
        "rename or delete its declaration -- the check looks the "
        "declaration up by name and fails if it is gone (moving it to "
        "another file in the library, unchanged, is fine). Your task is "
        "to prove what is asked, not to ask an easier question. If you "
        "conclude a statement is wrong or unprovable as written, do not edit "
        "it and do not route around it: write the argument into the plan "
        "file, leave the build green, and end the session. A human "
        "adjudicates a change to the question.\n"
    )
    if spec.off_limits:
        out.append(
            "  * Do not touch the OFF-LIMITS paths, and do not reproduce "
            "their content elsewhere to shadow them.\n"
        )
    native_allowed = all(a in spec.axioms for a in NATIVE_DECIDE_AXIOMS)
    out.append(
        "  * No new `axiom` declarations. Axioms beyond the allowed list"
        + ("" if native_allowed
           else " -- including the ones `native_decide` introduces --")
        + " fail the check no matter where they hide.\n"
    )
    if "noncomputable" in spec.forbid or "partial" in spec.forbid:
        words = " or ".join(
            f"`{w}`" for w in ("noncomputable", "partial") if w in spec.forbid
        )
        out.append(
            f"  * Do not declare anything {words} in the target module(s).\n"
        )
    out.append(
        f"  * **Keep `{plan_rel}`.** Begin every session by reading it; if "
        "it holds no session entries yet, you are the first session. End "
        "every session by appending an entry: what you did, the approach "
        "you settled on and why, what you tried that did NOT work and why, "
        "and what to do next. Never delete the file or what is already "
        "written there; accumulate. Leave it in the working tree like any "
        "other file -- it is committed with your work. The next session "
        "cannot see this one; that file is the only thing that carries "
        "over.\n"
        "  * **Verify with the `check_goal` tool.** It runs the exact "
        "termination check described in the GOAL -- the one the harness "
        "runs between sessions -- and shows you its verdict and output. "
        "When you believe the GOAL is reached, run it and read the result: "
        "the check is the definition of done, and its opinion overrides "
        "yours. If it says NOT PASSED, the output names the failing "
        "condition -- fix that, do not argue with it in prose. It runs a "
        "full build, so use it at natural checkpoints, not after every "
        "edit.\n"
        "  * When you finish a session the project must build without "
        "errors. A remaining `sorry` is far better than a broken proof.\n"
        "  * Store temporary files in /tmp via mktemp; leave none in the "
        "repository. Delete unused lemmas and definitions.\n"
    )

    if spec.approach:
        out.append(f"\n## APPROACH\n\n{spec.approach}\n")

    out.append(
        "\n"
        "## YOUR TASK THIS SESSION\n"
        "\n"
        f"Read `{plan_rel}`, then make substantial progress towards the GOAL, "
        "then append to the plan file what you did and what you learned. You "
        "do not have to finish now -- previous sessions may already have made "
        "progress, and later ones can build on yours. You only have to make "
        "real progress and leave the build green.\n"
        "\n"
        "Good ways to make progress:\n"
        "\n"
        "  * fill in one or more of the listed sorries;\n"
        "  * correct an earlier session's mistake, recording in the plan "
        "file what it was and how it was found;\n"
        "  * add a missing supporting lemma, documenting in the plan file "
        "why it is needed and how it will be used.\n"
    )
    return "".join(out)


# ---------------------------------------------------------------------------
# The generated check-goal script
# ---------------------------------------------------------------------------

# Substitution is by unique @GERBIL_*@ markers and str.replace, not
# str.format/string.Template: the template is dense with bash `${...}`, Lean
# `s!"{...}"`, and printf `%s`, and a marker scheme has no escaping hazards
# (it is also exactly how absriscv's sed-based generator worked).
# _build assembles the pieces and asserts no marker survives.

_CHECK_HEADER = """\
#!/usr/bin/env bash
# check_goal.sh -- generated by `gerbil run --fill-sorry`. Exits 0 iff the
# GOAL is achieved: the project builds, each designated sorry's declaration
# depends on no sorryAx and no disallowed axiom, and every restriction holds.
#
#   (exit 0 ends the ralph loop; non-zero continues it)
set -u

fail() { echo "NOT DONE: $*"; exit 1; }

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) \\
  || fail "not inside a git repository"

BASE=@GERBIL_BASE@
git cat-file -e "$BASE^{commit}" 2>/dev/null \\
  || fail "base commit $BASE is not in this repository"
"""

_CHECK_OFF_LIMITS = """\

# ---------------------------------------------------------------------
# 1. Off-limits paths must match the base commit exactly. .gerbil/ is
#    gerbil's own bookkeeping (plan file, folded session logs) and is
#    exempt. Patterns are matched with bash [[ == ]] globbing, where `*`
#    crosses `/` -- the same fnmatch semantics gerbil's host-side patch
#    gate applies.
# ---------------------------------------------------------------------
OFF_LIMITS=(@GERBIL_OFF_LIMITS_ARRAY@)
bad=""
while IFS=$'\\t' read -r st path _; do
  [ -n "${st:-}" ] || continue
  case "$path" in @GERBIL_SKIP@) continue ;; esac
  for pat in "${OFF_LIMITS[@]}"; do
    if [[ $path == $pat ]]; then
      bad="$bad  $st  $path"$'\\n'; break
    fi
  done
done < <(git -C "$ROOT" diff --no-renames --name-status "$BASE" -- .)
if [ -n "$bad" ]; then
  printf '%s' "$bad"
  fail "off-limits paths differ from base ${BASE:0:12}"
fi
"""

_CHECK_BUILD = """\

# ---------------------------------------------------------------------
# 2. The project must build -- and so must every module of the
#    designated declarations' library, enumerated from the tracked
#    .lean files AS OF THIS CHECK: a designated declaration may
#    legitimately have moved to a file created mid-task, and an
#    unimported new file still needs its .olean for the sweep below.
#    (A path containing whitespace cannot be a Lean module: skipped.)
# ---------------------------------------------------------------------
LOG=$(mktemp); DIR=$(mktemp -d)
trap 'rm -f "$LOG"; rm -rf "$DIR"' EXIT
MODULES=$(git -C "$ROOT" ls-files -- @GERBIL_LEAN_PATHSPECS@ \\
  | grep -v ' ' | sed -e 's|^@GERBIL_STRIP@||' -e 's|\\.lean$||' -e 's|/|.|g')
[ -n "$MODULES" ] || fail "no tracked .lean files under: @GERBIL_ROOTS@"
if ! lake build >"$LOG" 2>&1; then
  tail -n 40 "$LOG"; fail "lake build failed"
fi
if ! lake build $(printf '+%s ' $MODULES) >>"$LOG" 2>&1; then
  tail -n 40 "$LOG"; fail "building the library's module(s) failed"
fi

# ---------------------------------------------------------------------
# 3. Declaration-scoped Lean checks, elaborated against the built
#    library. Each designated sorry is checked through its enclosing
#    declaration, re-resolved BY NAME across every module of the
#    library (line numbers drift under the agent's edits, and whole
#    declarations may legitimately move between files; a renamed or
#    deleted declaration is a hard failure, never a pass).
#    collectAxioms is transitive, so "no sorryAx, no disallowed axiom"
#    covers every helper a designated proof routes through -- while
#    other sorries in the same files are simply not part of the goal.
#    native_decide needs no dedicated check: its axioms are absent from
#    the allowed list unless the spec permitted them.
# ---------------------------------------------------------------------
{
  echo "import Lean"
  for m in $MODULES; do echo "import $m"; done
  MODLIST="["
  for m in $MODULES; do MODLIST="$MODLIST\\`$m, "; done
  echo "def gerbilModules : List Lean.Name := ${MODLIST%, }]"
  cat <<'GERBIL_CHECK_EOF'

open Lean

#eval show CoreM Unit from do
  let env ← getEnv
  let designated : List (Name × String) := @GERBIL_DECLS@
  let allowed : List Name := @GERBIL_ALLOWED_AXIOMS@
  let projIdxs := gerbilModules.filterMap env.getModuleIdx?
  -- Resolve each designated name across the library: an exact match on
  -- the display name, with a unique suffix match as the fallback (the
  -- preflight scan may under-qualify a namespace). Exact wins outright
  -- so a designated bare `foo` is never blocked by a namespaced
  -- `Ns.foo` cousin. Display names because a private declaration's
  -- real name is mangled (_private.<Mod>.0.<Name>), which both fails a
  -- suffix match and is classified isInternal -- privateToUserName?
  -- undoes that, so private theorems resolve like public ones. The
  -- recorded origin module is error-message context only: the
  -- declaration is free to move between the library's files.
  let mut resolved : Array Name := #[]
  for (mod, s) in designated do
    let mut exact : Array Name := #[]
    let mut cands : Array Name := #[]
    for (c, _) in env.constants.toList do
      if (env.getModuleIdxFor? c).any (projIdxs.contains ·) then
        let d := (privateToUserName? c).getD c
        if !d.isInternal then
          if d.toString == s then
            exact := exact.push c
          else if d.toString.endsWith ("." ++ s) then
            cands := cands.push c
    let picked := if exact.isEmpty then cands else exact
    if picked.size == 1 then
      resolved := resolved.push picked[0]!
    else if picked.isEmpty then
      throwError "no declaration matching {s} (originally in {mod}) \\
anywhere in the library -- the designated sorry's declaration must \\
keep its name and statement"
    else
      throwError "several declarations match {s}: {picked}"
  let mut uniq : Array Name := #[]
  for c in resolved do
    unless uniq.contains c do uniq := uniq.push c
  let mut bad : Array (Name × Name) := #[]
  for c in uniq do
    for ax in (← collectAxioms c) do
      unless allowed.contains ax do
        bad := bad.push (c, ax)
  unless bad.isEmpty do
    throwError "disallowed axioms (sorryAx means the sorry survives): \\
{bad.toList.take 8}"
  IO.println s!"axiom check OK: {uniq.toList} depend on nothing beyond \\
{allowed}"
@GERBIL_FORBID_BLOCK@
GERBIL_CHECK_EOF
} >"$DIR/Check.lean"
"""

# Spliced into the #eval above (same do-block, so `env`/`projIdxs`/
# `uniq` are in scope) when the spec forbids noncomputable/partial.
# Walks the designated declarations and everything they use,
# transitively, WITHIN the library's modules: dependencies outside the
# library are not the agent's doing.
_CHECK_FORBID_BLOCK = """\
  let idxs := projIdxs
  let mut queue := uniq.toList
  let mut seenc : Array Name := #[]
  let mut badc : Array (Name × String) := #[]
  while !queue.isEmpty do
    let c := queue.head!
    queue := queue.tail!
    if seenc.contains c then continue
    seenc := seenc.push c
    if (env.getModuleIdxFor? c).any (idxs.contains ·) then
@GERBIL_FORBID_NONCOMPUTABLE@
@GERBIL_FORBID_PARTIAL@
      if let some ci := env.find? c then
        for d in ci.getUsedConstantsAsSet.toList do
          queue := d :: queue
  unless badc.isEmpty do
    throwError "forbidden declarations (designated, or used by one): \\
{badc.toList.take 8}"
  IO.println "computability check OK"\
"""

_FORBID_NONCOMPUTABLE = """\
      if isNoncomputable env c then
        badc := badc.push (c, "noncomputable")\
"""

# A `partial def` elaborates to an opaque constant, so plain `opaque` is
# rejected along with it -- both are equally non-executable.
_FORBID_PARTIAL = """\
      if (env.find? c).any (· matches .opaqueInfo _) then
        badc := badc.push (c, "partial/opaque")\
"""

_CHECK_FOOTER = """\

if ! lake env lean "$DIR/Check.lean" >"$DIR/check.log" 2>&1; then
  cat "$DIR/check.log"
  fail "Lean-level goal checks failed (see above)"
fi
cat "$DIR/check.log"

echo "DONE: all designated sorries are filled and every restriction holds"
exit 0
"""


def build_check_script(spec: FillSorrySpec, *, base: str, subdir: str) -> str:
    """The check-goal script run in-container as the --ralph_done check.
    `base` is the chain_base commit off-limits paths are pinned to; `subdir`
    is the project's path inside the repo ('' at the root). The script runs
    with CWD = the project dir (sandbox.run_script), so lake commands need no
    cd; git commands use -C "$ROOT" so pattern paths are repo-relative."""
    modules = target_modules(spec)

    script = _CHECK_HEADER.replace("@GERBIL_BASE@", shlex.quote(base))

    if spec.off_limits:
        prefix = subdir + "/" if subdir else ""
        pats = [prefix + p for p in normalize_off_limits(spec.off_limits)]
        array = " ".join(shlex.quote(p) for p in pats)
        skips = [".gerbil/*"] + ([f"{subdir}/.gerbil/*"] if subdir else [])
        script += (
            _CHECK_OFF_LIMITS
            .replace("@GERBIL_OFF_LIMITS_ARRAY@", array)
            .replace("@GERBIL_SKIP@", "|".join(skips))
        )

    missing = [str(p) for p in spec.sorries if str(p) not in spec.decls]
    if missing:
        raise ValueError(
            "no enclosing declaration known for: " + ", ".join(missing)
            + " (run validate first, or name it in the spec)"
        )
    lean_decls = "[" + ", ".join(
        f'(`{module_of(p.file)}, "{spec.decls[str(p)]}")' for p in spec.sorries
    ) + "]"
    lean_axioms = "[" + ", ".join(f"`{a}" for a in spec.axioms) + "]"

    forbid_block = ""
    if "noncomputable" in spec.forbid or "partial" in spec.forbid:
        forbid_block = (
            _CHECK_FORBID_BLOCK
            .replace("@GERBIL_FORBID_NONCOMPUTABLE@\n",
                     _FORBID_NONCOMPUTABLE + "\n"
                     if "noncomputable" in spec.forbid else "")
            .replace("@GERBIL_FORBID_PARTIAL@\n",
                     _FORBID_PARTIAL + "\n"
                     if "partial" in spec.forbid else "")
        )
    # The library's modules are enumerated at CHECK time (a declaration may
    # move to a file created mid-task); generation only fixes which library
    # roots to look under -- the first path component of each designated
    # file, e.g. Hello/Basic.lean -> the `Hello.lean` + `Hello/*.lean`
    # tracked files. Git's plain pathspec `*` crosses `/`.
    prefix = f"{subdir}/" if subdir else ""
    roots = sorted({m.split(".")[0] for m in modules})
    pathspecs = " ".join(
        shlex.quote(p)
        for r in roots
        for p in (f"{prefix}{r}.lean", f"{prefix}{r}/*.lean")
    )
    script += (
        _CHECK_BUILD
        .replace("@GERBIL_LEAN_PATHSPECS@", pathspecs)
        .replace("@GERBIL_STRIP@", prefix)
        .replace("@GERBIL_ROOTS@", ", ".join(prefix + r for r in roots))
        .replace("@GERBIL_DECLS@", lean_decls)
        .replace("@GERBIL_ALLOWED_AXIOMS@", lean_axioms)
        .replace("@GERBIL_FORBID_BLOCK@\n",
                 forbid_block + "\n" if forbid_block else "")
    )

    script += _CHECK_FOOTER

    assert "@GERBIL_" not in script, "unexpanded marker in check script"
    return script


# The starting point `gerbil new-fill` opens in the user's editor: every
# spec key present, explained, and commented out except the one required
# field. Kept beside load_spec so the two cannot drift apart (the test suite
# asserts every key is mentioned).
SPEC_TEMPLATE = """\
# gerbil fill-sorry task spec: fill in `sorries`, save, exit.
# Paths are project-root-relative, positions 1-indexed. Docs: docs/fill-sorry.md

# Each entry: "FILE:LINE[:COL]" | "Ns.declName" | { pos = "...", decl = "..." }
sorries = [
]

# off_limits = ["Spec.lean", "Contracts/"]                 # frozen paths/globs
# axioms = ["propext", "Classical.choice", "Quot.sound"]   # allowed (= default)
# forbid = ["noncomputable", "partial"]                    # or "native_decide"
# approach = "Try induction on n."      # notes spliced into the prompt
# approach_file = "notes/approach.md"   # ...or read them from a file
# plan = "my-task.md"                   # plan-file name (in .gerbil/plans/)
# check_timeout = 1800                  # seconds the goal check may run
# ralph = 10                            # session budget (CLI --ralph overrides)
"""


# ---------------------------------------------------------------------------
# Session metadata
# ---------------------------------------------------------------------------


def spec_from_meta(meta: dict) -> FillSorrySpec:
    """Rebuild a spec from session_start's recorded metadata -- the inverse
    of session_meta, for `gerbil resume`. `approach` and `ralph` are absent
    on purpose: a resume replays the already-generated prompt from the log
    and continues the recorded chain, so neither is needed again. Tolerant
    of missing keys (a hand-edited or future log) the same way the rest of
    resume is."""
    sorries: list[SorryPos] = []
    decls: dict[str, str] = {}
    for entry in meta.get("sorries", []):
        (pos,) = parse_positions(entry["pos"])
        sorries.append(pos)
        if entry.get("decl"):
            decls[str(pos)] = entry["decl"]
    return FillSorrySpec(
        sorries=sorries,
        off_limits=list(meta.get("off_limits", [])),
        axioms=list(meta.get("axioms", [])) or list(STANDARD_AXIOMS),
        forbid=list(meta.get("forbid", [])),
        plan=meta.get("plan"),
        check_timeout=int(meta.get("check_timeout", DEFAULT_CHECK_TIMEOUT)),
        decls=decls,
    )


def session_meta(spec: FillSorrySpec, plan: str) -> dict:
    """What session_start records about the mode, and what `gerbil resume`
    needs to rehydrate it: the patch gate's off_limits, the plan file to
    re-upload, and the check timeout. (The check script itself is recorded
    separately, as ralph_done_script.)"""
    return {
        "sorries": [
            {"pos": str(p), "decl": spec.decls.get(str(p), "")}
            for p in spec.sorries
        ],
        "off_limits": list(spec.off_limits),
        "axioms": list(spec.axioms),
        "forbid": list(spec.forbid),
        "plan": plan,
        "check_timeout": spec.check_timeout,
    }
