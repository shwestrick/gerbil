"""--fill-sorry mode: the pure core.

Position/spec parsing, preflight validation, plan naming, the generated
prompt and check-goal script, the off-limits gate, and the session-metadata
round-trip. No Docker, no network -- the one external tool used is `git`
(scratch repos in temp dirs), which every gerbil environment has.
"""

import subprocess
import tempfile
from pathlib import Path

from gerbil import fill_sorry
from gerbil.fill_sorry import (
    DEFAULT_CHECK_TIMEOUT,
    NATIVE_DECIDE_AXIOMS,
    STANDARD_AXIOMS,
    SorryPos,
    build_check_script,
    build_prompt,
    gate_violations,
    load_spec,
    module_of,
    normalize_off_limits,
    parse_positions,
    plan_name,
    resolve_decl,
    spec_from_positions,
    target_modules,
    validate,
)


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"fill-sorry test failed at: {label}\n{detail}")


def rejects(label: str, fn, *args) -> None:
    try:
        fn(*args)
    except ValueError:
        check(label, True)
    else:
        check(label, False, "expected ValueError")


def spec_file(content: str) -> Path:
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "task.toml"
    p.write_text(content)
    return p


def git_project(files: dict[str, str]) -> Path:
    """A throwaway git repo with `files` committed (and nothing else)."""
    tmp = Path(tempfile.mkdtemp())
    for name, content in files.items():
        p = tmp / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    run = lambda *a: subprocess.run(
        ["git", "-C", str(tmp), *a], capture_output=True, check=True,
        env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
             "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp)},
    )
    run("init", "-q")
    run("add", "-A")
    run("commit", "-q", "-m", "init", "--no-verify")
    return tmp


LEAN = "theorem foo : 1 = 1 := by\n  sorry\n\ntheorem bar : 2 = 2 := sorry\n"


def test_positions() -> None:
    print("=== position parsing ===")

    check("FILE:LINE", parse_positions("Foo/Bar.lean:42")
          == [SorryPos("Foo/Bar.lean", 42)])
    check("FILE:LINE:COL", parse_positions("Foo/Bar.lean:42:7")
          == [SorryPos("Foo/Bar.lean", 42, 7)])
    check("comma list, whitespace",
          parse_positions(" A.lean:1 , B/C.lean:2:3 ")
          == [SorryPos("A.lean", 1), SorryPos("B/C.lean", 2, 3)])
    check("str() round-trips the zoom format",
          str(SorryPos("A.lean", 5, 2)) == "A.lean:5:2"
          and str(SorryPos("A.lean", 5)) == "A.lean:5")

    rejects("no line number", parse_positions, "Foo.lean")
    rejects("zero line", parse_positions, "Foo.lean:0")
    rejects("zero column", parse_positions, "Foo.lean:1:0")
    rejects("non-numeric", parse_positions, "a:b:c")
    rejects("absolute path", parse_positions, "/etc/x.lean:1")
    rejects("dotdot escape", parse_positions, "../x.lean:1")
    rejects("empty", parse_positions, " , ")


def test_resolver() -> None:
    print("\n=== position -> declaration resolution ===")

    def r(src: str, line: int):
        return resolve_decl(src.splitlines(), line)

    check("plain theorem",
          r("theorem foo : P := by\n  sorry\n", 2) == ("foo", ""))
    check("namespace qualifies",
          r("namespace A\ntheorem foo : P := sorry\nend A\n", 2)
          == ("A.foo", ""))
    check("nested namespaces",
          r("namespace A.B\nnamespace C\ndef d := sorry\n", 3)
          == ("A.B.C.d", ""))
    check("end pops the namespace",
          r("namespace A\nend A\ntheorem foo : P := sorry\n", 3)
          == ("foo", ""))
    check("section does not qualify but end pops it",
          r("namespace A\nsection\nend\ntheorem t : P := sorry\n", 4)
          == ("A.t", ""))
    check("dotted declaration name",
          r("namespace A\ndef B.c := sorry\n", 2) == ("A.B.c", ""))
    check("_root_ escapes the namespace",
          r("namespace A\ndef _root_.g := sorry\n", 2) == ("g", ""))
    check("modifiers and attributes skipped",
          r("@[simp]\nprivate noncomputable def h := sorry\n", 2)
          == ("h", ""))
    check("last declaration above wins",
          r("def a := 1\ntheorem b : P := by\n  sorry\n", 3) == ("b", ""))
    check("commented-out header ignored",
          r("theorem real : P := by\n/- theorem fake : Q := by -/\n  sorry\n",
            3) == ("real", ""))

    name, why = r("example : P := sorry\n", 1)
    check("example is a resolution error", name is None and "example" in why)
    name, why = r("instance : C := sorry\n", 1)
    check("anonymous instance is a resolution error",
          name is None and "anonymous" in why)
    name, why = r("-- just a comment\nsorry\n", 2)
    check("no header at all is a resolution error",
          name is None and "no declaration" in why)


def test_modules() -> None:
    print("\n=== module names ===")

    check("nested path", module_of("Foo/Bar/Baz.lean") == "Foo.Bar.Baz")
    check("root file", module_of("Basic.lean") == "Basic")
    spec = spec_from_positions(parse_positions("B.lean:1,A.lean:2,B.lean:9"))
    check("target_modules sorted + deduped", target_modules(spec) == ["A", "B"])


def test_spec() -> None:
    print("\n=== TOML spec ===")

    full = load_spec(spec_file(
        'sorries = ["Foo/Bar.lean:42", "Foo/Baz.lean:107:9"]\n'
        'off_limits = ["Spec.lean", "Contracts/"]\n'
        'axioms = ["propext"]\n'
        'forbid = ["noncomputable", "partial"]\n'
        'approach = "try induction"\n'
        'plan = "mytask"\n'
        'check_timeout = 600\n'
        'ralph = 5\n'
    ))
    check("full spec round-trips",
          full.sorries == [SorryPos("Foo/Bar.lean", 42),
                           SorryPos("Foo/Baz.lean", 107, 9)]
          and full.off_limits == ["Spec.lean", "Contracts/"]
          and full.axioms == ["propext"]
          and full.forbid == ["noncomputable", "partial"]
          and full.approach == "try induction"
          and full.plan == "mytask.md"
          and full.check_timeout == 600 and full.ralph == 5)

    minimal = load_spec(spec_file('sorries = ["A.lean:1"]\n'))
    check("defaults", minimal.axioms == STANDARD_AXIOMS
          and minimal.off_limits == [] and minimal.forbid == []
          and minimal.plan is None
          and minimal.check_timeout == DEFAULT_CHECK_TIMEOUT
          and minimal.ralph is None)

    tmp = spec_file('sorries = ["A.lean:1"]\napproach_file = "notes.md"\n')
    (tmp.parent / "notes.md").write_text("the notes\n")
    check("approach_file read relative to the spec",
          load_spec(tmp).approach == "the notes")

    tabled = load_spec(spec_file(
        'sorries = ["A.lean:1", { pos = "B.lean:2:3", decl = "Ns.tricky" }]\n'
    ))
    check("table entry pins the declaration",
          tabled.sorries == [SorryPos("A.lean", 1), SorryPos("B.lean", 2, 3)]
          and tabled.decls == {"B.lean:2:3": "Ns.tricky"})

    def bad(label: str, content: str, expect: str) -> None:
        try:
            load_spec(spec_file(content))
        except ValueError as exc:
            check(label, expect in str(exc), f"{expect!r} not in: {exc}")
        else:
            check(label, False, "expected ValueError")

    bad("unknown key", 'sorries = ["A.lean:1"]\noff_limit = []\n', "off_limit")
    bad("missing sorries", 'axioms = []\n', "sorries")
    bad("empty sorries", 'sorries = []\n', "sorries")
    bad("bad position", 'sorries = ["A.lean"]\n', "A.lean")
    bad("sorryAx allowed", 'sorries = ["A.lean:1"]\naxioms = ["sorryAx"]\n',
        "sorryAx")
    bad("unknown forbid", 'sorries = ["A.lean:1"]\nforbid = ["axioms"]\n',
        "forbid")
    bad("native_decide contradiction",
        'sorries = ["A.lean:1"]\nforbid = ["native_decide"]\n'
        f'axioms = ["propext", "{NATIVE_DECIDE_AXIOMS[0]}"]\n',
        "native_decide")
    bad("both approach forms",
        'sorries = ["A.lean:1"]\napproach = "x"\napproach_file = "y"\n',
        "not both")
    check("plan accepts its own qualified form",
          load_spec(spec_file(
              'sorries = ["A.lean:1"]\nplan = ".gerbil/plans/b.md"\n')).plan
          == "b.md")
    bad("plan with a directory",
        'sorries = ["A.lean:1"]\nplan = "docs/notes.md"\n', "bare")
    bad("plan escaping the project",
        'sorries = ["A.lean:1"]\nplan = "../b.md"\n', "bare")
    bad("bad check_timeout",
        'sorries = ["A.lean:1"]\ncheck_timeout = 0\n', "check_timeout")
    bad("bad ralph", 'sorries = ["A.lean:1"]\nralph = 0\n', "ralph")
    bad("sorries not a list", 'sorries = "A.lean:1"\n', "sorries")
    bad("table entry with unknown key",
        'sorries = [{ pos = "A.lean:1", name = "x" }]\n', "table entry")
    bad("table entry with a position list",
        'sorries = [{ pos = "A.lean:1,B.lean:2", decl = "x" }]\n',
        "one position")


def test_plan_name() -> None:
    print("\n=== plan naming ===")

    a = spec_from_positions(parse_positions("Foo/Bar.lean:42,Baz.lean:7"))
    b = spec_from_positions(parse_positions("Baz.lean:7,Foo/Bar.lean:42"))
    c = spec_from_positions(parse_positions("Foo/Bar.lean:43"))
    check("deterministic", plan_name(a) == plan_name(a))
    check("hash ignores listing order (same task)",
          plan_name(a).split("-")[-1] == plan_name(b).split("-")[-1])
    check("different task, different name", plan_name(a) != plan_name(c))
    check("always under .gerbil/plans/, slug from first file",
          plan_name(a).startswith(".gerbil/plans/fill-bar-"))
    check("override wins (qualified)",
          plan_name(fill_sorry.FillSorrySpec(
              sorries=a.sorries, plan="my.md")) == ".gerbil/plans/my.md")
    check("already-qualified override is not double-prefixed",
          plan_name(fill_sorry.FillSorrySpec(
              sorries=a.sorries, plan=".gerbil/plans/my.md"))
          == ".gerbil/plans/my.md")


def test_validate() -> None:
    print("\n=== preflight validation ===")

    proj = git_project({"A/B.lean": LEAN, "Spec.lean": "-- spec\n"})

    spec = spec_from_positions(parse_positions("A/B.lean:2,A/B.lean:4"))
    errors, warnings, excerpts = validate(
        spec, project_dir=proj, repo_root=proj)
    check("clean spec has no errors", errors == [], "\n".join(errors))
    check("excerpts extracted",
          excerpts[SorryPos("A/B.lean", 2)] == "sorry"
          and "bar" in excerpts[SorryPos("A/B.lean", 4)])
    check("declarations resolved",
          spec.decls == {"A/B.lean:2": "foo", "A/B.lean:4": "bar"},
          repr(spec.decls))

    spec = spec_from_positions(parse_positions("A/B.lean:2"))
    spec.decls["A/B.lean:2"] = "Explicit.name"
    validate(spec, project_dir=proj, repo_root=proj)
    check("explicit decl is not overwritten",
          spec.decls["A/B.lean:2"] == "Explicit.name")

    unresolvable = git_project({"E.lean": "example : True := by\n  sorry\n"})
    spec = spec_from_positions(parse_positions("E.lean:2"))
    errors, _, _ = validate(spec, project_dir=unresolvable,
                            repo_root=unresolvable)
    check("unresolvable declaration is an error (with spec hint)",
          any("enclosing declaration" in e and "decl =" in e for e in errors),
          "\n".join(errors))

    spec = spec_from_positions(parse_positions("Missing.lean:1"))
    errors, _, _ = validate(spec, project_dir=proj, repo_root=proj)
    check("missing file is an error", any("no such file" in e for e in errors))

    (proj / "Untracked.lean").write_text("sorry\n")
    spec = spec_from_positions(parse_positions("Untracked.lean:1"))
    errors, _, _ = validate(spec, project_dir=proj, repo_root=proj)
    check("untracked file is an error",
          any("not tracked" in e for e in errors))

    spec = spec_from_positions(parse_positions("A/B.lean:99"))
    errors, _, _ = validate(spec, project_dir=proj, repo_root=proj)
    check("line out of range is an error",
          any("only" in e for e in errors))

    spec = spec_from_positions(parse_positions("A/B.lean:2"))
    spec.off_limits = ["A/"]
    errors, _, _ = validate(spec, project_dir=proj, repo_root=proj)
    check("off-limits sorry file is an error",
          any("off_limits" in e for e in errors))

    spec = spec_from_positions(parse_positions("A/B.lean:1"))
    _, warnings, _ = validate(spec, project_dir=proj, repo_root=proj)
    check("line without sorry warns",
          any("does not contain" in w for w in warnings))

    spec = spec_from_positions(parse_positions("A/B.lean:2"))
    _, warnings, _ = validate(spec, project_dir=proj, repo_root=proj)
    check("undesignated sorries in the file get a note",
          any("not part of the goal" in w for w in warnings))

    spec = spec_from_positions(
        parse_positions("A/B.lean:2,A/B.lean:2,A/B.lean:4"))
    _, warnings, _ = validate(spec, project_dir=proj, repo_root=proj)
    check("duplicates deduped with a warning",
          any("duplicate" in w for w in warnings)
          and spec.sorries == [SorryPos("A/B.lean", 2), SorryPos("A/B.lean", 4)])

    # The plan file is ordinary tracked work product: a committed copy is a
    # continuing task, while the states that would silently break the memory
    # chain are errors.
    spec = spec_from_positions(parse_positions("A/B.lean:2"))
    name = plan_name(spec)
    cont = git_project({"A/B.lean": LEAN, name: "# earlier sessions\n"})
    errors, _, _ = validate(spec, project_dir=cont, repo_root=cont)
    check("tracked plan file is fine (continuing task)",
          errors == [], "\n".join(errors))

    spec = spec_from_positions(parse_positions("A/B.lean:2"))
    stray = git_project({"A/B.lean": LEAN})
    stray_plan = stray / plan_name(spec)
    stray_plan.parent.mkdir(parents=True)
    stray_plan.write_text("stray\n")
    errors, _, _ = validate(spec, project_dir=stray, repo_root=stray)
    check("untracked existing plan file is an error",
          any("not tracked" in e for e in errors))

    # .gerbil/ is conventionally gitignored; the plan is force-included past
    # that, so an ignoring repo must validate cleanly.
    spec = spec_from_positions(parse_positions("A/B.lean:2"))
    ignoring = git_project({"A/B.lean": LEAN, ".gitignore": ".gerbil/\n"})
    errors, _, _ = validate(spec, project_dir=ignoring, repo_root=ignoring)
    check("gitignored .gerbil/ is fine", errors == [], "\n".join(errors))

    # Project nested inside the repo: repo-relative tracking still resolves.
    root = git_project({"sub/A/B.lean": LEAN, "top.txt": "x\n"})
    spec = spec_from_positions(parse_positions("A/B.lean:2,A/B.lean:4"))
    errors, _, _ = validate(spec, project_dir=root / "sub", repo_root=root)
    check("nested project dir resolves tracked files",
          errors == [], "\n".join(errors))


def test_seed_plan() -> None:
    print("\n=== plan seed ===")

    spec = spec_from_positions(parse_positions("Foo/Bar.lean:42,Baz.lean:7"))
    spec.decls = {"Foo/Bar.lean:42": "Ns.foo", "Baz.lean:7": "baz"}
    seed = fill_sorry.seed_plan(spec)
    check("seed lists the designated sorries with declarations",
          "`Foo/Bar.lean:42` (in `Ns.foo`)" in seed
          and "`Baz.lean:7` (in `baz`)" in seed)
    check("seed states the append protocol",
          "Append a session entry" in seed
          and "no session has run yet" in seed)


def test_off_limits() -> None:
    print("\n=== off-limits matching + patch gate ===")

    norm = normalize_off_limits(["Spec.lean", "Contracts/", "*.axioms"])
    check("bare file matches itself",
          fill_sorry.is_off_limits("Spec.lean", norm))
    check("bare dir matches its contents",
          fill_sorry.is_off_limits("Contracts/A/B.lean", norm))
    check("glob matches", fill_sorry.is_off_limits("x/y.axioms", norm))
    check("unrelated path does not match",
          not fill_sorry.is_off_limits("Other.lean", norm))

    changed = ["Spec.lean", "Contracts/C.lean", "Mine.lean",
               ".gerbil/gerbil-1.jsonl"]
    check("gate flags exactly the violations",
          gate_violations(changed, ["Spec.lean", "Contracts/"], "")
          == ["Spec.lean", "Contracts/C.lean"])
    check("gate exempts .gerbil/",
          gate_violations([".gerbil/x.jsonl"], ["*"], "") == [])
    check("gate with nested project prefixes patterns",
          gate_violations(["sub/Spec.lean", "Spec.lean"], ["Spec.lean"], "sub")
          == ["sub/Spec.lean"])
    check("gate exempts nested .gerbil/",
          gate_violations(["sub/.gerbil/x.jsonl"], ["*"], "sub") == [])
    check("empty off_limits gates nothing",
          gate_violations(changed, [], "") == [])


def test_prompt() -> None:
    print("\n=== generated prompt ===")

    spec = spec_from_positions(parse_positions("Foo/Bar.lean:42,Foo/Baz.lean:7:2"))
    spec.decls = {"Foo/Bar.lean:42": "Ns.foo", "Foo/Baz.lean:7:2": "baz"}
    excerpts = {spec.sorries[0]: "theorem foo : P := sorry",
                spec.sorries[1]: "exact sorry"}
    plan_rel = ".gerbil/plans/fill-bar-abcd1234.md"

    p = build_prompt(spec, plan_rel=plan_rel, excerpts=excerpts)
    check("positions listed with declarations and excerpts",
          "`Foo/Bar.lean:42` (in `Ns.foo`) -- `theorem foo : P := sorry`" in p
          and "`Foo/Baz.lean:7:2` (in `baz`)" in p)
    check("plan path present", plan_rel in p)
    check("designated declarations named in the GOAL",
          "`Ns.foo`" in p and "`baz`" in p)
    check("designated-only scoping stated",
          "Fill only the designated sorries" in p
          and "do not block completion" in p)
    check("standard axioms named", "`Quot.sound`" in p)
    check("no off-limits section by default",
          "OFF-LIMITS" not in p and "whole repository is yours" in p)
    check("no approach section by default", "## APPROACH" not in p)
    check("native_decide named as disallowed", "native_decide" in p)

    spec.off_limits = ["Spec.lean"]
    spec.forbid = ["noncomputable"]
    spec.approach = "Use the Wf invariant."
    p = build_prompt(spec, plan_rel=plan_rel, excerpts=excerpts)
    check("off-limits section present",
          "OFF-LIMITS" in p and "`Spec.lean`" in p)
    check("forbid sentence present", "`noncomputable`" in p)
    check("approach spliced", "## APPROACH" in p and "Wf invariant" in p)

    spec.axioms = STANDARD_AXIOMS + NATIVE_DECIDE_AXIOMS
    p = build_prompt(spec, plan_rel=plan_rel, excerpts=excerpts)
    check("native_decide caveat dropped when its axioms are allowed",
          "including the ones `native_decide` introduces" not in p)


def test_check_script() -> None:
    print("\n=== generated check script ===")

    spec = spec_from_positions(parse_positions("Foo/Bar.lean:42"))
    try:
        build_check_script(spec, base="d", subdir="")
    except ValueError:
        check("unresolved declaration refuses to generate", True)
    else:
        check("unresolved declaration refuses to generate", False)

    spec.decls = {"Foo/Bar.lean:42": "Ns.foo"}
    s = build_check_script(spec, base="deadbeefcafe", subdir="")
    check("base embedded", "BASE=deadbeefcafe" in s)
    check("no off-limits sweep when unrestricted", "OFF_LIMITS" not in s)
    check("module built by name", "lake build +Foo.Bar" in s)
    check("module imported", "import Foo.Bar\n" in s)
    check("designated (module, decl) pairs in Lean list",
          '[(`Foo.Bar, "Ns.foo")]' in s)
    check("allowed axioms in Lean list",
          "[`propext, `Classical.choice, `Quot.sound]" in s)
    check("no computability block by default", "computability" not in s)
    check("no unexpanded markers", "@GERBIL_" not in s)
    check("exits 0 at DONE", s.rstrip().endswith("exit 0"))

    spec.off_limits = ["Spec.lean", "Contracts/", "with space.lean"]
    spec.forbid = ["noncomputable", "partial"]
    s = build_check_script(spec, base="deadbeefcafe", subdir="sub/dir")
    check("off-limits array quoted and dir-expanded",
          "sub/dir/Spec.lean" in s and "'sub/dir/Contracts/*'" in s
          and "'sub/dir/with space.lean'" in s)
    check("both .gerbil skips with a nested project",
          ".gerbil/*|sub/dir/.gerbil/*" in s)
    check("noncomputable check emitted", "isNoncomputable" in s)
    check("partial check emitted", "opaqueInfo" in s)
    check("forbid walks the dependency closure",
          "getUsedConstantsAsSet" in s)
    check("no unexpanded markers (full)", "@GERBIL_" not in s)

    spec.forbid = ["noncomputable"]
    s = build_check_script(spec, base="d", subdir="")
    check("partial check omitted when only noncomputable forbidden",
          "isNoncomputable" in s and "opaqueInfo" not in s)


def test_session_round_trip() -> None:
    print("\n=== session metadata round-trip ===")

    from gerbil.resume import parse_session
    from gerbil.session import Session

    spec = spec_from_positions(parse_positions("A.lean:1"))
    spec.off_limits = ["Spec.lean"]
    spec.decls = {"A.lean:1": "Ns.a"}
    meta = fill_sorry.session_meta(spec, ".gerbil/plans/fill-a-12345678.md")
    check("meta carries positions with declarations",
          meta["sorries"] == [{"pos": "A.lean:1", "decl": "Ns.a"}])

    tmp = Path(tempfile.mkdtemp())
    log = tmp / "s.jsonl"
    session = Session(
        path=log, model="m", project_dir=tmp, prompt_file=tmp / "p.md",
        base_commit="base", ralph={"iteration": 1, "total": 2,
                                   "chain_base": "base", "ancestors": []},
        ralph_done_script="#!/bin/sh\nexit 1\n", fill_sorry=meta,
    )
    session.record_turn("user", "the prompt")
    session.close()

    parsed = parse_session(log)
    check("fill_sorry dict round-trips", parsed.fill_sorry == meta)
    rebuilt = fill_sorry.spec_from_meta(parsed.fill_sorry)
    check("spec_from_meta inverts session_meta",
          rebuilt.sorries == spec.sorries
          and rebuilt.off_limits == spec.off_limits
          and rebuilt.axioms == spec.axioms
          and rebuilt.check_timeout == spec.check_timeout
          and rebuilt.decls == spec.decls
          and fill_sorry.plan_name(rebuilt)
          == ".gerbil/plans/fill-a-12345678.md")
    check("check script rides along untouched",
          parsed.ralph_done_script == "#!/bin/sh\nexit 1\n")
    check("ordinary sessions parse to None",
          parse_session_none(tmp).fill_sorry is None)


def parse_session_none(tmp: Path):
    from gerbil.resume import parse_session
    from gerbil.session import Session

    log = tmp / "plain.jsonl"
    Session(path=log, model="m", project_dir=tmp,
            prompt_file=tmp / "p.md").close()
    return parse_session(log)


def main() -> None:
    test_positions()
    test_resolver()
    test_modules()
    test_spec()
    test_plan_name()
    test_validate()
    test_seed_plan()
    test_off_limits()
    test_prompt()
    test_check_script()
    test_session_round_trip()
    print("\nAll fill-sorry tests passed.")


if __name__ == "__main__":
    main()
