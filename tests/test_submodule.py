"""Submodule handling: population, sanitization, and containment.

gerbil supports repos that *use* submodules, but the agent does no submodule
manipulation. Two halves, both tested here:

  - the container gets submodule contents in full, as if `git submodule update
    --init --recursive` had run, sourced from the host working tree (no network),
    with the same history stripping the superproject gets;
  - nothing the agent does to a submodule ever reaches the patch, because a patch
    physically cannot carry it -- format-patch renders a gitlink as one
    "Subproject commit <sha>" line, and the objects behind it die with the
    container.

Phase 1 (preflight) needs no Docker. Phase 2 does; it stubs the mathlib fetch.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

from gerbil.cli import _require_clean_submodules
from gerbil.sandbox import LeanSandbox, submodule_entries


def git(root: Path, *args: str, check: bool = True) -> str:
    """git in `root`, with an identity and local-path submodules allowed."""
    r = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t",
         "-c", "protocol.file.allow=always", *args],
        cwd=root, capture_output=True, text=True, check=check,
    )
    return r.stdout


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"submodule test failed at: {label}\n{detail}")


def make_repo(root: Path, name: str, content: str) -> Path:
    """A standalone one-file git repo, usable as a submodule source."""
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(content)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    return root


def make_super(tmp: Path) -> Path:
    """A Lake-ish repo with a submodule `dep`, which itself has a submodule
    `dep/inner` -- so recursion is exercised, not just the flat case."""
    inner = make_repo(tmp / "src-inner", "Inner.lean", "def inner := 1\n")
    dep = make_repo(tmp / "src-dep", "Dep.lean", "def dep := 2\n")
    git(dep, "submodule", "add", "-q", str(inner), "inner")
    git(dep, "commit", "-qm", "add inner")

    # A branch the sanitizer must strip from the *submodule's* history too.
    git(dep, "checkout", "-qb", "dep-secret")
    (dep / "SECRET.txt").write_text("submodule secret\n")
    git(dep, "add", "-A")
    git(dep, "commit", "-qm", "secret")
    git(dep, "checkout", "-q", "main")

    root = tmp / "super"
    root.mkdir()
    (root / "Main.lean").write_text("def main := 0\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "init")
    git(root, "submodule", "add", "-q", str(dep), "dep")
    git(root, "submodule", "update", "--init", "--recursive")
    git(root, "commit", "-qm", "add dep")
    return root


# ----------------------------------------------------------------------
# Phase 1 -- preflight (no Docker)
# ----------------------------------------------------------------------

def phase1() -> None:
    print("\n=== phase 1: preflight (no Docker) ===")
    with tempfile.TemporaryDirectory() as tmp:
        root = make_super(Path(tmp))

        entries = submodule_entries(root)
        check("enumerates submodules recursively, parents first",
              [p for p, _ in entries] == ["dep", "dep/inner"], repr(entries))
        check("records the gitlink sha",
              entries[0][1] == git(root, "rev-parse", "HEAD:dep").strip())

        _require_clean_submodules(root)
        print("[PASS] clean, fully-initialized submodules are accepted")

        # dirty: an uncommitted change inside the submodule
        (root / "dep" / "Dep.lean").write_text("def dep := 999\n")
        try:
            _require_clean_submodules(root)
            check("dirty submodule rejected", False, "no SystemExit raised")
        except SystemExit as e:
            check("dirty submodule rejected", "uncommitted changes" in str(e), str(e))
            check("dirty error names the submodule", "dep" in str(e), str(e))
        git(root / "dep", "checkout", "--", "Dep.lean")

        # moved: submodule HEAD no longer matches the recorded gitlink
        (root / "dep" / "Dep.lean").write_text("def dep := 3\n")
        git(root / "dep", "commit", "-qam", "moved")
        try:
            _require_clean_submodules(root)
            check("moved submodule rejected", False, "no SystemExit raised")
        except SystemExit as e:
            check("moved submodule rejected",
                  "different commit" in str(e), str(e))
            check("moved error gives the fix command",
                  "git submodule update --init --recursive" in str(e), str(e))

    # uninitialized: what a fresh `git clone` without --recursive leaves behind --
    # the submodule directory exists but is empty.
    with tempfile.TemporaryDirectory() as tmp:
        root = make_super(Path(tmp))
        shutil.rmtree(root / "dep")
        (root / "dep").mkdir()
        try:
            _require_clean_submodules(root)
            check("uninitialized submodule rejected", False, "no SystemExit raised")
        except SystemExit as e:
            check("uninitialized submodule rejected",
                  "not initialized" in str(e), str(e))
            check("uninitialized error gives the fix command",
                  "git submodule update --init --recursive" in str(e), str(e))

    # A repo with no submodules must sail through untouched.
    with tempfile.TemporaryDirectory() as tmp:
        plain = make_repo(Path(tmp) / "plain", "Main.lean", "def main := 0\n")
        check("no-submodule repo enumerates empty", submodule_entries(plain) == [])
        _require_clean_submodules(plain)
        print("[PASS] no-submodule repo passes preflight unchanged")


# ----------------------------------------------------------------------
# Phase 2 -- sandbox population and containment (needs Docker)
# ----------------------------------------------------------------------

def phase2() -> None:
    print("\n=== phase 2: sandbox population + containment (Docker) ===")
    LeanSandbox._fetch_mathlib_cache = lambda self: None

    with tempfile.TemporaryDirectory() as tmp:
        root = make_super(Path(tmp))
        dep_sha = git(root, "rev-parse", "HEAD:dep").strip()

        with LeanSandbox(project_dir=root) as sb:
            check("sandbox recorded both submodules",
                  sb.submodule_paths == ["dep", "dep/inner"],
                  repr(sb.submodule_paths))

            # --- population ---
            check("submodule contents uploaded",
                  sb.read_file("dep/Dep.lean") == "def dep := 2\n")
            check("nested submodule contents uploaded",
                  sb.read_file("dep/inner/Inner.lean") == "def inner := 1\n")
            r = sb.run("git status --porcelain")
            check("baseline working tree is clean", r.stdout.strip() == "", repr(r.stdout))
            check("submodule HEAD matches the recorded gitlink",
                  sb._git_at("dep", "rev-parse HEAD").stdout.strip() == dep_sha)
            r = sb._git_at("dep", "rev-parse --git-dir")
            check("submodule is a real repo in the container", r.exit_code == 0, repr(r.stderr))

            # --- sanitization reaches into submodules too ---
            r = sb._git_at("dep", "branch -a")
            check("submodule: secret branch stripped",
                  "dep-secret" not in r.stdout, repr(r.stdout))
            check("submodule: no remotes",
                  sb._git_at("dep", "remote").stdout.strip() == "")
            r = sb.run("cat dep/SECRET.txt")
            check("submodule: secret file absent", r.exit_code != 0, repr(r.stdout))

            base = sb.head()

            # --- the agent misbehaves in every way we care about ---
            sb.write_file("Main.lean", "def main := 42\n")          # legitimate work
            sb.write_file("dep/Dep.lean", "def dep := 111\n")        # edit in submodule
            sb.write_file("dep/inner/Inner.lean", "def inner := 7\n")
            sb.run("cd dep && git -c user.email=a@a -c user.name=a commit -qam 'agent'")
            sb.run("git config -f .gitmodules submodule.dep.branch hacked")
            # ...and invents a submodule of its own
            sb.run(
                "mkdir -p newdep && cd newdep && git init -q && echo x > x.txt && "
                "git add -A && git -c user.email=a@a -c user.name=a commit -qm new"
            )
            new_sha = sb.run("cd newdep && git rev-parse HEAD").stdout.strip()
            sb.run(f"git update-index --add --cacheinfo 160000,{new_sha},newdep")

            wip = sb.wip_patch(base)
            check("wip patch carries no submodule pointer",
                  "Subproject commit" not in wip, wip)

            sb.squash_commit(base, "session work")
            patch = sb.format_patch(base)

            check("patch carries the real work", "def main := 42" in patch, patch)
            check("patch has no submodule pointer",
                  "Subproject commit" not in patch, patch)
            check("patch does not touch .gitmodules",
                  ".gitmodules" not in patch, patch)
            check("patch does not carry submodule file contents",
                  "def dep := 111" not in patch and "def inner := 7" not in patch, patch)
            check("agent-invented submodule not exported",
                  "newdep" not in patch, patch)

            # Containment is index-only: the agent's files stay put, so anything
            # still reading them mid-session sees what it wrote.
            check("agent's submodule edit survives on disk",
                  sb.read_file("dep/Dep.lean") == "def dep := 111\n")

    # A submodule-only session must read as "nothing changed", not as an empty patch.
    with tempfile.TemporaryDirectory() as tmp:
        root = make_super(Path(tmp))
        with LeanSandbox(project_dir=root) as sb:
            base = sb.head()
            sb.write_file("dep/Dep.lean", "def dep := 5\n")
            sb.run("cd dep && git -c user.email=a@a -c user.name=a commit -qam x")
            check("submodule-only session commits nothing",
                  sb.squash_commit(base, "should not happen") is False)
            check("submodule-only session has an empty wip patch",
                  sb.wip_patch(base) == "")

    # And a repo without submodules behaves exactly as before.
    with tempfile.TemporaryDirectory() as tmp:
        plain = make_repo(Path(tmp) / "plain", "Main.lean", "def main := 0\n")
        with LeanSandbox(project_dir=plain) as sb:
            check("no-submodule sandbox records none", sb.submodule_paths == [])
            base = sb.head()
            sb.write_file("Main.lean", "def main := 1\n")
            sb.squash_commit(base, "work")
            check("no-submodule patch unaffected",
                  "def main := 1" in sb.format_patch(base))


def main() -> None:
    phase1()
    phase2()
    print("\nAll submodule tests passed.")


if __name__ == "__main__":
    main()
