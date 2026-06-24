"""Smoke test for the Docker plumbing in sandbox.py.

Stubs out the mathlib cache fetch (slow, needs network + a mathlib project) and
exercises everything else: container lifecycle, file upload, read/write/edit
tools, bash, command timeout, and git diff.
"""

import subprocess
import tempfile
from pathlib import Path

from gerbil import tools
from gerbil.sandbox import LeanSandbox


def make_project(root: Path, subdir: str = "") -> Path:
    """Create a git repo at `root` with a (tiny) Lake-ish project at root/subdir.
    Returns the project directory. When subdir is "", the project is the repo root."""
    proj = root / subdir if subdir else root
    proj.mkdir(parents=True, exist_ok=True)
    (proj / "Hello.lean").write_text("def hello := 1\n")
    (root / "README.md").write_text("# test repo\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    return proj


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"smoke test failed at: {label}\n{detail}")


def main() -> None:
    # Skip the slow mathlib fetch; we're testing Docker plumbing here.
    LeanSandbox._fetch_mathlib_cache = lambda self: None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root)

        with LeanSandbox(project_dir=root) as sb:
            # 1. uploaded file is readable inside the container
            r = sb.read_file("Hello.lean")
            check("read uploaded file", r == "def hello := 1\n", repr(r))

            # 2. write a new file, read it back
            sb.write_file("Sub/New.lean", "theorem t : True := trivial\n")
            r = sb.read_file("Sub/New.lean")
            check("write+read nested file", r == "theorem t : True := trivial\n", repr(r))

            # 3. read_file tool
            res = tools.dispatch(sb, "read_file", {"path": "Hello.lean"})
            check("read_file tool", res.content == "def hello := 1\n" and not res.is_error)

            # 4. edit_file tool (unique match)
            res = tools.dispatch(
                sb, "edit_file",
                {"path": "Hello.lean", "old_string": "1", "new_string": "42"},
            )
            check("edit_file tool", not res.is_error, res.content)
            check("edit applied", sb.read_file("Hello.lean") == "def hello := 42\n")

            # 5. edit_file with missing string -> error
            res = tools.dispatch(
                sb, "edit_file",
                {"path": "Hello.lean", "old_string": "nope", "new_string": "x"},
            )
            check("edit_file missing string errors", res.is_error, res.content)

            # 6. bash tool, success
            res = tools.dispatch(sb, "bash", {"command": "echo hi && ls"})
            check("bash success", not res.is_error and "hi" in res.content, res.content)

            # 7. bash tool, nonzero exit
            res = tools.dispatch(sb, "bash", {"command": "exit 3"})
            check("bash nonzero exit flagged", res.is_error and "exit code: 3" in res.content, res.content)

            # 8. command timeout
            r = sb.run("sleep 5", timeout=1.0)
            check("timeout detected", r.timeout_occurred, repr(r))

            # 9. git diff reflects our edits + new file
            diff = sb.get_diff()
            check("diff shows edit", "def hello := 42" in diff, diff)
            check("diff shows new file", "Sub/New.lean" in diff, diff)

            # 10. resume plumbing: the working-tree patch + base commit fully
            # reconstruct the tree. Commit the changes, roll back to base with
            # checkout_force, then reapply the saved patch -- exactly what
            # `gerbil run --resume` does.
            base = sb.head()
            patch = sb.get_diff()          # edited Hello.lean + new Sub/New.lean
            sb.commit("wip")
            check("resume: commit advanced HEAD", sb.head() != base)

            sb.checkout_force(base)
            check("resume: checkout_force restores base",
                  sb.read_file("Hello.lean") == "def hello := 1\n")
            check("resume: checkout_force drops later file",
                  sb.run("test -f Sub/New.lean").exit_code != 0)

            sb.apply_diff(patch)
            check("resume: apply_diff restores edit",
                  sb.read_file("Hello.lean") == "def hello := 42\n")
            check("resume: apply_diff restores new file",
                  sb.read_file("Sub/New.lean") == "theorem t : True := trivial\n")

            # 11. ralph-resume plumbing: a chain rebuilds from a host-reachable
            # base by replaying each prior session's format-patch via git_am --
            # exactly how `--resume` reconstructs a mid-chain session's base.
            sb.commit("chain base")
            chain_base = sb.head()

            sb.write_file("Hello.lean", "def hello := 100\n")  # "session 1"
            sb.commit("c1")
            patch1 = sb.format_patch(chain_base)
            mid = sb.head()

            sb.write_file("Two.lean", "def two := 2\n")          # "session 2"
            sb.commit("c2")
            patch2 = sb.format_patch(mid)
            orig_tree = sb.run("git rev-parse HEAD^{tree}").stdout.strip()

            sb.checkout_force(chain_base)
            check("ralph: rolled back to chain base",
                  sb.read_file("Hello.lean") == "def hello := 42\n"
                  and sb.run("test -f Two.lean").exit_code != 0)

            sb.git_am(patch1)
            sb.git_am(patch2)
            check("ralph: git_am replays session 1",
                  sb.read_file("Hello.lean") == "def hello := 100\n")
            check("ralph: git_am replays session 2",
                  sb.read_file("Two.lean") == "def two := 2\n")
            # The replayed history reproduces the exact tree (commit shas differ,
            # which is fine -- gerbil format-patches the tree, not the shas).
            replayed_tree = sb.run("git rev-parse HEAD^{tree}").stdout.strip()
            check("ralph: replayed tree matches original",
                  replayed_tree == orig_tree, f"{replayed_tree} != {orig_tree}")

            # 12. large patches: git_am / apply_diff must stage the patch as a
            # file, not inline it on the command line (a single exec arg is capped
            # at ~128 KiB; real session patches exceed it -- the reported bug).
            pre_big = sb.head()
            big = "-- big\n" + ("abcdefgh " * 30_000) + "\n"  # ~270 KB
            sb.write_file("Big.lean", big)
            sb.commit("big file")
            big_patch = sb.format_patch(pre_big)
            check("big: format-patch exceeds the exec arg limit",
                  len(big_patch) > 200_000, str(len(big_patch)))

            sb.checkout_force(pre_big)
            check("big: rolled back", sb.run("test -f Big.lean").exit_code != 0)
            sb.git_am(big_patch)  # would raise "argument list too long" if inlined
            check("big: git_am applies a large patch", sb.read_file("Big.lean") == big)

            sb.checkout_force(pre_big)
            sb.write_file("Big.lean", big)
            big_diff = sb.get_diff()
            sb.checkout_force(pre_big)
            sb.apply_diff(big_diff)
            check("big: apply_diff applies a large patch",
                  sb.read_file("Big.lean") == big)

    # Lake project in a subdirectory of the repo (not the repo root).
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        proj = make_project(root, subdir="lean/proof")

        with LeanSandbox(project_dir=proj, repo_root=root) as sb:
            # file ops are relative to the Lake project (the subdir)
            r = sb.read_file("Hello.lean")
            check("subdir: read project file", r == "def hello := 1\n", repr(r))

            # the rest of the repo came along (uploaded from repo root)
            r = sb.run("cat /workspace/project/README.md")
            check("subdir: repo root uploaded", "# test repo" in r.stdout, repr(r.stdout))

            # the subdir is writable by the sandbox user (lake creates .lake here)
            r = sb.run("mkdir -p .lake/packages && echo ok")
            check("subdir: writable by sandbox user", r.exit_code == 0 and "ok" in r.stdout,
                  repr((r.exit_code, r.stderr)))

            # edits + diff still work; diff paths are relative to the repo root
            sb.write_file("Hello.lean", "def hello := 99\n")
            diff = sb.get_diff()
            check("subdir: diff shows edit", "def hello := 99" in diff, diff)
            check("subdir: diff path under subdir", "lean/proof/Hello.lean" in diff, diff)

            # Regression: an agent running `git init` in the Lake subdir creates a
            # nested .git that hijacks plain git discovery. gerbil's own git is
            # pinned to the real repo, so it must be immune (the bug was a silent
            # 0-byte patch when format-patch ran against the nested repo).
            real_head = sb.head()
            sb.run("git init -q")  # the destructive command, as the agent ran it
            check("tamper: nested .git hijacks plain git",
                  sb.run("git rev-parse --show-toplevel").stdout.strip()
                  .endswith("lean/proof"))
            check("tamper: pinned head ignores nested repo", sb.head() == real_head)
            diff = sb.get_diff()
            check("tamper: pinned diff still sees real edit",
                  "def hello := 99" in diff and "lean/proof/Hello.lean" in diff, diff)
            sb.commit("real commit")
            patch = sb.format_patch(real_head)
            check("tamper: format_patch works against real base despite nested .git",
                  patch.strip() != "" and "lean/proof/Hello.lean" in patch, patch[:120])

    # The live resume snapshot (wip_patch) must capture changes the agent
    # committed itself, not just uncommitted ones -- the reported bug was that it
    # diffed against HEAD and silently dropped the agent's own commits.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root)
        with LeanSandbox(project_dir=root) as sb:
            base = sb.head()
            # The agent makes an INTERNAL commit...
            sb.write_file("Committed.lean", "def c := 1\n")
            sb.run("git add -A && git -c user.email=a@a -c user.name=a "
                   "commit -qm 'agent internal commit'")
            head_after_commit = sb.head()
            # ...and also leaves uncommitted + untracked changes.
            sb.write_file("Hello.lean", "def hello := 999\n")
            sb.write_file("Untracked.lean", "def u := 2\n")

            wip = sb.wip_patch(base)
            check("wip: snapshot is non-empty", wip.strip() != "")
            check("wip: snapshot did not move HEAD", sb.head() == head_after_commit)

            # Reconstruct from a clean base, exactly as --resume does (git apply).
            sb.checkout_force(base)
            check("wip: rollback drops the internal commit",
                  sb.run("test -f Committed.lean").exit_code != 0)
            sb.apply_diff(wip)
            check("wip: restores internally-committed file",
                  sb.read_file("Committed.lean") == "def c := 1\n")
            check("wip: restores uncommitted edit",
                  sb.read_file("Hello.lean") == "def hello := 999\n")
            check("wip: restores untracked file",
                  sb.read_file("Untracked.lean") == "def u := 2\n")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
