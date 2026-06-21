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


def make_project(root: Path) -> None:
    (root / "Hello.lean").write_text("def hello := 1\n")
    (root / "README.md").write_text("# test project\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )


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

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
