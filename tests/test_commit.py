"""End-to-end test for `gerbil commit` (cli.cmd_commit).

Covers the full patch round trip with no LLM involved: a sandbox session on a
*contaminated* host repo (secret branch, tag, remote -- stripped from the
upload by _sanitized_git_dir), agent-style work squashed and format-patched
exactly as _finalize_session does, the patch dropped into .gerbil/, and then
`gerbil commit` run on the host: git am applies it, a re-run skips it via the
patch-id dedup, and the host repo's own git data survives untouched.

Needs Docker and the lean-sandbox image (the mathlib cache fetch is stubbed).
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from gerbil.sandbox import LeanSandbox


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"commit test failed at: {label}\n{detail}")


def host(*args: str, cwd: Path, ok: bool = True):
    r = subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd, capture_output=True, text=True,
    )
    if ok and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr}")
    return r


def gerbil_commit(root: Path):
    return subprocess.run(
        [sys.executable, "-m", "gerbil", "commit", "--at", str(root)],
        capture_output=True, text=True,
    )


def main() -> None:
    # Skip the slow mathlib fetch; we're testing the patch round trip.
    LeanSandbox._fetch_mathlib_cache = lambda self: None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        (root / "lakefile.lean").write_text("-- lakefile\n")
        (root / "Hello.lean").write_text("def hello := 1\n")
        host("init", "-q", "-b", "main", cwd=root)
        host("add", "-A", cwd=root)
        host("commit", "-qm", "init", cwd=root)
        # Contaminate like a real repo: a secret branch with its own commit, a
        # tag, and a remote. None of it enters the sandbox; all of it must
        # still be intact on the host after the round trip.
        host("checkout", "-qb", "secret", cwd=root)
        (root / "SECRET.txt").write_text("leak\n")
        host("add", "-A", cwd=root)
        host("commit", "-qm", "secret", cwd=root)
        host("tag", "v-secret", cwd=root)
        host("checkout", "-q", "main", cwd=root)
        host("remote", "add", "origin", "https://example.com/private.git", cwd=root)
        pre_head = host("rev-parse", "HEAD", cwd=root).stdout.strip()

        # "Session": an edit committed by the agent plus an uncommitted new
        # file, then squash + format-patch -- exactly what _finalize_session
        # does at the end of a run.
        with LeanSandbox(project_dir=root) as sb:
            base = sb.head()
            sb.write_file("Hello.lean", "def hello := 42\n")
            sb.commit("intermediate agent commit")
            sb.write_file("New.lean", "def n := 3\n")  # left uncommitted
            check("squash produced a commit",
                  sb.squash_commit(base, "prove the thing\n\nsession footer"))
            patch_text = sb.format_patch(base)

        out_dir = root / ".gerbil"
        out_dir.mkdir()
        (out_dir / "gerbil-260723-000000.patch").write_text(patch_text)

        r = gerbil_commit(root)
        check("gerbil commit exits 0", r.returncode == 0, r.stdout + r.stderr)
        check("patch was committed", "1 committed" in r.stdout, r.stdout)
        check("HEAD advanced",
              host("rev-parse", "HEAD", cwd=root).stdout.strip() != pre_head)
        check("edit landed on host",
              (root / "Hello.lean").read_text() == "def hello := 42\n")
        check("uncommitted file landed on host",
              (root / "New.lean").read_text() == "def n := 3\n")
        check("commit message preserved",
              host("log", "-1", "--format=%s", cwd=root).stdout.strip()
              == "prove the thing")
        status = host("status", "--porcelain", cwd=root).stdout
        dirt = [l for l in status.splitlines() if ".gerbil" not in l]
        check("host tree clean after am (bar the untracked .gerbil/)",
              dirt == [], status)
        check("exactly one commit added",
              len(host("rev-list", f"{pre_head}..HEAD", cwd=root).stdout.split()) == 1)

        # A second run must classify the same patch as already committed (the
        # stable patch-id check), not re-apply or fail on it.
        r = gerbil_commit(root)
        check("re-run skips as already committed",
              r.returncode == 0 and "1 already committed" in r.stdout,
              r.stdout + r.stderr)

        # The host repo's own git data is untouched by the whole round trip.
        check("host secret branch intact",
              host("rev-parse", "--verify", "secret", cwd=root, ok=False).returncode == 0)
        check("host tag intact", "v-secret" in host("tag", cwd=root).stdout)
        check("host remote intact", "origin" in host("remote", cwd=root).stdout)

    print("\ngerbil commit end-to-end test passed.")


if __name__ == "__main__":
    main()
