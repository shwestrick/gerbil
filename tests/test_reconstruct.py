"""End-to-end test for `gerbil reconstruct-patch` (Docker-backed).

Builds a real git repo + a synthetic session log whose tool calls edit files,
runs cmd_reconstruct_patch, and checks the rebuilt patch captures the edits.
Run with: uv run python tests/test_reconstruct.py
"""

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

from gerbil import cli
from gerbil.sandbox import LeanSandbox


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


def make_repo(root: Path) -> str:
    """A clean git repo with a minimal Lake project. Returns the HEAD commit."""
    (root / "lakefile.toml").write_text("name = \"t\"\n")
    (root / "Hello.lean").write_text("def hello := 1\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root, check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    # Skip the slow mathlib fetch (we pass skip_cache anyway, but be safe).
    LeanSandbox._fetch_mathlib_cache = lambda self: None

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base = make_repo(root)

        # A session that edits Hello.lean and adds New.lean, then "completes".
        sessions = root / "sessions"
        sessions.mkdir()
        log = sessions / "gerbil-test-01.jsonl"
        events = [
            {"event": "session_start", "model": "claude-sonnet-4-6",
             "base_commit": base, "project_dir": str(root),
             "prompt_file": str(root / "p.md"), "gerbil_version": "test"},
            {"event": "turn", "role": "user", "content": "edit things"},
            {"event": "turn", "role": "assistant", "content": "editing"},
            {"event": "tool_call", "name": "edit_file",
             "args": {"path": "Hello.lean", "old_string": "1", "new_string": "42"}},
            {"event": "tool_result", "name": "edit_file", "result": "edited Hello.lean"},
            {"event": "tool_call", "name": "write_file",
             "args": {"path": "New.lean", "content": "def n := 7\n"}},
            {"event": "tool_result", "name": "write_file", "result": "wrote"},
            {"event": "tool_call", "name": "bash",
             "args": {"command": "echo 'def b := 9' > Bash.lean"}},
            {"event": "tool_result", "name": "bash", "result": ""},
            {"event": "turn", "role": "user",
             "content": "The task is complete. Here is the final git diff of all "
                        "your changes:\n\ndiff ..."},
            {"event": "turn", "role": "assistant",
             "content": "Add definitions\n\nAdds n and b, sets hello to 42."},
        ]
        log.write_text("".join(json.dumps(e) + "\n" for e in events))

        # reconstruct-patch also drops an archive copy in ~/.gerbil/sessions/.
        archived = Path.home() / ".gerbil" / "sessions" / "gerbil-test-01.patch"
        archived.unlink(missing_ok=True)

        args = SimpleNamespace(
            session_file=str(log), at=str(root), skip_cache=True, force=False,
        )
        try:
            cli.cmd_reconstruct_patch(args)
        finally:
            archived.unlink(missing_ok=True)  # don't litter the real home dir

        patch_path = root / ".gerbil" / "gerbil-test-01.patch"
        check("patch file written", patch_path.is_file())
        patch = patch_path.read_text()
        check("patch is non-empty", patch.strip() != "")
        check("patch has the edit_file change", "def hello := 42" in patch, patch)
        check("patch has the write_file change",
              "New.lean" in patch and "def n := 7" in patch, patch)
        check("patch has the bash-produced file",
              "Bash.lean" in patch and "def b := 9" in patch, patch)
        check("patch carries the recovered commit message",
              "Add definitions" in patch, patch[:400])

        # The reconstructed patch applies cleanly back onto the base commit.
        applies = subprocess.run(
            ["git", "apply", "--check", str(patch_path)], cwd=root,
            capture_output=True,
        )
        check("reconstructed patch applies on base", applies.returncode == 0,
              applies.stderr.decode())

        # The patch now exists; a second run without --force must refuse to
        # overwrite (no TTY to confirm), bailing before any sandbox work.
        before = patch_path.read_text()
        try:
            cli.cmd_reconstruct_patch(SimpleNamespace(
                session_file=str(log), at=str(root), skip_cache=True, force=False,
            ))
            check("refuses overwrite without --force / TTY", False)
        except SystemExit:
            check("refuses overwrite without --force / TTY", True)
        check("existing patch untouched when refused",
              patch_path.read_text() == before)

    print("\nreconstruct-patch end-to-end test passed.")


if __name__ == "__main__":
    main()
