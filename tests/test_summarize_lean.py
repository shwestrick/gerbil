"""Summarize's lean-line accounting: per-session deltas and the running-total
anchor. No Docker, no network -- everything runs against throwaway host git
repos.

The two subtle cases, both observed in the wild on the same project:
- one commit carrying several session logs (sessions ported wholesale from
  another checkout) must not credit its whole diff to every log it carries --
  that counted the same 15k-line port once per session and inflated the
  running total by ~390k lines;
- a history rewrite (rebase/squash) leaves every recorded base_commit SHA
  dangling, so the running total anchors instead at the parent of the first
  commit a delta was attributed to -- measured from history as it is now,
  which survives any rewrite.
"""

import argparse
import contextlib
import io
import json
import subprocess
import tempfile
from pathlib import Path

from gerbil.cli import (
    _committed_lean_delta,
    _lean_line_count,
    _log_commit,
    _project_patches,
    _project_session_logs,
    cmd_summarize,
)


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"summarize lean test failed at: {label}\n{detail}")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def make_repo() -> Path:
    repo = Path(tempfile.mkdtemp())
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "test@test")
    git(repo, "config", "user.name", "test")
    return repo


def commit(repo: Path, message: str, files: dict[str, str]) -> str:
    """Write `files`, commit everything, return the commit SHA."""
    for name, content in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def session_log(base_commit: str) -> str:
    """A minimal completed-session log with the given recorded base."""
    return (
        json.dumps({"event": "session_start", "model": "test-model",
                    "base_commit": base_commit}) + "\n"
        + json.dumps({"event": "session_end"}) + "\n"
    )


def lean(n: int, tag: str = "x") -> str:
    return "".join(f"-- {tag} line {i}\n" for i in range(n))


def summarize(repo: Path) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        cmd_summarize(argparse.Namespace(at=str(repo)))
    return out.getvalue()


def row(output: str, session: str) -> str:
    return next(line for line in output.splitlines() if session in line)


DANGLING = "d" * 40  # a recorded base SHA that exists in no repo (post-rebase)


def test_helpers() -> None:
    print("=== delta helpers ===")
    repo = make_repo()
    rev0 = commit(repo, "base", {"A.lean": lean(10)})

    rev1 = commit(repo, "session 1", {
        ".gerbil/s1.jsonl": session_log(rev0),
        "A.lean": lean(10) + lean(5, "new"),          # +5 project lean lines
        ".lake/packages/dep/B.lean": lean(99, "dep"),  # excluded: not ours
        "notes.md": "not lean\n",
    })

    check("log commit found", _log_commit(repo, "s1.jsonl") == rev1)
    check("missing log has no commit", _log_commit(repo, "nope.jsonl") is None)
    check("single-log commit delta (project lean only)",
          _committed_lean_delta(repo, rev1) == (5, 0),
          str(_committed_lean_delta(repo, rev1)))

    # The wild case: one commit porting several sessions' logs at once. Its
    # diff belongs to no single session, so the delta must be None -- the old
    # behavior credited all 100 lines to BOTH logs.
    rev2 = commit(repo, "port two sessions from another checkout", {
        ".gerbil/s2.jsonl": session_log(DANGLING),
        ".gerbil/s3.jsonl": session_log(DANGLING),
        "Ported.lean": lean(100, "ported"),
    })
    check("multi-log commit refuses attribution",
          _committed_lean_delta(repo, rev2) is None)
    check("...but both logs still map to it",
          _log_commit(repo, "s2.jsonl") == rev2
          and _log_commit(repo, "s3.jsonl") == rev2)

    check("line count at base", _lean_line_count(repo, rev0) == 10)
    check("line count excludes .lake", _lean_line_count(repo, rev2) == 115,
          str(_lean_line_count(repo, rev2)))
    check("dangling commit not countable",
          _lean_line_count(repo, DANGLING) is None)

    # The subdirectory layout: logs folded at .gerbil/sessions/, patches at
    # .gerbil/patches/, with the flat legacy paths still readable.
    rev3 = commit(repo, "session 4", {
        ".gerbil/sessions/s4.jsonl": session_log(rev1),
        "A.lean": lean(10) + lean(5, "new") + lean(1, "more"),
    })
    check("log commit found under .gerbil/sessions/",
          _log_commit(repo, "s4.jsonl") == rev3)
    (repo / ".gerbil" / "patches").mkdir(parents=True, exist_ok=True)
    (repo / ".gerbil" / "patches" / "gerbil-02.patch").write_text("x")
    (repo / ".gerbil" / "gerbil-01.patch").write_text("y")
    (repo / ".gerbil" / "gerbil-02.patch").write_text("legacy dup")
    check("patches merged across layouts, patches/ wins a name clash",
          [(p.name, p.parent.name) for p in _project_patches(repo)]
          == [("gerbil-01.patch", ".gerbil"),
              ("gerbil-02.patch", "patches")],
          str(_project_patches(repo)))
    check("session logs merged across layouts",
          [p.name for p in _project_session_logs(repo)]
          == ["s1.jsonl", "s2.jsonl", "s3.jsonl", "s4.jsonl"],
          str(_project_session_logs(repo)))
    (repo / ".gerbil" / "patches" / "gerbil-02.patch").unlink()
    (repo / ".gerbil" / "gerbil-01.patch").unlink()
    (repo / ".gerbil" / "gerbil-02.patch").unlink()


def test_recorded_base_anchor() -> None:
    print("=== running total: recorded base commit intact ===")
    repo = make_repo()
    rev0 = commit(repo, "base", {"A.lean": lean(10)})
    commit(repo, "session 1", {
        ".gerbil/gerbil-01.jsonl": session_log(rev0),
        "A.lean": lean(10) + lean(4, "s1"),
    })
    out = summarize(repo)
    check("baseline anchored at recorded base",
          f"lean baseline: 10 lines at {rev0[:12]}" in out, out)
    check("absolute total from the first row",
          "+4/-0" in row(out, "gerbil-01") and "14" in row(out, "gerbil-01"),
          row(out, "gerbil-01"))


def test_rewritten_history_anchor() -> None:
    print("=== running total: rebased history + bulk-ported logs ===")
    repo = make_repo()
    commit(repo, "base", {"A.lean": lean(3)})
    # Two sessions ported wholesale (their recorded bases point into another
    # repo), then a real local session whose recorded base was rebased away.
    commit(repo, "port two sessions", {
        ".gerbil/gerbil-01.jsonl": session_log(DANGLING),
        ".gerbil/gerbil-02.jsonl": session_log(DANGLING),
        "Ported.lean": lean(50, "ported"),
    })
    rev2 = commit(repo, "session 3", {
        ".gerbil/gerbil-03.jsonl": session_log(DANGLING),
        "New.lean": lean(7, "s3"),
    })
    out = summarize(repo)
    check("baseline falls back to parent of first attributed commit",
          f"lean baseline: 53 lines just before {rev2[:12]}" in out, out)
    check("ported sessions get no delta and no total",
          row(out, "gerbil-01").rstrip().endswith("-           -")
          and row(out, "gerbil-02").rstrip().endswith("-           -"),
          row(out, "gerbil-01"))
    check("attributed session resumes the absolute total",
          "+7/-0" in row(out, "gerbil-03") and "60" in row(out, "gerbil-03"),
          row(out, "gerbil-03"))


def test_no_anchor_at_all() -> None:
    print("=== running total: nothing anchorable -> net change ===")
    repo = make_repo()
    commit(repo, "base", {"A.lean": lean(3)})
    # Logs present but never committed (e.g. hand-copied), sibling .patch too.
    gerbil_dir = repo / ".gerbil"
    gerbil_dir.mkdir()
    (gerbil_dir / "gerbil-01.jsonl").write_text(session_log(DANGLING))
    out = summarize(repo)
    check("net-change mode announced",
          "net change across sessions" in out, out)
    check("uncommitted log shows no delta",
          row(out, "gerbil-01").rstrip().endswith("-          +0"),
          row(out, "gerbil-01"))


def main() -> None:
    test_helpers()
    test_recorded_base_anchor()
    test_rewritten_history_anchor()
    test_no_anchor_at_all()
    print("\nAll summarize lean tests passed.")


if __name__ == "__main__":
    main()
