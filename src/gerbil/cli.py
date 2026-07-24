#!/usr/bin/env python3
"""gerbil -- a sandboxed Lean theorem-proving agent.

Usage:
    gerbil run [--prompt FILE] [--at DIRECTORY] [--ralph N] ...
    gerbil resume SESSION_FILE [--at DIRECTORY] ...
    gerbil commit [--at DIRECTORY]

(Normally invoked through the `gerbil` launcher, which also provides `update`
and `--version`. --at defaults to the current directory.)

The agent works on the real repository (full history) inside the container.
Each session's changes -- including any intermediate commits the agent made
itself -- are squashed into a single commit, and the result is emitted as a
git format-patch: a single .patch file holding the commit title, message, and
diff, committed on the host with `gerbil commit` (git am).

Outputs:
    ~/.gerbil/sessions/gerbil-TIMESTAMP.jsonl  the live session log (the true
                                         session file, written as the run proceeds)
    <project>/.gerbil/gerbil-TIMESTAMP.patch   git format-patch of the session's
                                         commit(s); apply with `git am`
The patch is also copied to ~/.gerbil/sessions/, so the archive holds all gerbil
data. The session log is also folded into the commit (and thus the patch) by
default; pass --omit-session-log to keep it out of the commit. Either way the
~/.gerbil/sessions/ archive keeps the log.

With --ralph N, N sessions run back-to-back on the same prompt (reusing the
sandbox); outputs are numbered gerbil-TIMESTAMP-NN and each session builds on
the previous one's commit.
"""

import argparse
import collections
import contextlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .agent import DEFAULT_INNER_MAX_TURNS, run_session
from .pricing import (
    MODEL_PRICING,
    estimate_cost,
    model_pricing,
    pricing_match,
)
from .sandbox import LeanSandbox
from .session import Session
from .render import style
from .tools import Toolset


DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _resolve_at(at: str | None) -> Path:
    """The Lake project to operate on. Defaults to the directory the launcher was
    invoked from (GERBIL_CWD), falling back to the current working directory."""
    base = at or os.environ.get("GERBIL_CWD") or os.getcwd()
    return Path(base).resolve()


def _require_git_repo(project_dir: Path) -> Path:
    """Exit with a clear message unless project_dir is inside a git repo, and
    return the repo's toplevel directory. gerbil works on the real repository
    (uploads .git, commits, format-patch); the Lake project may live anywhere
    inside it, not just at the root."""
    result = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            f"error: {project_dir} is not inside a git repository.\n"
            "gerbil needs the Lake project to live in a git repo -- run "
            "`git init` (and make a commit) first."
        )
    toplevel = Path(result.stdout.strip()).resolve()
    # Require at least one commit. A freshly `git init`'d repo (e.g. what
    # `lake new` leaves behind) has no HEAD, so gerbil has no base to diff
    # against and would silently produce an empty patch.
    head = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "--verify", "-q", "HEAD"],
        capture_output=True,
        text=True,
    )
    if head.returncode != 0:
        sys.exit(
            f"error: the git repo at {toplevel} has no commits yet.\n"
            "gerbil works against a base commit -- make an initial commit first:\n"
            "  git add -A && git commit -m 'initial commit'"
        )
    return toplevel


def _require_lake_project(project_dir: Path) -> None:
    """Exit unless project_dir is a Lake project root (has a lakefile)."""
    if not (
        (project_dir / "lakefile.toml").is_file()
        or (project_dir / "lakefile.lean").is_file()
    ):
        sys.exit(
            f"error: {project_dir} is not a Lake project "
            "(no lakefile.toml or lakefile.lean found)."
        )


def _require_clean_worktree(repo_root: Path) -> None:
    """Exit unless the working tree is clean. gerbil runs on top of a clean
    commit -- it uploads only tracked files (== HEAD) and commits the agent's
    changes on top. Uncommitted changes to tracked files would otherwise be
    swept into the agent's commit. Untracked files are ignored (not uploaded).
    Checked across the whole repo, since gerbil uploads and commits the repo."""
    dirty = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        sys.exit(
            f"error: {repo_root} has uncommitted changes.\n"
            "gerbil runs on top of a clean commit. Commit or stash them first:\n"
            "  git stash       # then re-run gerbil, and `git stash pop` after"
        )


_DOCKER_PERMISSION_HELP = """\
error: cannot connect to Docker -- permission denied.

Docker must be usable without sudo (gerbil talks to the daemon via the Docker
SDK, which cannot use sudo). To fix:
  - add yourself to the docker group:  sudo usermod -aG docker $USER
    then log out and back in (or run: newgrp docker), and verify with:
    docker run hello-world
  - or set up rootless Docker: https://docs.docker.com/engine/security/rootless/"""

_DOCKER_DAEMON_HELP = """\
error: cannot connect to the Docker daemon -- is it running?
Start it (e.g. `sudo systemctl start docker`) or open Docker Desktop, then \
retry."""


def _require_docker() -> None:
    """Exit with actionable guidance unless the Docker daemon is reachable."""
    import docker

    try:
        docker.from_env().ping()
        return
    except Exception as exc:
        detail = str(exc)
        msg = detail.lower()

    if "permission denied" in msg:
        sys.exit(_DOCKER_PERMISSION_HELP)
    if "connect" in msg or "daemon" in msg:
        sys.exit(_DOCKER_DAEMON_HELP)
    sys.exit(f"error: Docker is not usable: {detail}")


def main() -> None:
    # SIGTERM (kill, service managers) and SIGHUP (closed terminal) would end
    # the process without unwinding the `with` blocks that own the sandbox
    # container and MCP client -- the container would leak, running `sleep
    # infinity` forever. Convert them to SystemExit so every context manager
    # tears down on the way out; 128+N is the conventional exit code. SIGINT
    # already arrives as KeyboardInterrupt and needs no help.
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, lambda n, _frame: sys.exit(128 + n))
        except (ValueError, OSError):
            pass  # unsupported signal, or not the main thread

    parser = argparse.ArgumentParser(
        prog="gerbil",
        description="Sandboxed Lean theorem-proving agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a gerbil session on a Lake project")
    run_p.add_argument(
        "--at",
        metavar="DIRECTORY",
        help="Path to the Lean/Lake project (a git repo). Default: current dir.",
    )
    run_p.add_argument(
        "--prompt",
        metavar="FILE",
        help="Path to a file containing the task description (required). To "
        "continue a crashed session instead, use `gerbil resume`.",
    )
    run_p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"LLM to use (default: {DEFAULT_MODEL}). Provider is auto-detected. "
            f"Use `ollama:<NAME>` for a local model served by ollama (e.g. "
            f"ollama:qwen2.5-coder); gerbil starts the server if one isn't running. "
            f"Use `portkey:<MODEL>` (or a bare @provider/model catalog name) to "
            f"route through a Portkey AI gateway; set PORTKEY_API_KEY and, for a "
            f"self-hosted gateway, PORTKEY_BASE_URL. "
            f"Known cloud models: {', '.join(MODEL_PRICING)}."
        ),
    )
    run_p.add_argument(
        "--small-model",
        metavar="MODEL",
        default=None,
        help="Enable big-small mode: --model (the big model) drives the session "
        "and gets a zoom_in tool that hands the mechanical details of a single "
        "sorry to this smaller model, which works in a focused sub-session "
        "until it calls zoom_out with a summary. Provider is auto-detected, "
        "independently of --model.",
    )
    run_p.add_argument(
        "--zoom-max-turns",
        type=int,
        metavar="N",
        default=None,
        help=f"Turn cap for each zoomed-in sub-session (default: "
        f"{DEFAULT_INNER_MAX_TURNS}). On hitting it the small model gets one "
        "final forced turn to report a mandatory zoom_out summary back to the "
        "big model. Requires --small-model.",
    )
    run_p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Safety cap on agent turns (default: unlimited, runs until done).",
    )
    run_p.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip 'lake exe cache get' at startup (faster, but mathlib will "
        "rebuild from source on first use).",
    )
    run_p.add_argument(
        "--no-mcp",
        dest="mcp",
        action="store_false",
        help="Disable the lean-lsp MCP tools; use only the built-in tools.",
    )
    run_p.add_argument(
        "--ralph",
        type=int,
        metavar="N",
        default=None,
        help="Run N sessions back-to-back on the same prompt, reusing the "
        "sandbox. Each session is committed inside the container so the next "
        "builds on it; outputs are numbered gerbil-<ts>-NN.{jsonl,patch}.",
    )
    run_p.add_argument(
        "--ralph_done",
        metavar="SCRIPT",
        help="Path to a script run inside the container after each --ralph "
        "session (on that session's committed working tree, from the project "
        "dir). Exit code 0 ends the ralph loop; non-zero continues. Requires "
        "--ralph.",
    )
    run_p.add_argument(
        "--omit-session-log",
        action="store_true",
        help="Do not include the session .jsonl log in the commit (by default "
        "it is folded in via commit --amend before the patch is produced). "
        "The log is always archived in ~/.gerbil/sessions/ regardless.",
    )
    run_p.set_defaults(func=cmd_run)

    resume_p = sub.add_parser(
        "resume",
        help="continue a crashed/incomplete session from its .jsonl log",
    )
    resume_p.add_argument(
        "session_file",
        metavar="SESSION_FILE",
        help="The session .jsonl log to resume: recreate its git base, replay the "
        "log, and continue. The model and prompt are taken from the log.",
    )
    resume_p.add_argument(
        "--at",
        metavar="DIRECTORY",
        help="Path to the Lean/Lake project (a git repo). Default: the project "
        "recorded in the session.",
    )
    resume_p.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Safety cap on agent turns (default: unlimited, runs until done).",
    )
    resume_p.add_argument(
        "--zoom-max-turns",
        type=int,
        metavar="N",
        default=None,
        help="Override the resumed session's per-zoom turn cap (by default the "
        "cap recorded in the log is reused). Only applies to a big-small "
        "(--small-model) session; the small model itself always comes from "
        "the log, like the model and prompt.",
    )
    resume_p.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip 'lake exe cache get' at startup.",
    )
    resume_p.add_argument(
        "--no-mcp",
        dest="mcp",
        action="store_false",
        help="Disable the lean-lsp MCP tools; use only the built-in tools.",
    )
    resume_p.add_argument(
        "--ralph_done",
        metavar="SCRIPT",
        help="Override the resumed ralph chain's termination check (by default the "
        "script recorded in the session log is reused). Only applies to a resumed "
        "--ralph chain.",
    )
    resume_p.add_argument(
        "--omit-session-log",
        action="store_true",
        help="Do not include the session .jsonl log in the commit. The resumed "
        "session's recorded setting is inherited by default; this flag forces "
        "the log out. It is always archived in ~/.gerbil/sessions/ regardless.",
    )
    resume_p.set_defaults(func=cmd_resume)

    commit_p = sub.add_parser(
        "commit", help="git am the patches gerbil produced into the repo, in order"
    )
    commit_p.add_argument(
        "--at",
        metavar="DIRECTORY",
        help="Path to the Lean/Lake project (a git repo). Default: current dir.",
    )
    commit_p.set_defaults(func=cmd_commit)

    summ_p = sub.add_parser(
        "summarize",
        help="report token usage, estimated cost, and tool stats across the "
        "project's .gerbil/*.jsonl session logs, with a per-session breakdown "
        "tracking cost and lean code growth",
    )
    summ_p.add_argument(
        "--at",
        metavar="DIRECTORY",
        help="Path to the Lean/Lake project (a git repo). Default: current dir.",
    )
    summ_p.set_defaults(func=cmd_summarize)

    recon_p = sub.add_parser(
        "reconstruct-patch",
        help="rebuild a session's .patch by replaying its tool calls in a sandbox",
    )
    recon_p.add_argument(
        "session_file",
        metavar="SESSION_FILE",
        help="The session .jsonl log whose patch should be reconstructed.",
    )
    recon_p.add_argument(
        "--at",
        metavar="DIRECTORY",
        help="Path to the Lean/Lake project (a git repo). Default: the project "
        "recorded in the session.",
    )
    recon_p.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip 'lake exe cache get' at startup.",
    )
    recon_p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing .patch without asking for confirmation.",
    )
    recon_p.set_defaults(func=cmd_reconstruct_patch)

    args = parser.parse_args()
    args.func(args)


def _patch_id(repo_dir: Path, text: str) -> str | None:
    """The (stable) patch-id of a diff/format-patch -- a content hash that is
    independent of commit hash, line offsets, and surrounding commits."""
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "patch-id", "--stable"],
        input=text,
        capture_output=True,
        text=True,
    ).stdout.split()
    return out[0] if out else None


def _committed_patch_ids(repo_dir: Path) -> set[str]:
    """Patch-ids of every commit reachable from HEAD, to detect patches that have
    already been applied (regardless of later commits that touched the area)."""
    log = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "-p", "--no-color"],
        capture_output=True,
        text=True,
    ).stdout
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "patch-id", "--stable"],
        input=log,
        capture_output=True,
        text=True,
    ).stdout
    return {line.split()[0] for line in out.splitlines() if line.split()}


def _patch_applies(repo_dir: Path, patch: Path) -> bool:
    """Whether `patch` applies cleanly to the current tree. Run from the repo
    root, since format-patch paths are relative to it."""
    return subprocess.run(
        ["git", "-C", str(repo_dir), "apply", "--check", str(patch)],
        capture_output=True,
    ).returncode == 0


def cmd_commit(args) -> None:
    """Commit each .gerbil/gerbil-*.patch (a git format-patch) in order via git am.

    The .gerbil/ directory may hold stale patches. Each is classified first:
    already-committed (its patch-id is in history) => skip; applies cleanly =>
    new (git am); otherwise out of date => skip with a warning."""
    project_dir = _resolve_at(args.at)
    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    repo_root = _require_git_repo(project_dir)
    _require_lake_project(project_dir)

    # Patches are stored next to the Lake project, but their paths are relative to
    # the repo root, so all git operations run from there.
    out_dir = project_dir / ".gerbil"
    patches = sorted(out_dir.glob("gerbil-*.patch"))
    if not patches:
        sys.exit(f"no patches found in {out_dir}")

    committed = _committed_patch_ids(repo_root)
    applied = already = stale = 0
    for patch in patches:
        pid = _patch_id(repo_root, patch.read_text())
        if pid and pid in committed:
            print(f"{style('skip:', 'bold', 'gray')}      {patch.name} (already committed)")
            already += 1
        elif _patch_applies(repo_root, patch):
            print(f"{style('committing:', 'bold')} {patch.name}")
            result = subprocess.run(["git", "am", str(patch)], cwd=repo_root)
            if result.returncode != 0:
                sys.exit(
                    f"git am failed on {patch.name}. Resolve and `git am --continue`, "
                    "or `git am --abort` to back out."
                )
            applied += 1
            if pid:
                committed.add(pid)  # so a duplicate patch later also skips
        else:
            print(
                f"{style('skip:', 'bold', 'yellow')}      {patch.name} "
                "(out of date; does not apply)",
                file=sys.stderr,
            )
            stale += 1

    print(style(
        f"done -- {applied} committed, {already} already committed, "
        f"{stale} out of date",
        "bold",
    ))


def _scan_session(path: Path, count_replayed: bool = False) -> dict:
    """Tally one session log into a stats dict. Replayed events (re-emitted from a
    prior log by --resume) are skipped by default so a resumed chain isn't
    double-counted -- this mirrors how Session accumulates totals only from live
    turns. `count_replayed` folds them in instead: used when the parent log the
    events came from is NOT among the scanned logs (a crashed session never
    commits, so its log never reaches the project's .gerbil/ -- the replay
    inside its continuation is the only record of that spend).

    Returns: {model, small_model, base_commit, input_tokens, output_tokens,
    thinking_tokens, cache_read_tokens, cache_write_tokens, turns,
    zoom_input_tokens, zoom_output_tokens, zoom_thinking_tokens,
    zoom_cache_read_tokens, zoom_cache_write_tokens, zoom_turns,
    tool_calls (Counter), status} where status is 'completed' | 'errored' |
    'incomplete', thinking_tokens is the reasoning subset of output_tokens, and
    the cache counts are additional prompt tokens (input_tokens is the uncached
    remainder; pre-caching logs simply lack the fields and read as zero). In a
    big-small session, zoom-tagged turns (the small model's sub-sessions) land
    in the zoom_* counters so they can be priced at the small model's rates;
    the plain counters hold only the big model's share. turns/zoom_turns count
    assistant turns only, matching the live "--- turn N ---" counters (user
    turn events -- the prompt, nudges, the commit-message request -- carry no
    model response). A garbled log yields a sentinel with status 'unreadable'
    and zero usage."""
    model = "unknown"
    small_model = None
    resumed_from = None
    base_commit = ""
    input_tokens = output_tokens = thinking_tokens = turns = 0
    cache_read_tokens = cache_write_tokens = 0
    zoom_input_tokens = zoom_output_tokens = zoom_thinking_tokens = 0
    zoom_cache_read_tokens = zoom_cache_write_tokens = zoom_turns = 0
    tool_calls: collections.Counter = collections.Counter()
    status = "incomplete"  # no session_end/error recorded => crashed mid-run

    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {"model": model, "small_model": small_model,
                "resumed_from": resumed_from,
                "base_commit": base_commit, "input_tokens": 0,
                "output_tokens": 0, "thinking_tokens": 0, "cache_read_tokens": 0,
                "cache_write_tokens": 0, "turns": 0,
                "zoom_input_tokens": 0, "zoom_output_tokens": 0,
                "zoom_thinking_tokens": 0, "zoom_cache_read_tokens": 0,
                "zoom_cache_write_tokens": 0, "zoom_turns": 0,
                "tool_calls": tool_calls, "status": "unreadable"}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            # A partial trailing line from an interrupted write; ignore it.
            continue
        # Replayed events were already counted in the original session's log
        # (when that log is present -- see count_replayed above).
        if e.get("replayed") and not count_replayed:
            continue
        event = e.get("event")
        if event == "session_start":
            model = e.get("model", model)
            small_model = e.get("small_model", small_model)
            resumed_from = e.get("resumed_from", resumed_from)
            base_commit = e.get("base_commit", base_commit)
        elif event == "turn":
            usage = e.get("usage") or {}
            # Count only assistant turns, so the tally matches the live
            # "--- turn N ---" / "--- zoom turn k/N ---" counters (one per
            # model response). User-role turn events (the prompt, nudges, the
            # commit-message request) still contribute their usage -- always
            # zero today, but that is an accident of who records them.
            is_assistant = e.get("role") == "assistant"
            if e.get("zoom"):
                zoom_input_tokens += usage.get("input_tokens", 0)
                zoom_output_tokens += usage.get("output_tokens", 0)
                zoom_thinking_tokens += usage.get("thinking_tokens", 0)
                zoom_cache_read_tokens += usage.get("cache_read_tokens", 0)
                zoom_cache_write_tokens += usage.get("cache_write_tokens", 0)
                if is_assistant:
                    zoom_turns += 1
            else:
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                thinking_tokens += usage.get("thinking_tokens", 0)
                cache_read_tokens += usage.get("cache_read_tokens", 0)
                cache_write_tokens += usage.get("cache_write_tokens", 0)
                if is_assistant:
                    turns += 1
        elif event == "tool_call":
            # Inner (zoom-tagged) calls count too: they really ran. zoom_in /
            # zoom_out showing up alongside them is desirable visibility.
            tool_calls[e.get("name", "?")] += 1
        elif event == "session_end":
            status = "completed"
        elif event == "error":
            status = "errored"

    return {"model": model, "small_model": small_model,
            "resumed_from": resumed_from,
            "base_commit": base_commit,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens, "thinking_tokens": thinking_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_write_tokens": cache_write_tokens,
            "turns": turns,
            "zoom_input_tokens": zoom_input_tokens,
            "zoom_output_tokens": zoom_output_tokens,
            "zoom_thinking_tokens": zoom_thinking_tokens,
            "zoom_cache_read_tokens": zoom_cache_read_tokens,
            "zoom_cache_write_tokens": zoom_cache_write_tokens,
            "zoom_turns": zoom_turns,
            "tool_calls": tool_calls, "status": status}


def _scan_sessions(logs: list[Path]) -> list[dict]:
    """Scan a set of session logs with resume-aware accounting.

    A continuation log carries its parent's full pre-crash history as replayed
    events. When the parent log is among `logs`, those events are skipped (the
    parent's own row already counts them); when it is absent, they are folded
    into the continuation's row -- the parent crashed, so it never committed
    and its log never reached this directory, making the replay the only
    record of that spend. In practice a parent that IS present is one that
    completed (an already-complete resume), which is exactly when skipping is
    right. For a resume-of-a-resume the replay spans every ancestor; absent
    ancestors all crashed for the same reason, so folding on a missing direct
    parent stays correct down the chain."""
    stats = [_scan_session(p) for p in logs]
    names = {p.name for p in logs}
    return [
        _scan_session(p, count_replayed=True)
        if s["resumed_from"] and s["resumed_from"] not in names else s
        for p, s in zip(logs, stats)
    ]


def _zoom_any(s: dict) -> bool:
    """Whether a session's stats dict has any small-model (zoom) usage."""
    return bool(
        s["zoom_input_tokens"] or s["zoom_output_tokens"]
        or s["zoom_cache_read_tokens"] or s["zoom_cache_write_tokens"]
        or s["zoom_turns"]
    )


def _cost(s: dict) -> float | None:
    """Estimated USD cost of one session's stats dict, or None when the model's
    pricing is unknown (reported as N/A, never a made-up number). A big-small
    session prices its two buckets at their own models' rates; either bucket
    unpriced makes the whole session N/A."""
    main = estimate_cost(
        s["model"], s["input_tokens"], s["output_tokens"],
        s["cache_read_tokens"], s["cache_write_tokens"],
    )
    if not _zoom_any(s):
        return main
    zoom = estimate_cost(
        s["small_model"] or "unknown",
        s["zoom_input_tokens"], s["zoom_output_tokens"],
        s["zoom_cache_read_tokens"], s["zoom_cache_write_tokens"],
    )
    if main is None or zoom is None:
        return None
    return main + zoom


def _is_project_lean(path: str) -> bool:
    """Whether a repo-relative path counts as the project's own Lean code:
    a .lean file that is not inside any .lake/ directory (where Lake checks out
    dependency packages -- their sources aren't the user's code)."""
    return (
        path.endswith(".lean")
        and not path.startswith(".lake/")
        and "/.lake/" not in path
    )


def _patch_lean_delta(patch: Path) -> tuple[int, int] | None:
    """(added, removed) .lean line counts of one format-patch, restricted to
    project Lean files (see _is_project_lean), or None if the patch is
    unreadable. Non-lean files -- notably the folded-in .gerbil/*.jsonl session
    log -- are skipped, as is the commit-message preamble (counting only starts
    at a `diff --git` header)."""
    try:
        lines = patch.read_text().splitlines()
    except OSError:
        return None
    added = removed = 0
    in_lean_file = False
    for line in lines:
        if line == "-- ":
            # format-patch's trailing signature delimiter; the version line after
            # it would otherwise be mistaken for diff content.
            break
        if line.startswith("diff --git "):
            # Take the post-image (b/) path; rsplit keeps paths containing
            # spaces intact (only a path containing " b/" itself would confuse it).
            in_lean_file = _is_project_lean(line.rsplit(" b/", 1)[-1])
        elif in_lean_file:
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
    return added, removed


def _committed_lean_delta(project_dir: Path, log_name: str) -> tuple[int, int] | None:
    """(added, removed) project .lean line counts of the commit that carries a
    session's folded-in log, or None when no commit contains it. A session log
    reaches the project's .gerbil/ only by `gerbil commit` (git am) applying the
    patch the log was folded into, so the commit that added the log IS the
    session's squashed commit -- and the .patch file itself is typically gone
    once applied, making history the authoritative record of the diff."""
    result = subprocess.run(
        ["git", "-C", str(project_dir), "log", "--diff-filter=A",
         "--format=%H", "--", f".gerbil/{log_name}"],
        capture_output=True,
        text=True,
    )
    commits = result.stdout.split()
    if result.returncode != 0 or not commits:
        return None
    # numstat: one "<added>\t<removed>\t<path>" per file, "-" counts for binary.
    show = subprocess.run(
        ["git", "-C", str(project_dir), "show", "--numstat", "--format=",
         commits[0]],
        capture_output=True,
        text=True,
    )
    if show.returncode != 0:
        return None
    added = removed = 0
    for line in show.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] == "-":
            continue
        if _is_project_lean(parts[2]):
            added += int(parts[0])
            removed += int(parts[1])
    return added, removed


def _lean_line_count(project_dir: Path, commit: str) -> int | None:
    """Total project .lean lines at `commit` in the host repo, or None when the
    commit can't be inspected here (not a git repo, or the base predates/never
    reached this clone). `git grep -c -e ''` counts every line of every matching
    blob straight from the object store -- no checkout needed. `:(top)` anchors
    the pathspecs at the repo root regardless of where the Lake project lives."""
    if not commit:
        return None
    result = subprocess.run(
        ["git", "-C", str(project_dir), "grep", "-c", "-e", "", commit, "--",
         ":(top)*.lean", ":(top,exclude).lake/**", ":(top,exclude)**/.lake/**"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        # One "<commit>:<path>:<count>" per file; count follows the last colon.
        return sum(
            int(line.rsplit(":", 1)[1])
            for line in result.stdout.splitlines()
            if line
        )
    if result.returncode == 1 and not result.stderr.strip():
        return 0  # valid commit, just no .lean files in it
    return None


def cmd_summarize(args) -> None:
    """Aggregate token usage, estimated cost, and tool-call stats across all
    .gerbil/*.jsonl session logs in the project, ending with a per-session
    breakdown: cost, .lean lines added/removed (from the commit each log was
    folded into, or its sibling .patch when not yet committed), and a running
    total of the project's lean code size."""
    project_dir = _resolve_at(args.at)
    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")

    out_dir = project_dir / ".gerbil"
    logs = sorted(out_dir.glob("*.jsonl"))
    if not logs:
        # The live session log always lands in ~/.gerbil/sessions/; it also
        # reaches the project's .gerbil/ (via the committed patch) unless the
        # run used --omit-session-log.
        archive = Path.home() / ".gerbil" / "sessions"
        msg = f"no session logs (*.jsonl) found in {out_dir}"
        if archive.is_dir() and any(archive.glob("*.jsonl")):
            msg += (
                f"\nnote: session logs are archived in {archive} -- runs made "
                "with --omit-session-log keep them only there."
            )
        sys.exit(msg)

    stats = _scan_sessions(logs)

    # Big-small sessions keep the small model's (zoom) usage in separate
    # buckets for pricing; the overall token totals combine both.
    total_in = sum(s["input_tokens"] + s["zoom_input_tokens"] for s in stats)
    total_out = sum(s["output_tokens"] + s["zoom_output_tokens"] for s in stats)
    total_thinking = sum(
        s["thinking_tokens"] + s["zoom_thinking_tokens"] for s in stats
    )
    total_cache_read = sum(
        s["cache_read_tokens"] + s["zoom_cache_read_tokens"] for s in stats
    )
    total_cache_write = sum(
        s["cache_write_tokens"] + s["zoom_cache_write_tokens"] for s in stats
    )
    total_turns = sum(s["turns"] + s["zoom_turns"] for s in stats)
    costs = [_cost(s) for s in stats]
    total_cost = sum(c for c in costs if c is not None)
    unpriced = sum(1 for c in costs if c is None)

    tools: collections.Counter = collections.Counter()
    for s in stats:
        tools.update(s["tool_calls"])

    status_counts = collections.Counter(s["status"] for s in stats)

    # Per-model rollup of tokens + cost. A big-small session contributes to two
    # entries: its outer bucket to the big model's, its zoom bucket to the
    # small model's -- each priced at that model's own rates.
    by_model: dict[str, dict] = {}

    def _fold_model(name, inp, out, thinking, cache_read, cache_write):
        m = by_model.setdefault(
            name,
            {"sessions": 0, "input": 0, "output": 0, "thinking": 0,
             "cached": 0, "cost": 0.0},
        )
        m["sessions"] += 1
        m["input"] += inp
        m["output"] += out
        m["thinking"] += thinking
        m["cached"] += cache_read + cache_write
        # None poisons the rollup: one unpriced session makes the model's cost N/A.
        c = estimate_cost(name, inp, out, cache_read, cache_write)
        if c is None:
            m["cost"] = None
        elif m["cost"] is not None:
            m["cost"] += c

    for s in stats:
        _fold_model(
            s["model"], s["input_tokens"], s["output_tokens"],
            s["thinking_tokens"], s["cache_read_tokens"], s["cache_write_tokens"],
        )
        if _zoom_any(s):
            _fold_model(
                s["small_model"] or "unknown",
                s["zoom_input_tokens"], s["zoom_output_tokens"],
                s["zoom_thinking_tokens"], s["zoom_cache_read_tokens"],
                s["zoom_cache_write_tokens"],
            )

    bold = lambda t: style(t, "bold")
    print(bold(f"gerbil summary -- {len(logs)} session(s) in {out_dir}"))
    print()

    print(bold("Tokens"))
    print(f"  input:    {total_in:>12,}")
    # Cache reads/writes are prompt tokens beyond `input` (which is only the
    # uncached remainder), billed at ~0.1x / ~1.25x the input rate.
    if total_cache_read or total_cache_write:
        print(f"  cache-rd: {total_cache_read:>12,}  {style('(0.1x input rate)', 'gray')}")
        print(f"  cache-wr: {total_cache_write:>12,}  {style('(1.25x input rate)', 'gray')}")
    print(f"  output:   {total_out:>12,}")
    # thinking is a subset of output (billed at the output rate), so list it as a
    # breakdown beneath output rather than adding it into the total.
    if total_thinking:
        print(f"  thinking: {total_thinking:>12,}  {style('(of output)', 'gray')}")
    total_all = total_in + total_cache_read + total_cache_write + total_out
    print(f"  total:    {total_all:>12,}")
    print()

    print(bold("Estimated cost"))
    if unpriced == len(stats):
        print("  N/A " + style("(unknown model pricing)", "gray"))
    elif unpriced:
        print(f"  ~${total_cost:,.4f} " + style(
            f"(excludes {unpriced} session(s) with unknown pricing)", "gray"))
    else:
        print(f"  ~${total_cost:,.4f}")
    print()

    print(bold("Sessions"))
    parts = []
    for label, color in (("completed", "green"), ("errored", "red"),
                         ("incomplete", "yellow"), ("unreadable", "gray")):
        n = status_counts.get(label, 0)
        if n:
            parts.append(f"{style(str(n), 'bold', color)} {label}")
    print("  " + ", ".join(parts) if parts else "  (none)")
    print(f"  turns: {total_turns:,}")
    print()

    print(bold("Tool calls") + f"  ({sum(tools.values()):,} total)")
    if tools:
        width = max(len(name) for name in tools)
        for name, n in tools.most_common():
            print(f"  {name:<{width}}  {n:>6,}")
    else:
        print("  (none)")
    print()

    print(bold("By model"))
    for model, m in sorted(by_model.items(), key=lambda kv: -(kv[1]["cost"] or 0.0)):
        if model.startswith("ollama:"):
            known = style(" (local)", "gray")
        elif model in MODEL_PRICING:
            known = ""
        elif (match := pricing_match(model)) is not None:
            # A gateway model (e.g. portkey:@provider/...) priced by the unique
            # MODEL_PRICING key embedded in its name.
            known = style(f" (priced as {match})", "gray")
        else:
            known = style(" (pricing unknown)", "gray")
        thinking = (
            style(f" ({m['thinking']:,} thinking)", "gray") if m["thinking"] else ""
        )
        cached = (
            style(f" ({m['cached']:,} cached)", "gray") if m["cached"] else ""
        )
        cost_str = "cost: N/A" if m["cost"] is None else f"~${m['cost']:,.4f}"
        print(
            f"  {style(model, 'cyan')}{known}: "
            f"{m['sessions']} session(s), "
            f"{m['input'] + m['cached'] + m['output']:,} tokens"
            f"{cached}{thinking}, {cost_str}"
        )
    print()

    # Per-session breakdown, in chronological order (the timestamped filenames
    # sort that way). Each session's code delta comes from the commit its log
    # was folded into (or a not-yet-committed sibling .patch); the running
    # total is anchored at the first session's base commit when that commit is
    # inspectable in the host repo, and is a relative net change otherwise
    # (e.g. logs copied in from another machine).
    print(bold("Per session") + "  " +
          style("(lean = .lean lines, excluding .lake packages)", "gray"))
    baseline = _lean_line_count(project_dir, stats[0]["base_commit"])
    running = baseline if baseline is not None else 0
    if baseline is not None:
        print("  " + style(
            f"lean baseline: {baseline:,} lines at "
            f"{stats[0]['base_commit'][:12]}", "gray"))
    else:
        print("  " + style(
            "base commit not inspectable here -- lean total shown as net "
            "change across sessions", "gray"))

    rows = []
    for path, s in zip(logs, stats):
        # Git history first: a log lands in .gerbil/ via the commit it was
        # folded into, so that commit holds the session's diff. A sibling
        # .patch (not yet committed here, e.g. a hand-copied log) covers the
        # rest.
        delta = _committed_lean_delta(project_dir, path.name)
        if delta is None:
            patch = path.with_suffix(".patch")
            delta = _patch_lean_delta(patch) if patch.is_file() else None
        if delta is None:
            change = "-"  # no commit or patch for this log (crashed / no changes)
        else:
            a, r = delta
            running += a - r
            change = f"+{a:,}/-{r:,}"
        c = _cost(s)
        # Combined across both models of a big-small session; the per-model
        # split lives in the "By model" rollup above.
        prompt_tokens = (s["input_tokens"] + s["cache_read_tokens"]
                         + s["cache_write_tokens"] + s["zoom_input_tokens"]
                         + s["zoom_cache_read_tokens"]
                         + s["zoom_cache_write_tokens"])
        rows.append((
            path.stem,
            s["status"],
            f"{s['turns'] + s['zoom_turns']:,}",
            f"{prompt_tokens:,}",
            f"{s['output_tokens'] + s['zoom_output_tokens']:,}",
            "N/A" if c is None else f"~${c:,.4f}",
            change,
            f"{running:,}" if baseline is not None else f"{running:+,}",
        ))

    headers = ("session", "status", "turns", "in", "out", "cost",
               "lean +/-", "lean total")
    widths = [max(len(h), *(len(row[i]) for row in rows))
              for i, h in enumerate(headers)]
    status_color = {"completed": "green", "errored": "red",
                    "incomplete": "yellow", "unreadable": "gray"}

    def render(cells, colorize=False):
        # Pad first, then color: ANSI codes would otherwise skew the alignment.
        out = []
        for i, cell in enumerate(cells):
            pad = f"{cell:<{widths[i]}}" if i < 2 else f"{cell:>{widths[i]}}"
            if colorize and i == 1:
                pad = style(pad, status_color.get(cells[1], "gray"))
            out.append(pad)
        return ("  " + "  ".join(out)).rstrip()

    print(style(render(headers), "gray"))
    for row in rows:
        print(render(row, colorize=True))


def _finalize_session(
    sandbox, base: str, result, *,
    session_path: Path, patch_path: Path, out_dir: Path, archive_dir: Path,
    include_session: bool, footer: str,
) -> str | None:
    """Squash the session's work -- the agent's intermediate commits plus its
    uncommitted changes -- into a single commit on top of base, then emit its
    format-patch (always exactly one patch). Shared by `gerbil run` and
    `--resume`. Returns the patch filename if anything was produced, else None."""
    if result.commit_message:
        message = result.commit_message + "\n\n" + footer + "\n"
    else:
        # The session stopped before generating a message (e.g. hit max turns);
        # still collapse its work into one commit, under a stub message.
        message = "gerbil session (incomplete; no commit message)\n\n" + footer + "\n"

    squashed = sandbox.squash_commit(base, message)

    print(f"{style('session:', 'bold')} {session_path}")
    if not squashed:
        print(f"{style('patch:', 'bold')}   (no changes; skipped)")
        return None

    # Optionally fold the session log into the (single) commit, and thus the patch.
    if include_session:
        sandbox.amend_with_file(
            f".gerbil/{session_path.name}", session_path.read_text()
        )
    patch_text = sandbox.format_patch(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch_text)
    shutil.copy2(patch_path, archive_dir / patch_path.name)
    print(f"{style('patch:', 'bold')}   {patch_path} (git am)")
    return patch_path.name
    return None


def _abort(exc: BaseException, session) -> None:
    """Handle a crash or Ctrl-C interruption uniformly, then exit the process.

    Records the failure as the in-flight session's terminal event (if one is
    open), prints a clear message, and -- since an open session is always
    resumable (its base commit and model are recorded at session_start, and the
    live .wip.patch snapshot holds the working tree) -- shows the command to
    continue it. A KeyboardInterrupt reads as a clean interruption rather than an
    error. Either way the .wip.patch is left in place: only a clean session
    finish removes it, so an interrupted run can always be picked back up.

    Shared by `gerbil run` and `--resume`."""
    interrupted = isinstance(exc, KeyboardInterrupt)
    if session is not None:
        session.record_error(exc)  # the log already lives in ~/.gerbil/

    if interrupted:
        print(
            "\n" + style("interrupted:", "bold", "yellow")
            + " session stopped (Ctrl-C)",
            file=sys.stderr,
        )
    else:
        print(
            f"\n{style('error:', 'bold', 'red')} "
            f"aborted by {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )

    if session is not None:
        print(
            f"{style('session:', 'bold')} {session.path} "
            "(details recorded inside)",
            file=sys.stderr,
        )
        print(
            f"{style('resume:', 'bold')}  gerbil resume {session.path}",
            file=sys.stderr,
        )
    # 130 is the conventional shell exit code for SIGINT; 1 for any other failure.
    sys.exit(130 if interrupted else 1)


def cmd_run(args) -> None:
    if not args.prompt:
        sys.exit("error: --prompt is required "
                 "(to continue a crashed session, use `gerbil resume SESSION_FILE`).")
    if args.ralph is not None and args.ralph < 1:
        sys.exit("error: --ralph N must be >= 1")
    if args.zoom_max_turns is not None and not args.small_model:
        sys.exit("error: --zoom-max-turns only applies with --small-model.")
    if args.zoom_max_turns is not None and args.zoom_max_turns < 1:
        sys.exit("error: --zoom-max-turns N must be >= 1")
    inner_max_turns = args.zoom_max_turns or DEFAULT_INNER_MAX_TURNS
    ralph_done_script = _load_ralph_done_script(
        args.ralph_done, have_ralph=args.ralph is not None
    )

    project_dir = _resolve_at(args.at)
    prompt_file = Path(args.prompt).resolve()

    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    repo_root = _require_git_repo(project_dir)
    _require_lake_project(project_dir)
    _require_clean_worktree(repo_root)
    _require_docker()
    if not prompt_file.is_file():
        sys.exit(f"error: {prompt_file} is not a file")

    prompt = prompt_file.read_text()
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    iterations = args.ralph if args.ralph else 1
    width = max(2, len(str(iterations)))

    # The session log lives in the user-level ~/.gerbil/sessions/ archive -- this
    # is the true, incrementally-written session file. Patches are written into
    # the project's .gerbil/ (for applying) and also copied to the archive.
    archive_dir = Path.home() / ".gerbil" / "sessions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    out_dir = project_dir / ".gerbil"  # created lazily, only when a patch is written

    # The running gerbil version (commit hash), supplied by the launcher.
    version = os.environ.get("GERBIL_VERSION", "unknown")

    def session_name(i: int) -> str:
        # In --ralph mode, number the per-session output files; otherwise a single set.
        return f"gerbil-{timestamp}-{i:0{width}d}" if args.ralph else f"gerbil-{timestamp}"

    # The launcher builds a version-matched image and passes its tag here; fall
    # back to the default for direct dev use (uv run python -m gerbil ...).
    image = os.environ.get("GERBIL_SANDBOX_IMAGE", "gerbil-lean-sandbox:latest")

    session = None  # the in-flight session, for the error handler below
    try:
        with LeanSandbox(
            project_dir=project_dir, repo_root=repo_root, image=image,
            fetch_cache=not args.skip_cache,
        ) as sandbox:
            # The MCP server runs inside the container, so it must be started
            # after the sandbox is ready and torn down before it (ExitStack
            # guarantees that ordering on every exit path). Started once and
            # reused across all --ralph sessions. If it fails, warn and continue.
            with contextlib.ExitStack() as stack:
                # For an ollama model, make sure a host-side server is up (and the
                # model is pulled) before any turn runs; torn down on exit if we
                # started it. In big-small mode the small model may be the
                # ollama one (OllamaServer reuses an already-running server).
                _start_ollama(args.model, stack)
                if args.small_model:
                    _start_ollama(args.small_model, stack)
                mcp, mcp_warning = (
                    _start_mcp(sandbox, stack) if args.mcp else (None, None)
                )
                toolset = Toolset(sandbox, mcp, ralph=bool(args.ralph))

                # The whole ralph chain layers on this host-reachable commit;
                # `ancestors` accumulates each completed session's patch file, in
                # order, so every session records exactly what rebuilds its base.
                chain_base = sandbox.head() if args.ralph else ""
                ancestors: list[str] = []

                for i in range(1, iterations + 1):
                    if args.ralph:
                        print(
                            "\n" + style(
                                f"===== ralph session {i}/{iterations} =====",
                                "bold", "magenta",
                            ),
                            flush=True,
                        )
                    name = session_name(i)
                    session_path = archive_dir / f"{name}.jsonl"  # live session log
                    patch_path = out_dir / f"{name}.patch"        # project-level patch
                    wip_path = archive_dir / f"{name}.wip.patch"  # live resume patch

                    # Baseline before the session: recorded in the log so --resume
                    # can recreate it, and used to format-patch the session's commits.
                    session_base = sandbox.head()
                    ralph_meta = {
                        "iteration": i, "total": iterations,
                        "chain_base": chain_base, "ancestors": list(ancestors),
                    } if args.ralph else None

                    session = Session(
                        path=session_path,
                        model=args.model,
                        project_dir=project_dir,
                        prompt_file=prompt_file,
                        version=version,
                        base_commit=session_base,
                        ralph=ralph_meta,
                        ralph_done_script=ralph_done_script,
                        include_session=not args.omit_session_log,
                        small_model=args.small_model,
                        inner_max_turns=(
                            inner_max_turns if args.small_model else None
                        ),
                    )
                    if mcp_warning:
                        session.record_warning(mcp_warning)

                    result = run_session(
                        sandbox, session, prompt, args.model, toolset,
                        max_turns=args.max_turns, wip_patch_path=wip_path,
                        small_model=args.small_model,
                        inner_max_turns=inner_max_turns,
                    )
                    session.close()
                    session = None
                    # The session finished cleanly; the live resume patch is only
                    # useful if it had crashed, so drop it. (On a crash, the except
                    # handler leaves it in place for `gerbil resume`.)
                    wip_path.unlink(missing_ok=True)

                    footer = _run_footer(args, i if args.ralph else None, iterations)
                    patch_name = _finalize_session(
                        sandbox, session_base, result,
                        session_path=session_path, patch_path=patch_path,
                        out_dir=out_dir, archive_dir=archive_dir,
                        include_session=not args.omit_session_log, footer=footer,
                    )
                    # Once a session produces a commit, later sessions in the chain
                    # build on it -- record its patch as an ancestor for resume.
                    if patch_name and args.ralph:
                        ancestors.append(patch_name)

                    # Optionally run the user's termination check on this session's
                    # committed tree. Skip it on the final iteration (nothing left
                    # to skip) -- the loop ends anyway.
                    if ralph_done_script and i < iterations:
                        print(flush=True)
                        if _ralph_done(sandbox, ralph_done_script):
                            break
    except (Exception, KeyboardInterrupt) as exc:
        # Catch both crashes and Ctrl-C (KeyboardInterrupt is a BaseException, not
        # an Exception, so it must be named explicitly). The sandbox/MCP context
        # managers have already torn down by the time we get here. _abort records
        # the failure, points the user at the session, and exits non-zero.
        _abort(exc, session)


def _resolve_patch(name: str, dirs: list[Path]) -> Path | None:
    """Find an ancestor patch file by name across candidate directories (the
    session archive, then the project's .gerbil/)."""
    for d in dirs:
        cand = d / name
        if cand.is_file():
            return cand
    return None


def _reconstruct_anchor(parsed, source_file: Path, repo_root: Path,
                        patch_dirs: list[Path]) -> tuple[str, list[str]]:
    """Resolve the host-reachable commit a session's base is rebuilt from, plus
    the ancestor patch texts (a ralph chain's prior sessions) to replay on top of
    it. A single session anchors directly on its base commit with no ancestors.
    Exits with a clear message if the anchor is missing from the repo or an
    ancestor patch file can't be found. Shared by --resume and reconstruct-patch."""
    ralph = parsed.ralph
    if ralph:
        anchor = ralph.get("chain_base") or parsed.base_commit
        ancestor_names = list(ralph.get("ancestors") or [])
    else:
        anchor = parsed.base_commit
        ancestor_names = []

    if subprocess.run(
        ["git", "-C", str(repo_root), "cat-file", "-e", f"{anchor}^{{commit}}"],
        capture_output=True,
    ).returncode != 0:
        sys.exit(
            f"error: base commit {anchor[:12]} (from {source_file.name}) is not in "
            f"{repo_root}.\nThis needs the repository that produced the session, "
            "with its history intact."
        )

    texts = []
    for nm in ancestor_names:
        p = _resolve_patch(nm, patch_dirs)
        if p is None:
            sys.exit(
                f"error: ancestor patch {nm} (needed to rebuild "
                f"{source_file.name}) was not found in {patch_dirs[0]} "
                f"or {patch_dirs[-1]}."
            )
        texts.append(p.read_text())
    return anchor, texts


def _rebuild_base(sandbox, anchor: str, ancestor_texts: list[str]) -> None:
    """Recreate a session's committed starting point inside the sandbox: hard
    reset to the chain anchor, then `git am` each ancestor patch in order."""
    sandbox.checkout_force(anchor)
    if ancestor_texts:
        print(style(
            f"rebuilding chain on {anchor[:12]}: "
            f"replaying {len(ancestor_texts)} prior session(s)", "gray",
        ))
        for text in ancestor_texts:
            sandbox.git_am(text)


def cmd_resume(args) -> None:
    """Continue a crashed/incomplete session from its .jsonl log.

    Recreate the world the session started in, then replay the logged
    conversation and hand control back to the agent loop:

      - single session: hard-reset the tree to the recorded base commit and
        reapply the live working-tree patch the crash left behind.
      - --ralph chain: hard-reset to the chain's host-reachable base, `git am`
        the recorded ancestor patches (the prior sessions) to rebuild the
        committed history, then reapply the crashed session's working-tree patch.
        The remaining iterations of the loop run afterward, as fresh sessions.

    The continuation is written as fresh, self-contained session logs/patches
    (the original logs are left untouched), each itself resumable.
    """
    from .resume import parse_session

    resume_file = Path(args.session_file).resolve()
    if not resume_file.is_file():
        sys.exit(f"error: {resume_file} is not a file")
    try:
        parsed = parse_session(resume_file)
    except Exception as exc:
        sys.exit(f"error: cannot parse session {resume_file.name}: {exc}")

    if not parsed.base_commit:
        sys.exit(
            f"error: {resume_file.name} has no recorded base commit -- it predates "
            "--resume support and cannot be resumed."
        )
    if not parsed.model:
        sys.exit(f"error: {resume_file.name} does not record a model; cannot resume.")

    # Operate on --at if given, else the project the session ran in.
    project_dir = _resolve_at(args.at) if args.at else Path(parsed.project_dir).resolve()
    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    repo_root = _require_git_repo(project_dir)
    _require_lake_project(project_dir)
    _require_clean_worktree(repo_root)
    _require_docker()

    archive_dir = Path.home() / ".gerbil" / "sessions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    out_dir = project_dir / ".gerbil"
    version = os.environ.get("GERBIL_VERSION", "unknown")
    image = os.environ.get("GERBIL_SANDBOX_IMAGE", "gerbil-lean-sandbox:latest")
    patch_dirs = [resume_file.parent, out_dir]  # where ancestor patches may live

    # Resolve the reconstruction plan. For a ralph session, the chain layers on a
    # host-reachable `chain_base` and `ancestors` rebuild this session's base; for
    # a single session, its base commit is the anchor and there are no ancestors.
    ralph = parsed.ralph
    # The termination check survives a resume: a command-line --ralph_done wins,
    # else fall back to the script recorded in the crashed session's log, so the
    # rebuilt chain keeps the same stop condition without re-supplying it.
    ralph_done_script = _load_ralph_done_script(
        args.ralph_done, have_ralph=bool(ralph)
    )
    if ralph_done_script is None and parsed.ralph_done_script is not None:
        ralph_done_script = parsed.ralph_done_script
        print(style(
            f"using --ralph_done check recorded in {resume_file.name}", "gray",
        ), flush=True)
    # The session-log setting survives a resume: inherit the choice recorded in
    # the log, and let `gerbil resume --omit-session-log` still force it off.
    include_session = parsed.include_session and not args.omit_session_log
    if not parsed.include_session and not args.omit_session_log:
        print(style(
            f"honoring --omit-session-log recorded in {resume_file.name}", "gray",
        ), flush=True)
    # Big-small mode survives a resume: the small model and per-zoom turn cap
    # come from the log (like the model and prompt); --zoom-max-turns overrides
    # just the cap.
    if args.zoom_max_turns is not None and not parsed.small_model:
        sys.exit(
            "error: --zoom-max-turns only applies to a big-small "
            f"(--small-model) session; {resume_file.name} records none."
        )
    if args.zoom_max_turns is not None and args.zoom_max_turns < 1:
        sys.exit("error: --zoom-max-turns N must be >= 1")
    inner_max_turns = (
        args.zoom_max_turns or parsed.inner_max_turns or DEFAULT_INNER_MAX_TURNS
    )
    anchor, ancestor_patches = _reconstruct_anchor(
        parsed, resume_file, repo_root, patch_dirs
    )
    # The ancestor patch *names* (this chain's prior sessions). _reconstruct_anchor
    # resolves their contents; the names come straight from the ralph metadata and
    # seed `running_ancestors` so the continuation chain stays resumable.
    ancestor_names = list(ralph.get("ancestors") or []) if ralph else []
    if ralph:
        start_iter = int(ralph.get("iteration", 1))
        total_iters = int(ralph.get("total", start_iter))
    else:
        start_iter = total_iters = 1

    # The live working-tree patch sits next to the session log (same stem).
    src_wip = resume_file.parent / f"{resume_file.stem}.wip.patch"
    working_patch = src_wip.read_text() if src_wip.is_file() else ""

    # Output naming. Single: "<stem>-resume-<ts>". Ralph: a fresh chain prefix
    # "<chain>-resume-<ts>" with each session numbered by its original iteration.
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    chain_stem = re.sub(r"-\d+$", "", resume_file.stem) if ralph else resume_file.stem
    iwidth = max(2, len(str(total_iters)))

    def out_name(i: int) -> str:
        if ralph:
            return f"{chain_stem}-resume-{timestamp}-{i:0{iwidth}d}"
        return f"{resume_file.stem}-resume-{timestamp}"

    if parsed.complete:
        print(style(
            f"note: {resume_file.name} looks already-complete (ends on the model); "
            "resuming will just (re)generate its commit message.", "yellow",
        ))

    session = None
    try:
        with LeanSandbox(
            project_dir=project_dir, repo_root=repo_root, image=image,
            fetch_cache=not args.skip_cache,
        ) as sandbox:
            with contextlib.ExitStack() as stack:
                # The resumed session's model may be an ollama one; ensure a
                # host-side server (and the model) is available before continuing.
                _start_ollama(parsed.model, stack)
                if parsed.small_model:
                    _start_ollama(parsed.small_model, stack)
                mcp, mcp_warning = (
                    _start_mcp(sandbox, stack) if args.mcp else (None, None)
                )
                toolset = Toolset(sandbox, mcp, ralph=bool(ralph))

                # Rebuild the committed history this session started from.
                _rebuild_base(sandbox, anchor, ancestor_patches)

                # `ancestors` for the continuation: the original prior patches,
                # then each session we (re)produce here, so the new logs stay
                # resumable across the resume boundary.
                running_ancestors = list(ancestor_names)

                for i in range(start_iter, total_iters + 1):
                    if ralph:
                        print(
                            "\n" + style(
                                f"===== ralph session {i}/{total_iters} "
                                "(resumed) =====", "bold", "magenta",
                            ),
                            flush=True,
                        )
                    name = out_name(i)
                    session_path = archive_dir / f"{name}.jsonl"
                    patch_path = out_dir / f"{name}.patch"
                    wip_path = archive_dir / f"{name}.wip.patch"

                    iter_base = sandbox.head()
                    seeded = i == start_iter  # only the crashed session is replayed

                    # Lay the crashed session's uncommitted edits back on top.
                    if seeded and working_patch.strip():
                        try:
                            sandbox.apply_diff(working_patch)
                        except Exception as exc:
                            print(
                                f"{style('warning:', 'bold', 'yellow')} could not "
                                f"apply the saved working-tree patch ({exc}); "
                                "continuing from the base commit only.",
                                file=sys.stderr,
                            )

                    ralph_meta = {
                        "iteration": i, "total": total_iters,
                        "chain_base": anchor, "ancestors": list(running_ancestors),
                    } if ralph else None
                    session = Session(
                        path=session_path,
                        model=parsed.model,
                        project_dir=project_dir,
                        prompt_file=Path(parsed.prompt_file),
                        version=version,
                        base_commit=iter_base,
                        resumed_from=resume_file.name,
                        ralph=ralph_meta,
                        ralph_done_script=ralph_done_script,
                        include_session=include_session,
                        small_model=parsed.small_model,
                        inner_max_turns=(
                            inner_max_turns if parsed.small_model else None
                        ),
                    )
                    if mcp_warning:
                        session.record_warning(mcp_warning)

                    if seeded:
                        # Carry the pre-crash history forward so the new log is
                        # complete (and itself resumable), then continue it.
                        for ev in parsed.events:
                            session.record_replayed(ev)
                        print(style(
                            f"resuming {resume_file.name}: base {iter_base[:12]}, "
                            f"{len(parsed.events)} events replayed, "
                            f"model {parsed.model}", "gray",
                        ))
                        seed_messages = parsed.messages
                    else:
                        seed_messages = None

                    result = run_session(
                        sandbox, session, parsed.prompt, parsed.model, toolset,
                        max_turns=args.max_turns, messages=seed_messages,
                        wip_patch_path=wip_path,
                        small_model=parsed.small_model,
                        inner_max_turns=inner_max_turns,
                        # Only the crashed (seeded) session can be mid-zoom;
                        # later ralph iterations start fresh.
                        pending_zoom=parsed.pending_zoom if seeded else None,
                    )
                    session.close()
                    session = None
                    wip_path.unlink(missing_ok=True)

                    footer = (
                        "authored by Gerbil:\n"
                        f"--model {parsed.model}\n"
                        f"resume {resume_file.name}"
                    )
                    if parsed.small_model:
                        footer += f"\n--small-model {parsed.small_model}"
                    if ralph:
                        footer += f"\n--ralph (session {i}/{total_iters})"
                    patch_name = _finalize_session(
                        sandbox, iter_base, result,
                        session_path=session_path, patch_path=patch_path,
                        out_dir=out_dir, archive_dir=archive_dir,
                        include_session=include_session, footer=footer,
                    )
                    if patch_name and ralph:
                        running_ancestors.append(patch_name)

                    # Same termination check as a fresh run; skip on the last iter.
                    if ralph_done_script and i < total_iters:
                        print(flush=True)
                        if _ralph_done(sandbox, ralph_done_script):
                            break
    except (Exception, KeyboardInterrupt) as exc:
        # A resumed run is itself resumable: _abort points at this continuation's
        # own log (and Ctrl-C is caught the same as a crash -- see cmd_run).
        _abort(exc, session)


# The tool calls that mutate sandbox state and so must be replayed to reproduce a
# session's working tree. read_file and the lean_* MCP tools are read-only (and
# the search tools are rate-limited), so they are skipped. Filtering is by name
# only, so a big-small session's zoom-tagged (inner) bash/write/edit calls are
# intentionally replayed too -- the sub-sessions mutate the same working tree --
# while zoom_in/zoom_out themselves are naturally skipped.
_REPLAY_TOOLS = {"bash", "write_file", "edit_file"}


def _confirm(question: str) -> bool:
    """Ask a yes/no question on the terminal. Returns False on EOF or when there
    is no TTY, so a missing answer never counts as 'yes'."""
    if not sys.stdin.isatty():
        return False
    try:
        return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:
        return False


def cmd_reconstruct_patch(args) -> None:
    """Rebuild a session's .patch by *replaying its tool calls* in a fresh sandbox.

    Unlike --resume (which restores the working tree from the live .wip.patch
    snapshot), this re-executes every state-mutating tool call the session logged
    -- bash, write_file, edit_file -- on top of the session's reconstructed base,
    then commits the result and emits the patch. No LLM is involved. Useful when
    the original patch is missing or was corrupted (e.g. the agent mangled git).

    Caveat: replay is only as deterministic as the commands themselves -- bash
    that depends on time, network, or randomness may not reproduce exactly.
    """
    from .resume import parse_session

    session_file = Path(args.session_file).resolve()
    if not session_file.is_file():
        sys.exit(f"error: {session_file} is not a file")
    try:
        parsed = parse_session(session_file)
    except Exception as exc:
        sys.exit(f"error: cannot parse session {session_file.name}: {exc}")
    if not parsed.base_commit:
        sys.exit(
            f"error: {session_file.name} has no recorded base commit -- it predates "
            "base-commit tracking and cannot be reconstructed."
        )

    project_dir = _resolve_at(args.at) if args.at else Path(parsed.project_dir).resolve()
    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    repo_root = _require_git_repo(project_dir)
    _require_lake_project(project_dir)
    _require_clean_worktree(repo_root)
    _require_docker()

    archive_dir = Path.home() / ".gerbil" / "sessions"
    archive_dir.mkdir(parents=True, exist_ok=True)
    out_dir = project_dir / ".gerbil"
    image = os.environ.get("GERBIL_SANDBOX_IMAGE", "gerbil-lean-sandbox:latest")
    patch_dirs = [session_file.parent, out_dir]

    anchor, ancestor_patches = _reconstruct_anchor(
        parsed, session_file, repo_root, patch_dirs
    )

    # The mutating tool calls to replay, in order.
    calls = [
        (e["name"], e.get("args") or {})
        for e in parsed.events
        if e.get("event") == "tool_call" and e.get("name") in _REPLAY_TOOLS
    ]
    if not calls:
        sys.exit(f"error: {session_file.name} logged no replayable tool calls.")

    patch_path = out_dir / f"{session_file.stem}.patch"

    # Confirm before clobbering an existing patch -- and do it now, before the
    # (slow) sandbox boot + replay, not after.
    if patch_path.is_file() and patch_path.stat().st_size > 0 and not args.force:
        if not sys.stdin.isatty():
            sys.exit(
                f"error: {patch_path} already exists. Re-run with --force to "
                "overwrite (no terminal available to confirm)."
            )
        if not _confirm(f"{patch_path} already exists. Overwrite it?"):
            sys.exit("aborted: existing patch left in place.")

    try:
        with LeanSandbox(
            project_dir=project_dir, repo_root=repo_root, image=image,
            fetch_cache=not args.skip_cache,
        ) as sandbox:
            # Built-in tools only; replay never needs the (read-only) MCP tools.
            toolset = Toolset(sandbox, mcp=None, ralph=False)

            _rebuild_base(sandbox, anchor, ancestor_patches)
            base = sandbox.head()

            print(style(
                f"replaying {len(calls)} tool call(s) from {session_file.name} "
                f"on {base[:12]}...", "gray",
            ))
            errors = 0
            for n, (name, call_args) in enumerate(calls, 1):
                print(
                    f"  {style('->', 'cyan')} "
                    f"{style(name, 'bold', 'cyan')}({_short_args(call_args)})",
                    flush=True,
                )
                result = toolset.dispatch(name, call_args)
                if result.is_error:
                    errors += 1
                    preview = result.content[:160].replace("\n", " ")
                    print(f"     {style('<- [error]', 'yellow')} {preview}", flush=True)
            if errors:
                print(style(
                    f"note: {errors} of {len(calls)} replayed tool call(s) reported "
                    "an error (this can be normal -- the agent saw these too).",
                    "yellow",
                ))

            # Squash whatever the replay produced (the agent's replayed commits
            # plus leftover changes) into a single commit, under the session's own
            # message if it generated one, then emit the patch.
            msg = parsed.commit_message or f"Reconstructed patch for {session_file.name}"
            footer = f"authored by Gerbil:\n--reconstruct-patch {session_file.name}"
            if not sandbox.squash_commit(base, msg + "\n\n" + footer + "\n"):
                sys.exit(
                    "error: replay produced no changes -- nothing to reconstruct. "
                    "The session may have made no committable edits, or its bash "
                    "commands did not reproduce outside their original run."
                )
            patch_text = sandbox.format_patch(base)

            existed = patch_path.is_file() and patch_path.stat().st_size > 0
            out_dir.mkdir(parents=True, exist_ok=True)
            patch_path.write_text(patch_text)
            shutil.copy2(patch_path, archive_dir / patch_path.name)
            note = " (overwrote existing)" if existed else ""
            print(f"{style('patch:', 'bold')}   {patch_path} (git am){note}")
    except SystemExit:
        raise
    except KeyboardInterrupt:
        # reconstruct-patch is not a model session, so there is nothing to resume;
        # just stop cleanly instead of dumping a traceback.
        print(
            "\n" + style("interrupted:", "bold", "yellow")
            + " reconstruct-patch stopped (Ctrl-C)",
            file=sys.stderr,
        )
        sys.exit(130)
    except Exception as exc:
        print(
            f"\n{style('error:', 'bold', 'red')} "
            f"reconstruct-patch failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _short_args(args: dict) -> str:
    """A compact one-line rendering of tool args for the replay log."""
    s = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) <= 100 else s[:100] + "..."


def _start_mcp(sandbox, stack):
    """Start the lean-lsp MCP client, registering it for teardown on the stack.

    Started once and reused across all --ralph sessions. Returns
    (McpClient | None, warning | None): on failure, the client is None and a
    warning string is returned (printed here, and recorded to each session) so
    the run continues with just the built-in tools.
    """
    try:
        from .mcp_client import McpClient

        mcp = stack.enter_context(McpClient(sandbox, project_path=sandbox.project_path))
        banner = f"[mcp: {len(mcp.list_tools())} lean tools available"
        if mcp.disabled_tools:
            # Surface that the network tools were withheld, so it is clear the
            # sandbox stays hermetic (and why those tools aren't offered).
            banner += f"; {len(mcp.disabled_tools)} network tools disabled"
        banner += "]"
        print(style(banner, "gray"), flush=True)
        return mcp, None
    except Exception as exc:
        warning = f"lean-lsp MCP unavailable: {type(exc).__name__}: {exc}"
        print(
            f"{style('warning:', 'bold', 'yellow')} {warning}; "
            "continuing with built-in tools only.",
            file=sys.stderr,
        )
        return None, warning


def _start_ollama(model, stack):
    """Ensure a host-side ollama server is reachable for an `ollama:<NAME>` model,
    registering it for teardown on the stack (only a server we start is stopped).

    No-op for any non-ollama model. Verifies the requested model is pulled locally
    before the session begins, failing fast with a `ollama pull` hint otherwise.
    Reused across all --ralph sessions, like the MCP client."""
    from .ollama import OllamaServer, ensure_model_available, is_ollama_model

    if not is_ollama_model(model):
        return
    stack.enter_context(OllamaServer())
    ensure_model_available(model)
    print(style(f"[ollama: serving {model}]", "gray"), flush=True)


def _load_ralph_done_script(path_str: str | None, *, have_ralph: bool) -> str | None:
    """Read the --ralph_done check script, or return None if not requested. Exits
    with a clear message if it's given without --ralph or the file is missing."""
    if not path_str:
        return None
    if not have_ralph:
        sys.exit("error: --ralph_done only applies to a --ralph session.")
    p = Path(path_str)
    if not p.is_file():
        sys.exit(f"error: --ralph_done script {p} is not a file")
    return p.read_text()


def _ralph_done(sandbox, script: str) -> bool:
    """Run the --ralph_done check inside the container on the session's committed
    tree. Exit code 0 means the loop is finished (return True to stop); any other
    code means keep going. The script's output is shown either way."""
    print(style("running --ralph_done check...", "bold", "magenta"), flush=True)
    result = sandbox.run_script(script)
    out = (result.stdout + result.stderr).strip()
    if out:
        print(style(out.replace("\n", "\n  "), "gray"), flush=True)
    if result.exit_code == 0:
        print(style("[ralph_done: check passed (exit 0) -- stopping loop]",
                    "bold", "magenta"), flush=True)
        return True
    print(style(
        f"[ralph_done: check did not pass (exit {result.exit_code}) -- continuing]",
        "gray",
    ), flush=True)
    return False


def _run_footer(args, iteration=None, total=None) -> str:
    """A trailer describing this Gerbil run, appended to the commit message."""
    max_turns = args.max_turns if args.max_turns is not None else "unlimited"
    lines = [
        "authored by Gerbil:",
        f"--model {args.model}",
        f"--max-turns {max_turns}",
    ]
    if getattr(args, "small_model", None):
        lines.append(f"--small-model {args.small_model}")
    if iteration is not None:
        lines.append(f"--ralph (session {iteration}/{total})")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
