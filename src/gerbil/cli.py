#!/usr/bin/env python3
"""gerbil -- a sandboxed Lean theorem-proving agent.

Usage:
    gerbil run [--prompt FILE] [--at DIRECTORY] [--ralph N] ...
    gerbil apply [--at DIRECTORY]

(Normally invoked through the `gerbil` launcher, which also provides `update`
and `--version`. --at defaults to the current directory.)

The agent works on the real repository (full history) inside the container.
Each session's changes are committed there, and the result is emitted as a
git format-patch -- a single .patch file holding the commit title, message, and
diff, applied on the host with `gerbil apply` (git am).

Outputs:
    ~/.gerbil/sessions/gerbil-TIMESTAMP.jsonl  the live session log (the true
                                         session file, written as the run proceeds)
    <project>/.gerbil/gerbil-TIMESTAMP.patch   git format-patch of the session's
                                         commit(s); apply with `git am`
The patch is also copied to ~/.gerbil/sessions/, so the archive holds all gerbil
data. The session log only reaches the --at project if --include-session is
passed (it is folded into the commit, and thus the patch).

With --ralph N, N sessions run back-to-back on the same prompt (reusing the
sandbox); outputs are numbered gerbil-TIMESTAMP-NN and each session builds on
the previous one's commit.
"""

import argparse
import contextlib
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .agent import MODEL_PRICING, run_session
from .sandbox import LeanSandbox
from .session import Session
from .term import style
from .tools import Toolset


DEFAULT_MODEL = "gemini-3.1-pro-preview"


def _resolve_at(at: str | None) -> Path:
    """The Lake project to operate on. Defaults to the directory the launcher was
    invoked from (GERBIL_CWD), falling back to the current working directory."""
    base = at or os.environ.get("GERBIL_CWD") or os.getcwd()
    return Path(base).resolve()


def _require_git_repo(project_dir: Path) -> None:
    """Exit with a clear message unless project_dir is the root of a git repo.
    gerbil works on the real repository (uploads .git, commits, format-patch),
    so the Lake project must be a git repo, and we operate from its root."""
    result = subprocess.run(
        ["git", "-C", str(project_dir), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(
            f"error: {project_dir} is not a git repository.\n"
            "gerbil needs the Lake project to be a git repo -- run `git init` "
            "(and make a commit) there first."
        )
    toplevel = Path(result.stdout.strip()).resolve()
    if toplevel != project_dir:
        sys.exit(
            f"error: {project_dir} is inside a git repo but not its root.\n"
            f"Run gerbil from the project root instead: {toplevel}"
        )
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
            f"error: the git repo at {project_dir} has no commits yet.\n"
            "gerbil works against a base commit -- make an initial commit first:\n"
            "  git add -A && git commit -m 'initial commit'"
        )


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


def _require_clean_worktree(project_dir: Path) -> None:
    """Exit unless the working tree is clean. gerbil runs on top of a clean
    commit -- it uploads only tracked files (== HEAD) and commits the agent's
    changes on top. Uncommitted changes to tracked files would otherwise be
    swept into the agent's commit. Untracked files are ignored (not uploaded)."""
    dirty = subprocess.run(
        ["git", "-C", str(project_dir), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        sys.exit(
            f"error: {project_dir} has uncommitted changes.\n"
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
        required=True,
        metavar="FILE",
        help="Path to a file containing the task description.",
    )
    run_p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"LLM to use (default: {DEFAULT_MODEL}). Provider is auto-detected. "
            f"Known models: {', '.join(MODEL_PRICING)}."
        ),
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
        "--include-session",
        action="store_true",
        help="Include the session .jsonl log in the commit (folded in via "
        "commit --amend before the patch is produced).",
    )
    run_p.set_defaults(func=cmd_run)

    apply_p = sub.add_parser(
        "apply", help="git am the patches gerbil produced, in order"
    )
    apply_p.add_argument(
        "--at",
        metavar="DIRECTORY",
        help="Path to the Lean/Lake project (a git repo). Default: current dir.",
    )
    apply_p.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


def cmd_apply(args) -> None:
    """Apply each .gerbil/gerbil-*.patch (a git format-patch) in order via git am."""
    project_dir = _resolve_at(args.at)
    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    _require_git_repo(project_dir)
    _require_lake_project(project_dir)

    out_dir = project_dir / ".gerbil"
    patches = sorted(out_dir.glob("gerbil-*.patch"))
    if not patches:
        sys.exit(f"no patches found in {out_dir}")

    for patch in patches:
        print(f"{style('applying:', 'bold')} {patch.name}")
        result = subprocess.run(["git", "am", str(patch)], cwd=project_dir)
        if result.returncode != 0:
            sys.exit(
                f"git am failed on {patch.name}. Resolve and `git am --continue`, "
                "or `git am --abort` to back out."
            )
    print(style("done", "bold"))


def cmd_run(args) -> None:
    if args.ralph is not None and args.ralph < 1:
        sys.exit("error: --ralph N must be >= 1")

    project_dir = _resolve_at(args.at)
    prompt_file = Path(args.prompt).resolve()

    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    _require_git_repo(project_dir)
    _require_lake_project(project_dir)
    _require_clean_worktree(project_dir)
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

    def archive(path: Path) -> None:
        if path.exists():
            shutil.copy2(path, archive_dir / path.name)

    def session_name(i: int) -> str:
        # In --ralph mode, number the per-session output files; otherwise a single set.
        return f"gerbil-{timestamp}-{i:0{width}d}" if args.ralph else f"gerbil-{timestamp}"

    # The launcher builds a version-matched image and passes its tag here; fall
    # back to the default for direct dev use (uv run python -m gerbil ...).
    image = os.environ.get("GERBIL_SANDBOX_IMAGE", "lean-sandbox:latest")

    session = None  # the in-flight session, for the error handler below
    try:
        with LeanSandbox(
            project_dir=project_dir, image=image, fetch_cache=not args.skip_cache
        ) as sandbox:
            # The MCP server runs inside the container, so it must be started
            # after the sandbox is ready and torn down before it (ExitStack
            # guarantees that ordering on every exit path). Started once and
            # reused across all --ralph sessions. If it fails, warn and continue.
            with contextlib.ExitStack() as stack:
                mcp, mcp_warning = (
                    _start_mcp(sandbox, stack) if args.mcp else (None, None)
                )
                toolset = Toolset(sandbox, mcp, ralph=bool(args.ralph))

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

                    session = Session(
                        path=session_path,
                        model=args.model,
                        project_dir=project_dir,
                        prompt_file=prompt_file,
                        version=version,
                    )
                    if mcp_warning:
                        session.record_warning(mcp_warning)

                    # Baseline before the session, so we can format-patch its commits.
                    session_base = sandbox.head()

                    result = run_session(
                        sandbox, session, prompt, args.model, toolset,
                        max_turns=args.max_turns,
                    )
                    session.close()
                    session = None

                    # Commit the agent's uncommitted changes on real HEAD, with the
                    # generated message + a footer recording how this run was invoked.
                    if result.commit_message:
                        footer = _run_footer(args, i if args.ralph else None, iterations)
                        full = result.commit_message + "\n\n" + footer + "\n"
                        sandbox.commit(full)

                    print(f"{style('session:', 'bold')} {session_path}")

                    # The session "produced" something iff HEAD advanced (whether by
                    # gerbil's commit above or commits the agent made itself).
                    if sandbox.head() != session_base:
                        # Optionally fold the session log into the commit (and thus
                        # the patch); otherwise it never reaches the --at project.
                        if args.include_session:
                            sandbox.amend_with_file(
                                f".gerbil/{session_path.name}", session_path.read_text()
                            )
                        out_dir.mkdir(parents=True, exist_ok=True)
                        patch_path.write_text(sandbox.format_patch(session_base))
                        archive(patch_path)
                        print(f"{style('patch:', 'bold')}   {patch_path} (git am)")
                    else:
                        print(f"{style('patch:', 'bold')}   (no changes; skipped)")

                    # The model can call ralph_done to end the loop early.
                    if toolset.ralph_done:
                        reason = toolset.ralph_done_reason or "task complete"
                        print(
                            "\n" + style(f"[ralph_done: {reason}]", "bold", "magenta"),
                            flush=True,
                        )
                        break
    except Exception as exc:
        # Catch-all: record the failure as the in-flight session's terminal event
        # (if one is open), point the user at it, and exit non-zero.
        if session is not None:
            session.record_error(exc)  # the log already lives in ~/.gerbil/
        print(
            f"\n{style('error:', 'bold', 'red')} "
            f"aborted by {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if session is not None:
            print(
                f"{style('session:', 'bold')} {session.path} "
                "(error details recorded inside)",
                file=sys.stderr,
            )
        sys.exit(1)


def _start_mcp(sandbox, stack):
    """Start the lean-lsp MCP client, registering it for teardown on the stack.

    Started once and reused across all --ralph sessions. Returns
    (McpClient | None, warning | None): on failure, the client is None and a
    warning string is returned (printed here, and recorded to each session) so
    the run continues with just the built-in tools.
    """
    try:
        from .mcp_client import McpClient

        mcp = stack.enter_context(McpClient(sandbox))
        print(
            style(f"[mcp: {len(mcp.list_tools())} lean tools available]", "gray"),
            flush=True,
        )
        return mcp, None
    except Exception as exc:
        warning = f"lean-lsp MCP unavailable: {type(exc).__name__}: {exc}"
        print(
            f"{style('warning:', 'bold', 'yellow')} {warning}; "
            "continuing with built-in tools only.",
            file=sys.stderr,
        )
        return None, warning


def _run_footer(args, iteration=None, total=None) -> str:
    """A trailer describing this Gerbil run, appended to the commit message."""
    max_turns = args.max_turns if args.max_turns is not None else "unlimited"
    lines = [
        "authored by Gerbil:",
        f"--model {args.model}",
        f"--max-turns {max_turns}",
    ]
    if iteration is not None:
        lines.append(f"--ralph (session {iteration}/{total})")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
