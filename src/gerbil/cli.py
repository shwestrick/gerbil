#!/usr/bin/env python3
"""gerbil -- a sandboxed Lean theorem-proving agent.

Usage:
    gerbil --at DIRECTORY --prompt FILE [--ralph N]

The agent works on the real repository (full history) inside the container.
Each session's changes are committed there, and the result is emitted as a
git format-patch -- a single .patch file holding the commit title, message, and
diff, applied on the host with `git am`.

Outputs (written into a .gerbil/ directory inside the --at project):
    gerbil-TIMESTAMP.jsonl    session log (model, turns, token counts, tool calls)
    gerbil-TIMESTAMP.patch    git format-patch of the session's commit(s)

With --ralph N, N sessions run back-to-back on the same prompt (reusing the
sandbox); each set of outputs is numbered gerbil-TIMESTAMP-NN.{jsonl,patch} and
each session builds on the previous one's commit.

Every session log and patch is also archived to ~/.gerbil/ (same filenames),
regardless of what happens to the project-level copies.
"""

import argparse
import contextlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

from .agent import MODEL_PRICING, run_session
from .sandbox import LeanSandbox
from .session import Session
from .term import style
from .tools import Toolset


DEFAULT_MODEL = "gemini-3.1-pro-preview"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gerbil",
        description="Sandboxed Lean theorem-proving agent.",
    )
    parser.add_argument(
        "--at",
        required=True,
        metavar="DIRECTORY",
        help="Path to an existing Lean/Lake project (must be a git repo).",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        metavar="FILE",
        help="Path to a file containing the task description.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"LLM to use (default: {DEFAULT_MODEL}). Provider is auto-detected. "
            f"Known models: {', '.join(MODEL_PRICING)}."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Safety cap on agent turns (default: unlimited, runs until done).",
    )
    parser.add_argument(
        "--skip-cache",
        action="store_true",
        help="Skip 'lake exe cache get' at startup (faster, but mathlib will "
        "rebuild from source on first use).",
    )
    parser.add_argument(
        "--no-mcp",
        dest="mcp",
        action="store_false",
        help="Disable the lean-lsp MCP tools; use only the built-in tools.",
    )
    parser.add_argument(
        "--ralph",
        type=int,
        metavar="N",
        default=None,
        help="Run N sessions back-to-back on the same prompt, reusing the "
        "sandbox. Each session is committed inside the container so the next "
        "builds on it; outputs are numbered gerbil-<ts>-NN.{jsonl,patch}.",
    )
    parser.add_argument(
        "--include-session",
        action="store_true",
        help="Include the session .jsonl log in the commit (folded in via "
        "commit --amend before the patch is produced).",
    )
    args = parser.parse_args()

    if args.ralph is not None and args.ralph < 1:
        sys.exit("error: --ralph N must be >= 1")

    project_dir = Path(args.at).resolve()
    prompt_file = Path(args.prompt).resolve()

    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    if not prompt_file.is_file():
        sys.exit(f"error: {prompt_file} is not a file")

    prompt = prompt_file.read_text()
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    iterations = args.ralph if args.ralph else 1
    width = max(2, len(str(iterations)))

    # Outputs go in a .gerbil/ directory inside the project, to keep its root clean.
    out_dir = project_dir / ".gerbil"
    out_dir.mkdir(parents=True, exist_ok=True)

    # A user-level archive of ALL gerbil data (every session log and patch, same
    # filenames), kept regardless of what happens to the project-level copies.
    archive_dir = Path.home() / ".gerbil"
    archive_dir.mkdir(parents=True, exist_ok=True)

    def archive(path: Path) -> None:
        if path.exists():
            shutil.copy2(path, archive_dir / path.name)

    def stem(i: int) -> Path:
        # In --ralph mode, number the per-session output files; otherwise a
        # single unnumbered set.
        name = f"gerbil-{timestamp}-{i:0{width}d}" if args.ralph else f"gerbil-{timestamp}"
        return out_dir / name

    session = None  # the in-flight session, for the error handler below
    try:
        with LeanSandbox(
            project_dir=project_dir, fetch_cache=not args.skip_cache
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
                    s = stem(i)
                    session_path = s.with_suffix(".jsonl")
                    patch_path = s.with_suffix(".patch")

                    session = Session(
                        path=session_path,
                        model=args.model,
                        project_dir=project_dir,
                        prompt_file=prompt_file,
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
                    archive(session_path)  # archive the log before any deletion

                    # Commit the agent's uncommitted changes on real HEAD, with the
                    # generated message + a footer recording how this run was invoked.
                    if result.commit_message:
                        footer = _run_footer(args, i if args.ralph else None, iterations)
                        full = result.commit_message + "\n\n" + footer + "\n"
                        sandbox.commit(full)

                    # The session "produced" something iff HEAD advanced (whether by
                    # gerbil's commit above or commits the agent made itself).
                    if sandbox.head() != session_base:
                        # Optionally fold the session log into the commit, so the
                        # format-patch carries it too.
                        if args.include_session:
                            sandbox.amend_with_file(
                                f".gerbil/{session_path.name}", session_path.read_text()
                            )
                        patch_path.write_text(sandbox.format_patch(session_base))
                        archive(patch_path)
                        # The session log is now embedded in the patch; drop the
                        # loose host copy so `git am` doesn't collide with it.
                        if args.include_session:
                            session_path.unlink()
                            print(f"{style('session:', 'bold')} (embedded in patch)")
                        else:
                            print(f"{style('session:', 'bold')} {session_path}")
                        print(f"{style('patch:', 'bold')}   {patch_path} (git am)")
                    else:
                        print(f"{style('session:', 'bold')} {session_path}")
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
            session.record_error(exc)
            archive(session.path)  # preserve the errored session log too
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
