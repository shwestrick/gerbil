#!/usr/bin/env python3
"""gerbil -- a sandboxed Lean theorem-proving agent.

Usage:
    gerbil --at DIRECTORY --prompt FILE

Outputs (in the current working directory):
    gerbil-TIMESTAMP.jsonl   session log (model, turns, token counts, tool calls)
    gerbil-TIMESTAMP.patch   git diff of changes gerbil made to the Lean project
"""

import argparse
import contextlib
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
    args = parser.parse_args()

    project_dir = Path(args.at).resolve()
    prompt_file = Path(args.prompt).resolve()

    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    if not prompt_file.is_file():
        sys.exit(f"error: {prompt_file} is not a file")

    prompt = prompt_file.read_text()
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    session_path = project_dir / f"gerbil-{timestamp}.jsonl"
    diff_path = project_dir / f"gerbil-{timestamp}.patch"
    commit_path = project_dir / f"gerbil-{timestamp}.commit"

    session = Session(
        path=session_path,
        model=args.model,
        project_dir=project_dir,
        prompt_file=prompt_file,
    )

    try:
        with LeanSandbox(
            project_dir=project_dir, fetch_cache=not args.skip_cache
        ) as sandbox:
            # The MCP server runs inside the container, so it must be started
            # after the sandbox is ready and torn down before it (ExitStack
            # guarantees that ordering on every exit path). If it fails to start,
            # warn and continue with just the built-in tools.
            with contextlib.ExitStack() as stack:
                mcp = None
                if args.mcp:
                    mcp = _start_mcp(sandbox, session, stack)
                toolset = Toolset(sandbox, mcp)
                result = run_session(
                    sandbox, session, prompt, args.model, toolset,
                    max_turns=args.max_turns,
                )
    except Exception as exc:
        # Catch-all: record the failure as the session's terminal event, point
        # the user at the session file, and exit non-zero.
        session.record_error(exc)
        print(
            f"\n{style('error:', 'bold', 'red')} "
            f"session aborted by {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print(
            f"{style('session:', 'bold')} {session_path} "
            "(error details recorded inside)",
            file=sys.stderr,
        )
        sys.exit(1)

    session.close()

    diff = result.diff if result else ""
    diff_path.write_text(diff)

    print(f"{style('session:', 'bold')} {session_path}")
    print(f"{style('diff:', 'bold')}    {diff_path}")

    # The commit message was produced as the session's final turn; append a
    # footer recording how this run of Gerbil was invoked.
    if result and result.commit_message:
        footer = _run_footer(args)
        commit_path.write_text(result.commit_message + "\n\n" + footer + "\n")
        print(f"{style('commit:', 'bold')}  {commit_path}")
    else:
        print(f"{style('commit:', 'bold')}  (no changes; skipped)")


def _start_mcp(sandbox, session, stack):
    """Start the lean-lsp MCP client, registering it for teardown on the stack.

    Returns the McpClient, or None if it could not be started (in which case the
    failure is recorded to the session and a warning is printed -- the run then
    continues with just the built-in tools).
    """
    try:
        from .mcp_client import McpClient

        mcp = stack.enter_context(McpClient(sandbox))
        print(
            style(f"[mcp: {len(mcp.list_tools())} lean tools available]", "gray"),
            flush=True,
        )
        return mcp
    except Exception as exc:
        session.record_warning(
            f"lean-lsp MCP unavailable: {type(exc).__name__}: {exc}"
        )
        print(
            f"{style('warning:', 'bold', 'yellow')} lean-lsp MCP unavailable "
            f"({type(exc).__name__}: {exc}); continuing with built-in tools only.",
            file=sys.stderr,
        )
        return None


def _run_footer(args) -> str:
    """A trailer describing this Gerbil run, appended to the commit message."""
    max_turns = args.max_turns if args.max_turns is not None else "unlimited"
    return (
        "authored by Gerbil:\n"
        f"--model {args.model}\n"
        f"--max-turns {max_turns}"
    )


if __name__ == "__main__":
    main()
