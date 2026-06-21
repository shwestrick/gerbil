#!/usr/bin/env python3
"""gerbil -- a sandboxed Lean theorem-proving agent.

Usage:
    gerbil --at DIRECTORY --prompt FILE

Outputs (in the current working directory):
    gerbil-TIMESTAMP.jsonl   session log (model, turns, token counts, tool calls)
    gerbil-TIMESTAMP.patch   git diff of changes gerbil made to the Lean project
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from agent import MODEL_PRICING, generate_commit_message, run_session
from sandbox import LeanSandbox
from session import Session


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
    args = parser.parse_args()

    project_dir = Path(args.at).resolve()
    prompt_file = Path(args.prompt).resolve()

    if not project_dir.is_dir():
        sys.exit(f"error: {project_dir} is not a directory")
    if not prompt_file.is_file():
        sys.exit(f"error: {prompt_file} is not a file")

    prompt = prompt_file.read_text()
    timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
    session_path = Path(f"gerbil-{timestamp}.jsonl")
    diff_path = Path(f"gerbil-{timestamp}.patch")
    commit_path = Path(f"gerbil-{timestamp}.commit")

    session = Session(
        path=session_path,
        model=args.model,
        project_dir=project_dir,
        prompt_file=prompt_file,
    )

    diff = ""
    try:
        with LeanSandbox(
            project_dir=project_dir, fetch_cache=not args.skip_cache
        ) as sandbox:
            run_session(
                sandbox, session, prompt, args.model, max_turns=args.max_turns
            )
            diff = sandbox.get_diff()
    finally:
        session.close()

    diff_path.write_text(diff)

    print(f"session: {session_path}")
    print(f"diff:    {diff_path}")

    # Generate a commit title + message describing the changes.
    if diff.strip():
        commit_msg = generate_commit_message(args.model, prompt, diff)
        commit_path.write_text(commit_msg + "\n")
        print(f"commit:  {commit_path}")
        print(f"\n{commit_msg.splitlines()[0]}")
    else:
        print("commit:  (no changes; skipped)")


if __name__ == "__main__":
    main()
