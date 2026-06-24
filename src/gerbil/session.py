"""Append-only JSONL session recorder.

Each line is a self-contained JSON event. The file is written incrementally,
so a crash mid-session leaves everything up to the crash intact.

Event types:
  session_start   — written once at the top
  turn            — one per LLM message (role, content, usage)
  tool_call       — one per tool invocation sent to the sandbox
  tool_result     — one per sandbox response
  session_end     — written once at the bottom with totals
  warning         — non-terminal note about a recoverable problem
  error           — terminal event if the session aborts with an exception
"""

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Session:
    def __init__(
        self,
        path: Path,
        model: str,
        project_dir: Path,
        prompt_file: Path,
        version: str = "unknown",
        base_commit: str = "",
        resumed_from: str | None = None,
        ralph: dict[str, Any] | None = None,
        ralph_done_script: str | None = None,
        include_session: bool = False,
    ):
        self.path = path
        self.model = model
        self.project_dir = project_dir
        self.prompt_file = prompt_file
        self.version = version
        self.base_commit = base_commit
        self.ralph = ralph
        self.ralph_done_script = ralph_done_script
        self.include_session = include_session
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0

        # base_commit anchors the git state this session starts from -- the HEAD
        # the agent's changes are layered on top of. It is what `--resume` checks
        # out to recreate the starting world before replaying the log.
        start = {
            "event": "session_start",
            "timestamp": _now(),
            "gerbil_version": version,
            "model": model,
            "project_dir": str(project_dir),
            "prompt_file": str(prompt_file),
            "base_commit": base_commit,
            # Whether this session's run folded its .jsonl log into the commit
            # (--include-session). Recorded so `gerbil resume` inherits the setting
            # without the user re-supplying it.
            "include_session": include_session,
        }
        if resumed_from is not None:
            start["resumed_from"] = resumed_from
        # In --ralph mode: {iteration, total, chain_base, ancestors}. chain_base
        # is the host-reachable commit the whole chain layers on; ancestors lists
        # the prior sessions' patch files, in order, that rebuild this session's
        # base. Together they let --resume reconstruct a mid-chain session without
        # reading any sibling logs.
        if ralph is not None:
            start["ralph"] = ralph
        # The --ralph_done check script's content, recorded so `--resume` can
        # rebuild a ralph chain's termination check without the user re-supplying
        # it (a command-line --ralph_done still overrides).
        if ralph_done_script is not None:
            start["ralph_done_script"] = ralph_done_script
        self._append(start)

    def record_turn(
        self,
        role: str,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
    ) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_thinking_tokens += thinking_tokens
        # thinking_tokens is a subset of output_tokens (the inclusive,
        # output-rate-billed total), recorded for reporting.
        self._append({
            "event": "turn",
            "timestamp": _now(),
            "role": role,
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
            },
        })

    def record_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        thought_signature: str | None = None,
    ) -> None:
        event = {
            "event": "tool_call",
            "timestamp": _now(),
            "name": name,
            "args": args,
        }
        # Gemini attaches a base64 thought_signature to each function call; record
        # it so --resume can replay the call faithfully (Gemini rejects history
        # whose tool calls are missing their signatures).
        if thought_signature is not None:
            event["thought_signature"] = thought_signature
        self._append(event)

    def record_tool_result(self, name: str, result: Any) -> None:
        self._append({
            "event": "tool_result",
            "timestamp": _now(),
            "name": name,
            "result": result,
        })

    def close(self) -> None:
        self._append({
            "event": "session_end",
            "timestamp": _now(),
            "total_usage": {
                "input_tokens": self._total_input_tokens,
                "output_tokens": self._total_output_tokens,
                "thinking_tokens": self._total_thinking_tokens,
            },
        })

    def record_replayed(self, event: dict[str, Any]) -> None:
        """Re-emit a prior event verbatim into this (continuation) log, tagged
        `replayed` so it is distinguishable from live activity. Used by --resume
        to make the new log self-contained (and itself resumable) by carrying the
        full pre-crash history forward. Token totals are intentionally not touched
        -- the replayed turns were already counted in the original session."""
        e = dict(event)
        e["replayed"] = True
        self._append(e)

    def record_warning(self, message: str) -> None:
        """Non-terminal event noting a recoverable problem (e.g. MCP failed to
        start so the session continued with built-in tools only)."""
        self._append({
            "event": "warning",
            "timestamp": _now(),
            "message": message,
        })

    def record_error(self, exc: BaseException) -> None:
        """Terminal event when the session aborts. Records the error details and
        the usage accumulated so far; written instead of session_end."""
        self._append({
            "event": "error",
            "timestamp": _now(),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "total_usage": {
                "input_tokens": self._total_input_tokens,
                "output_tokens": self._total_output_tokens,
                "thinking_tokens": self._total_thinking_tokens,
            },
        })

    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")
