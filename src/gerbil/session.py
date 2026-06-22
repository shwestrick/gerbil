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
    def __init__(self, path: Path, model: str, project_dir: Path, prompt_file: Path):
        self.path = path
        self.model = model
        self.project_dir = project_dir
        self.prompt_file = prompt_file
        self._total_input_tokens = 0
        self._total_output_tokens = 0

        self._append({
            "event": "session_start",
            "timestamp": _now(),
            "model": model,
            "project_dir": str(project_dir),
            "prompt_file": str(prompt_file),
        })

    def record_turn(
        self,
        role: str,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._append({
            "event": "turn",
            "timestamp": _now(),
            "role": role,
            "content": content,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        })

    def record_tool_call(self, name: str, args: dict[str, Any]) -> None:
        self._append({
            "event": "tool_call",
            "timestamp": _now(),
            "name": name,
            "args": args,
        })

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
            },
        })

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
            },
        })

    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")
