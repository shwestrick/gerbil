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
  resumed         — continuation logs only: the boundary between the replayed
                    parent history and the live continuation (see below)

A continuation log (written by `gerbil resume`) opens with its own
session_start, then carries the parent log's events re-emitted verbatim with
`"replayed": true`, then a `resumed` marker, then its live events. That makes
the new log a complete, self-contained record of the whole session — important
because the crashed parent never commits, so its log never reaches the
project's .gerbil/; the replay is the only copy that does.

In big-small mode, turn/tool_call/tool_result events belonging to a zoomed-in
sub-session (the small model working on one sorry) carry `"zoom": true`. The
log stays a single flat stream; the tag is what lets resume and summarize
separate the inner conversation from the outer one.
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
        include_session: bool = True,
        small_model: str | None = None,
        inner_max_turns: int | None = None,
        image: str = "",
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
        self.small_model = small_model
        self.inner_max_turns = inner_max_turns
        self.image = image
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_thinking_tokens = 0
        self._total_cache_read_tokens = 0
        self._total_cache_write_tokens = 0

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
            # Whether this session's run folds its .jsonl log into the commit
            # (the default; --omit-session-log turns it off). Recorded so
            # `gerbil resume` inherits the setting without the user re-supplying it.
            "include_session": include_session,
        }
        # The sandbox image this session ran in (--image / .gerbil/config.toml
        # / the launcher's version-matched build). Recorded for provenance
        # only -- unlike model and prompt, `gerbil resume` resolves the image
        # fresh, since a recorded default tag is version-pinned and the
        # launcher prunes superseded ones.
        if image:
            start["image"] = image
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
        # Big-small mode: the small (zoom) model and the per-zoom turn cap,
        # recorded so `gerbil resume` inherits them without the user
        # re-supplying either (like model and include_session above).
        if small_model is not None:
            start["small_model"] = small_model
        if inner_max_turns is not None:
            start["inner_max_turns"] = inner_max_turns
        self._append(start)

    def record_turn(
        self,
        role: str,
        content: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        thinking_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        zoom: bool = False,
    ) -> None:
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_thinking_tokens += thinking_tokens
        self._total_cache_read_tokens += cache_read_tokens
        self._total_cache_write_tokens += cache_write_tokens
        # thinking_tokens is a subset of output_tokens (the inclusive,
        # output-rate-billed total), recorded for reporting. The cache counts
        # are ADDITIONAL prompt tokens (Anthropic semantics: input_tokens is
        # only the uncached remainder), billed at their own rates.
        event = {
            "event": "turn",
            "timestamp": _now(),
            "role": role,
            "content": content,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "cache_read_tokens": cache_read_tokens,
                "cache_write_tokens": cache_write_tokens,
            },
        }
        if zoom:
            event["zoom"] = True
        self._append(event)

    def record_tool_call(
        self,
        name: str,
        args: dict[str, Any],
        thought_signature: str | None = None,
        zoom: bool = False,
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
        if zoom:
            event["zoom"] = True
        self._append(event)

    def record_tool_result(self, name: str, result: Any, zoom: bool = False) -> None:
        event = {
            "event": "tool_result",
            "timestamp": _now(),
            "name": name,
            "result": result,
        }
        if zoom:
            event["zoom"] = True
        self._append(event)

    def close(self) -> None:
        self._append({
            "event": "session_end",
            "timestamp": _now(),
            "total_usage": {
                "input_tokens": self._total_input_tokens,
                "output_tokens": self._total_output_tokens,
                "thinking_tokens": self._total_thinking_tokens,
                "cache_read_tokens": self._total_cache_read_tokens,
                "cache_write_tokens": self._total_cache_write_tokens,
            },
        })

    def record_replayed(self, event: dict[str, Any]) -> None:
        """Re-emit a prior event verbatim into this (continuation) log, tagged
        `replayed` so it is distinguishable from live activity. Used by --resume
        to make the new log self-contained (and itself resumable) by carrying the
        parent log forward in full -- session_start, warnings, and the error that
        killed it included, not just the conversation. Token totals are
        intentionally not touched -- the replayed turns were already counted in
        the original session."""
        e = dict(event)
        e["replayed"] = True
        self._append(e)

    def record_resumed(self, resumed_from: str, replayed_events: int) -> None:
        """Boundary marker written right after the replayed parent history:
        everything above it (tagged `replayed`) was carried over from
        `resumed_from`; everything below is this run's live continuation."""
        self._append({
            "event": "resumed",
            "timestamp": _now(),
            "resumed_from": resumed_from,
            "replayed_events": replayed_events,
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
                "thinking_tokens": self._total_thinking_tokens,
                "cache_read_tokens": self._total_cache_read_tokens,
                "cache_write_tokens": self._total_cache_write_tokens,
            },
        })

    def _append(self, event: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")
