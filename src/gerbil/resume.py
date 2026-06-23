"""Parse a (possibly crashed) session log back into a resumable conversation.

`gerbil run --resume <session-file>` reads the append-only .jsonl a previous run
left behind, reconstructs the LLM conversation up to the point of the crash, and
hands it to the agent loop to continue. The session log records everything the
model saw -- turn text, tool calls (name + args), and the (already-truncated)
tool results -- so the conversation can be rebuilt faithfully. Two details are
synthesized here:

  - Tool-call ids (Anthropic tool_use_id / OpenAI tool_call_id) are not logged,
    but we own both the assistant tool_call and its matching user tool_result on
    replay, so consistent synthetic ids ("resume-N") suffice.
  - A turn cut off mid-execution can leave a tool_call with no recorded result;
    APIs require every tool_use to be answered, so we synthesize a placeholder
    result for any such dangling call.

The Gemini `raw_part` (which carries a thought_signature) is not recorded, so a
resumed Gemini conversation falls back to a plain function-call part; this is a
known fidelity gap for Gemini specifically.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ParsedSession:
    model: str
    base_commit: str
    prompt: str
    prompt_file: str
    project_dir: str
    version: str
    messages: list = field(default_factory=list)
    events: list = field(default_factory=list)  # turn/tool_call/tool_result, to replay
    complete: bool = False  # the convo already ended on the model (nothing pending)
    ralph: dict | None = None  # {iteration, total, chain_base, ancestors} in --ralph
    commit_message: str = ""  # the message the session generated, if it got that far


# Prefix of the message gerbil sends to request a commit message (see
# agent._commit_request). Used to locate the generated commit message in a log.
_COMMIT_REQUEST_PREFIX = "The task is complete. Here is the final git diff"


def _load_events(path: Path) -> list[dict]:
    """Read the JSONL log, tolerating a truncated/garbled final line -- a crash
    can leave a half-written record, which we simply skip."""
    events = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            # A partial trailing line from an interrupted write; ignore it.
            continue
    return events


def _rebuild_messages(events: list[dict]) -> tuple[list, bool]:
    """Rebuild the unified `messages` list (see providers.py) from log events.

    Returns (messages, complete). `complete` is True when the conversation ends
    on an assistant message with no outstanding tool results -- i.e. the model
    had finished, so there is nothing for the loop to continue.
    """
    messages: list = []
    assistant: dict | None = None
    results: list = []           # tool_result parts pending their user message
    pending: list = []           # [{id, name}] tool_calls awaiting a result
    counter = 0

    def flush_results() -> None:
        nonlocal results
        if results:
            messages.append({"role": "user", "content": results})
            results = []

    for e in events:
        kind = e.get("event")
        if e.get("replayed"):
            # Replayed history in a continuation log records the same turn/
            # tool_call/tool_result events; treat them identically.
            pass

        if kind == "turn" and e.get("role") == "user":
            flush_results()
            messages.append({"role": "user", "content": e.get("content", "")})
            assistant = None
        elif kind == "turn" and e.get("role") == "assistant":
            flush_results()
            assistant = {"role": "assistant", "content": []}
            text = e.get("content", "")
            if text:
                assistant["content"].append({"type": "text", "text": text})
            messages.append(assistant)
        elif kind == "tool_call":
            cid = f"resume-{counter}"
            counter += 1
            if assistant is None:  # defensive: a call with no preceding turn
                assistant = {"role": "assistant", "content": []}
                messages.append(assistant)
            part = {
                "type": "tool_call",
                "name": e.get("name"),
                "args": e.get("args") or {},
                "id": cid,
                "raw_part": None,
            }
            # Carry Gemini's thought_signature forward (base64); providers.py
            # rebuilds the function-call part with it so Gemini accepts the
            # replayed history.
            if e.get("thought_signature"):
                part["thought_signature"] = e["thought_signature"]
            assistant["content"].append(part)
            pending.append({"id": cid, "name": e.get("name")})
        elif kind == "tool_result":
            if pending:
                cid = pending.pop(0)["id"]
            else:  # a result with no matching call; mint a fresh id
                cid = f"resume-{counter}"
                counter += 1
            results.append({
                "type": "tool_result",
                "tool_name": e.get("name"),
                "content": e.get("result", ""),
                "tool_use_id": cid,
                "tool_call_id": cid,
            })
        # session_start / session_end / warning / error are ignored.

    # Any tool_call left without a result was cut off mid-turn; answer it so the
    # conversation is valid for the provider APIs.
    for call in pending:
        results.append({
            "type": "tool_result",
            "tool_name": call["name"],
            "content": "[resumed: tool result was not recorded before the "
                       "session was interrupted]",
            "tool_use_id": call["id"],
            "tool_call_id": call["id"],
        })
    flush_results()

    complete = bool(messages) and messages[-1]["role"] == "assistant"
    return messages, complete


def parse_session(path: Path) -> ParsedSession:
    """Parse a session .jsonl into a ParsedSession ready to continue."""
    events = _load_events(path)
    start = next((e for e in events if e.get("event") == "session_start"), None)
    if start is None:
        raise ValueError(f"{path} has no session_start event; not a session log")

    # The first user turn carries the exact prompt the session ran with -- more
    # reliable than re-reading prompt_file, which may have changed since.
    prompt = ""
    for e in events:
        if e.get("event") == "turn" and e.get("role") == "user":
            prompt = e.get("content", "")
            break

    messages, complete = _rebuild_messages(events)
    replay = [
        e for e in events
        if e.get("event") in ("turn", "tool_call", "tool_result")
    ]
    commit_message = _extract_commit_message(events)

    return ParsedSession(
        model=start.get("model", ""),
        base_commit=start.get("base_commit", ""),
        prompt=prompt,
        prompt_file=start.get("prompt_file", ""),
        project_dir=start.get("project_dir", ""),
        version=start.get("gerbil_version", "unknown"),
        messages=messages,
        events=replay,
        complete=complete,
        ralph=start.get("ralph"),
        commit_message=commit_message,
    )


def _extract_commit_message(events: list[dict]) -> str:
    """The commit message the session generated, if it reached the commit-message
    phase: the assistant turn right after gerbil's commit-message request. Empty
    if the session crashed earlier."""
    for i, e in enumerate(events):
        if (
            e.get("event") == "turn" and e.get("role") == "user"
            and str(e.get("content", "")).startswith(_COMMIT_REQUEST_PREFIX)
        ):
            for nxt in events[i + 1:]:
                if nxt.get("event") == "turn" and nxt.get("role") == "assistant":
                    return str(nxt.get("content", "")).strip()
    return ""
