"""Parse a (possibly crashed) session log back into a resumable conversation.

`gerbil resume <session-file>` reads the append-only .jsonl a previous run
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
    ralph_done_script: str | None = None  # the --ralph_done check script, if any
    # Whether the run folded its log into the commit. False when the key is
    # absent: such logs predate the setting, and their runs did not fold it in.
    include_session: bool = False
    # Big-small mode (--small-model): the small model and per-zoom turn cap
    # recorded at session_start, so a resume restores them automatically.
    small_model: str | None = None
    inner_max_turns: int | None = None
    # Set when the log ends inside a zoomed-in sub-session (the crash happened
    # while the small model was working). The outer `messages` then end on the
    # assistant turn whose zoom_in call is still unanswered:
    #   args          the zoom_in call's arguments (the sorry position)
    #   call_id       synthetic id of that pending zoom_in call
    #   held_results  tool_result parts for calls dispatched before the zoom_in
    #                 in the same outer turn -- they must share the eventual
    #                 zoom_in result's user message
    #   messages      the rebuilt inner conversation ([] = crashed before the
    #                 sub-session's first event; start it fresh)
    #   summary       the zoom_out summary, when the small model had already
    #                 zoomed out (nothing to re-run) -- else None
    pending_zoom: dict | None = None


# Prefix of the message gerbil sends to request a commit message (see
# prompts.commit_request). Used to locate the generated commit message in a log.
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


def _rebuild_messages(
    events: list[dict], hold_pending_zoom: bool = False
) -> tuple[list, bool, dict | None]:
    """Rebuild the unified `messages` list (see providers.py) from log events.

    Returns (messages, complete, pending_zoom). `complete` is True when the
    conversation ends on an assistant message with no outstanding tool results
    -- i.e. the model had finished, so there is nothing for the loop to
    continue.

    With `hold_pending_zoom`, a trailing unanswered zoom_in call is not given a
    placeholder result: it is returned as pending_zoom ({args, call_id,
    held_results} -- see ParsedSession.pending_zoom) so the caller can finish
    the sub-session and answer the call for real. The held_results are the
    already-recorded results of earlier calls in the same turn, kept out of the
    messages because they must ride in the same user message as the eventual
    zoom_in result (the provider APIs reject a turn's tool calls answered
    across two messages). `complete` is forced False in that case. Without the
    flag (or when no zoom_in is left dangling), pending_zoom is None and
    behavior is unchanged.
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
            pending.append({"id": cid, "name": e.get("name"),
                            "args": e.get("args") or {}})
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

    # A trailing unanswered zoom_in means the crash happened inside the
    # sub-session it launched: hold it (and the same turn's already-answered
    # results) for the caller to finish, instead of placeholdering it. Only the
    # LAST pending call can be a live zoom_in -- dispatch is sequential, so any
    # earlier pending zoom_in in the same turn is genuinely dangling and falls
    # through to the placeholder path below.
    pending_zoom = None
    if hold_pending_zoom and pending and pending[-1]["name"] == "zoom_in":
        call = pending.pop()
        pending_zoom = {
            "args": call["args"],
            "call_id": call["id"],
            "held_results": None,  # filled in after the placeholders below
        }

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
    if pending_zoom is not None:
        # Do NOT flush: these results must land in the same user message as the
        # zoom_in result the caller will produce.
        pending_zoom["held_results"] = results
        results = []
    flush_results()

    complete = (
        pending_zoom is None
        and bool(messages) and messages[-1]["role"] == "assistant"
    )
    return messages, complete, pending_zoom


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

    # In big-small mode the log is a flat stream holding two conversations:
    # the outer (big model) events, and zoom-tagged inner (small model) events
    # bracketed by each zoom_in tool_call and its tool_result. The outer
    # conversation is rebuilt from the untagged events alone -- for completed
    # zooms the zoom_in call and result sit adjacent in the filtered stream and
    # pair up normally, which exactly mirrors the live loop (the inner
    # conversation is discarded once its summary is returned).
    convo = [
        e for e in events
        if e.get("event") in ("turn", "tool_call", "tool_result")
    ]
    outer = [e for e in convo if not e.get("zoom")]
    # The log ends mid-zoom when its last conversation event is inner
    # (zoom-tagged), or is the zoom_in call itself (crashed before the
    # sub-session's first event was recorded).
    mid_zoom = bool(convo) and (
        bool(convo[-1].get("zoom"))
        or (
            convo[-1].get("event") == "tool_call"
            and convo[-1].get("name") == "zoom_in"
        )
    )

    messages, complete, pending_zoom = _rebuild_messages(
        outer, hold_pending_zoom=mid_zoom
    )
    if pending_zoom is not None:
        # Rebuild the interrupted inner conversation: the zoom-tagged events
        # after the last outer zoom_in call. Its own dangling tool calls get
        # the standard placeholder treatment.
        idx = max(
            i for i, e in enumerate(convo)
            if e.get("event") == "tool_call" and e.get("name") == "zoom_in"
            and not e.get("zoom")
        )
        inner_events = [e for e in convo[idx + 1:] if e.get("zoom")]
        inner_messages, _, _ = _rebuild_messages(inner_events)
        pending_zoom["messages"] = inner_messages
        # If the small model already called zoom_out, the crash hit the window
        # between that call and the outer zoom_in result being recorded --
        # recover the summary so nothing is re-run.
        pending_zoom["summary"] = next(
            (
                str((e.get("args") or {}).get("summary", ""))
                for e in reversed(inner_events)
                if e.get("event") == "tool_call" and e.get("name") == "zoom_out"
            ),
            None,
        )

    replay = convo
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
        ralph_done_script=start.get("ralph_done_script"),
        include_session=bool(start.get("include_session", False)),
        small_model=start.get("small_model"),
        inner_max_turns=start.get("inner_max_turns"),
        pending_zoom=pending_zoom,
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
