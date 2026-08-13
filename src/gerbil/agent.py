"""Agent loop — drives turns between the LLM and the sandbox until the model
stops requesting tools.

run_session() is called by the gerbil CLI once the sandbox is ready and the
mathlib cache is warm. It uses the unified provider stream (providers.stream)
and the sandbox tools (tools.dispatch), recording every turn, tool call, and
tool result to the Session as it goes.
"""

import time
from dataclasses import dataclass
from pathlib import Path

from .pricing import estimate_cost
from .prompts import (
    CONTEXT_TERMINAL,
    CONTEXT_URGENT,
    CONTEXT_WIND_DOWN,
    build_system_prompt,
    commit_request,
    context_pressure_note,
    zoom_task_prompt,
)
from .providers import (
    Done,
    TextDelta,
    ToolCall,
    TransientProviderError,
    Usage,
    _ToolMeta,
    get_context_window,
    stream,
)
from .sandbox import LeanSandbox
from .session import Session
from .render import _clip, style
from .tools import ZOOM_OUT_TOOL, Toolset, truncate_tool_output
from .view import PrintView, SessionView


@dataclass
class SessionResult:
    final_text: str       # the model's last task-phase message
    diff: str             # git patch of all changes made during the session
    commit_message: str   # generated commit title + body ("" if no changes)


# Turn cap for each zoomed-in sub-session in big-small mode (--zoom-max-turns
# overrides). Unlike the outer loop, the inner one has no natural "stopped
# calling tools" exit -- only zoom_out returns control -- so it always needs a cap.
DEFAULT_INNER_MAX_TURNS = 100


def _accumulate(dst: Usage, src: Usage) -> None:
    """Fold one turn's usage into a running total, field by field."""
    dst.input_tokens += src.input_tokens
    dst.output_tokens += src.output_tokens
    dst.thinking_tokens += src.thinking_tokens
    dst.cache_read_tokens += src.cache_read_tokens
    dst.cache_write_tokens += src.cache_write_tokens


def _context_level(usage: Usage | None, max_context: int | None) -> float | None:
    """The highest context-pressure threshold the conversation has crossed, or
    None if it is below all of them (or the window size is unknown, which is the
    case for any provider that won't report one -- gerbil then behaves exactly as
    it did before these guards existed)."""
    if usage is None or not max_context:
        return None
    fraction = usage.context_tokens / max_context
    for level in (CONTEXT_TERMINAL, CONTEXT_URGENT, CONTEXT_WIND_DOWN):
        if fraction >= level:
            return level
    return None


def _append_user_text(messages: list, text: str) -> None:
    """Add `text` to the conversation as user-visible input, merging it into the
    pending user message when there is one.

    The providers require every tool_call in an assistant turn to be answered by
    tool results in the *immediately* following user message, and reject two user
    messages in a row. So text that arrives while results are pending -- a
    context-pressure note, or the commit request when gerbil cuts a session short
    mid-turn -- has to join that message rather than follow it."""
    last = messages[-1] if messages else None
    if last is None or last["role"] != "user":
        messages.append({"role": "user", "content": text})
    elif isinstance(last["content"], list):
        last["content"] = last["content"] + [{"type": "text", "text": text}]
    else:
        last["content"] = f"{last['content']}\n\n{text}"


# Rough bytes-per-token for English text and diffs. Only ever used to decide how
# much of a diff can still fit, where the cost of being wrong one way is a
# slightly shorter diff and the other way is a failed final turn -- so it is
# deliberately pessimistic (real ratios run nearer 3.5-4).
CHARS_PER_TOKEN = 3.0
# Tokens to leave free for the commit message itself plus the request's own
# boilerplate. A commit message is a few hundred tokens; this is generous.
COMMIT_TURN_RESERVE = 2000


def _commit_diff(diff: str, max_context: int | None, usage: Usage | None) -> str:
    """The diff to show in the commit-message request, shortened if it would not
    fit in what's left of the context window.

    The final turn re-sends the whole conversation plus this diff. Normally
    there is room to spare, but a session that ran to CONTEXT_TERMINAL has only
    the last few percent left -- and a big diff there would blow the window on
    the one turn whose entire purpose is to explain the session's work before it
    is lost. Truncation is head+tail (tools.truncate_tool_output), which suits a
    diff: the first and last hunks are the most identifiable.

    Returns the diff untouched when the window is unknown or the room is ample,
    so the normal path is exactly as it was."""
    if not max_context or usage is None:
        return diff
    room = max_context - usage.context_tokens - COMMIT_TURN_RESERVE
    if room <= 0:
        # Nothing to spend. Ask for the message with the barest possible diff;
        # the model still has the whole session in view to describe.
        return truncate_tool_output(diff, 2000)
    budget = int(room * CHARS_PER_TOKEN)
    return truncate_tool_output(diff, budget)


def _usage_any(u: Usage) -> bool:
    return bool(
        u.input_tokens or u.output_tokens or u.thinking_tokens
        or u.cache_read_tokens or u.cache_write_tokens
    )


def _run_turn(model, system, messages, tools, provider, read_file=None,
              view: SessionView | None = None):
    """Stream a single turn, showing text and tool calls live via `view`.

    Returns (assistant_parts, tool_calls, text, usage).
    """
    view = view if view is not None else PrintView()
    assistant_parts = []
    current_text = ""
    tool_calls = []  # {name, args, id, raw_part}
    usage = Usage()

    for event in stream(model, system, messages, tools, provider):
        if isinstance(event, TextDelta):
            view.assistant_delta(event.text)
            current_text += event.text
        elif isinstance(event, ToolCall):
            if current_text:
                assistant_parts.append({"type": "text", "text": current_text})
                current_text = ""
            view.tool_call(event.name, event.args, read_file)
            tool_calls.append({
                "name": event.name,
                "args": event.args,
                "id": None,
                "raw_part": event.raw_part,
            })
        elif isinstance(event, _ToolMeta):
            if tool_calls:
                tool_calls[-1]["id"] = event.tool_use_id
        elif isinstance(event, Done):
            usage = event.usage

    if current_text:
        assistant_parts.append({"type": "text", "text": current_text})
    for tc in tool_calls:
        assistant_parts.append({
            "type": "tool_call",
            "name": tc["name"],
            "args": tc["args"],
            "id": tc["id"],
            "raw_part": tc["raw_part"],
        })

    text = "".join(p["text"] for p in assistant_parts if p["type"] == "text")
    return assistant_parts, tool_calls, text, usage


# How long to wait between retries of a transient provider failure.
RETRY_DELAY_SECONDS = 5

# HTTP status codes (and Google RPC status names) that mark a transient, retryable
# provider failure -- the service is up but momentarily can't serve us.
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_RETRYABLE_STATUS_NAMES = {
    "unavailable", "resource_exhausted", "internal", "aborted",
    "deadline_exceeded", "cancelled",
}
# Phrases (lowercased) that mark a transient failure when no status is exposed.
# Kept specific so genuinely permanent errors (bad request, auth, context length)
# are NOT matched and still abort the run.
_TRANSIENT_MARKERS = (
    "unavailable", "overloaded", "rate limit", "ratelimit",
    "resource exhausted", "too many requests", "try again",
    "temporarily", "internal error", "internal server error",
    "bad gateway", "gateway timeout", "service is currently",
    "connection reset", "connection aborted", "connection error",
    "server disconnected", "remote end closed", "timed out", "timeout",
    "peer closed connection", "incomplete chunked read", "incomplete read",
)

# Exception CLASSES that mark a transport-level failure: the HTTP exchange never
# completed, so no answer was produced and re-running the same turn is safe.
# Matched by name anywhere in the exception's MRO, so agent.py needs no import of
# httpx or any provider SDK (and picks up their subclasses for free).
#
# These matter because a transport failure carries no status code and its message
# need not contain any of the phrases above -- the case that motivated this was
# `httpx.RemoteProtocolError: peer closed connection without sending complete
# message body (incomplete chunked read)`, a stream cut off mid-body, which
# aborted a whole session that a retry would have carried through.
#
# Note what is NOT covered: httpx.HTTPStatusError is an HTTPError but not a
# TransportError, so real HTTP responses (401, 400, ...) still fall through to
# the status-code checks and stay permanent.
_TRANSIENT_EXC_NAMES = {
    "TransportError",        # httpx: connect/read/write/pool/protocol failures
    "ProtocolError",         # httpcore / urllib3 equivalent
    "APIConnectionError",    # openai + anthropic (APITimeoutError subclasses it)
    "ChunkedEncodingError",  # requests: stream cut mid-chunk
    "IncompleteRead",        # http.client: body shorter than advertised
    "ConnectionError",       # builtin (reset/aborted/refused) and requests'
    "TimeoutError",          # builtin, incl. socket timeouts
}


def _is_transient_error(exc: BaseException) -> bool:
    """Whether `exc` is a transient provider failure worth retrying (503/
    UNAVAILABLE, rate limits, 5xx, dropped or truncated connections).
    Conservative: anything not recognized returns False so permanent errors still
    abort the run.

    Three ways an error qualifies: it is a transport-level exception class
    (_TRANSIENT_EXC_NAMES), it carries a retryable status (_RETRYABLE_STATUS_*),
    or its message contains a transient phrase (_TRANSIENT_MARKERS)."""
    # A malformed/empty Gemini turn surfaces as no SDK exception, only a finish
    # reason; providers.py converts it to this, which is always retryable.
    if isinstance(exc, TransientProviderError):
        return True
    if any(cls.__name__ in _TRANSIENT_EXC_NAMES for cls in type(exc).__mro__):
        return True
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, bool):  # bool is an int subclass; never a status code
            continue
        if isinstance(val, int) and val in _RETRYABLE_STATUS_CODES:
            return True
        if isinstance(val, str) and val.strip().isdigit() and int(val) in _RETRYABLE_STATUS_CODES:
            return True
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS_CODES:
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.strip().lower() in _RETRYABLE_STATUS_NAMES:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


def _run_turn_with_retry(
    model, system, messages, tools, provider, read_file=None, session=None,
    view: SessionView | None = None,
):
    """Run one streamed turn, retrying transient provider failures every
    RETRY_DELAY_SECONDS until the call succeeds. Everything stays warm across the
    wait -- the session, the sandbox, and the conversation are untouched; only this
    one model call is repeated (from the same `messages`, so no state is lost).
    Non-transient errors propagate immediately, so the run still aborts (and stays
    resumable) on a real failure. A Ctrl-C during the wait aborts cleanly (it is a
    BaseException, not caught here)."""
    view = view if view is not None else PrintView()
    attempt = 0
    while True:
        started = time.monotonic()
        try:
            return _run_turn(model, system, messages, tools, provider, read_file,
                             view)
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            attempt += 1
            detail = _clip(f"{type(exc).__name__}: {exc}", 200)
            # How long the failed attempt ran is the tell for *which* kind of
            # failure this was -- an instant 503 is the provider bouncing us, but
            # a wait that matches providers.STREAM_IDLE_TIMEOUT is a stream that
            # went silent (typically a gateway sitting on a finished response).
            msg = (
                f"[provider unavailable after {time.monotonic() - started:.0f}s: "
                f"{detail}; retrying in {RETRY_DELAY_SECONDS}s (attempt {attempt})]"
            )
            view.notice(style(msg, "bold", "yellow"))
            if session is not None:
                session.record_warning(msg)
            time.sleep(RETRY_DELAY_SECONDS)


def _thought_sig(raw_part) -> str | None:
    """Base64 of a Gemini tool-call's thought_signature, for the session log.

    raw_part is the provider's original tool-call part: a google-genai Part for
    Gemini (which may carry a thought_signature), and None for every other
    provider. Returns None when there is no signature to record."""
    sig = getattr(raw_part, "thought_signature", None)
    if not sig:
        return None
    import base64

    return base64.b64encode(sig).decode()


def _update_wip_patch(
    sandbox: LeanSandbox, path: Path | None, base: str,
    view: SessionView | None = None,
) -> str | None:
    """Refresh the live resume snapshot -- a format-patch from `base` to the
    current state (committed + uncommitted), so a crash can be resumed or the
    patch applied directly. Best-effort: a checkpoint must never be the thing that
    crashes the session.

    The same patch text feeds the view (the TUI's per-file +/- table is parsed
    from it), so a live file list costs no container round-trips beyond the
    snapshot the loop was already taking. When no snapshot file is wanted, the
    patch is computed only if the view asks for it -- `wip_patch_path=None`
    keeps its long-standing meaning of "do nothing" under a PrintView.

    Returns the patch text when one was computed (the off-limits monitor
    reads the same snapshot), else None."""
    view = view if view is not None else PrintView()
    if path is None and not view.wants_wip_patch:
        return None
    try:
        text = sandbox.wip_patch(base)
    except Exception:
        return None
    view.wip_patch(text)
    if path is not None:
        try:
            path.write_text(text)
        except Exception:
            pass
    return text


def _warn_off_limits(check, wip_text, messages, session, view) -> None:
    """Fire the --fill-sorry off-limits monitor on this turn's wip snapshot
    and, when it reports newly-violated paths, put its warning where the
    model will actually see it: merged into the pending tool-result message
    (_append_user_text), recorded like a context-pressure warning, and echoed
    to the display. Warning only -- the patch gate and the termination check
    remain the enforcement; fixing the files is the model's job."""
    if check is None or wip_text is None:
        return
    note = check(wip_text)
    if not note:
        return
    _append_user_text(messages, note)
    session.record_warning(note)
    view.notice(style(
        "[warned the model: its changes modify off-limits path(s)]",
        "bold", "yellow"))


def _run_zoom(
    sandbox: LeanSandbox,
    session: Session,
    toolset: Toolset,
    provider: str | None,
    small_model: str,
    zoom_args: dict,
    inner_max_turns: int,
    wip_patch_path: Path | None,
    seed_messages: list | None = None,
    view: SessionView | None = None,
    off_limits_check=None,
) -> tuple[str, Usage]:
    """Run a zoomed-in sub-session: the small model works on the single sorry
    named by `zoom_args` (a zoom_in call's arguments) until it calls zoom_out.

    Returns (summary, usage): the zoom_out summary and the tokens the small
    model spent. Hitting `inner_max_turns` does not skip the summary -- one
    final forced turn (zoom_out as the only tool) demands it, so the big model
    always gets a real report. The caller feeds the summary back as the zoom_in
    tool result; the inner conversation itself is discarded (its file edits
    live on in the shared working tree, and its events live on in the log,
    tagged `zoom` so resume/summarize can tell them apart).

    `seed_messages` is a mid-zoom resume's rebuilt inner conversation; when
    given, it is continued as-is (its events were already replayed into the log,
    so nothing is re-recorded here)."""
    view = view if view is not None else PrintView()
    total = Usage()
    if seed_messages is None:
        prompt = zoom_task_prompt(zoom_args, inner_max_turns)
        messages = [{"role": "user", "content": prompt}]
        session.record_turn("user", prompt, zoom=True)
    else:
        messages = seed_messages

    # The sub-session runs on the plain gerbil system prompt (tool + git-state
    # notes; no ralph note -- it is not the repeating loop, and no zoom note --
    # it has no zoom_in). Everything zoom-specific lives in the task prompt the
    # big model supplied.
    system = build_system_prompt(
        bool(toolset.mcp_tool_names()), base_commit=session.base_commit,
        submodules=sandbox.submodule_paths,
    )
    tools = toolset.schemas(zoom="inner")
    max_context = get_context_window(small_model, provider)

    pos = f"{zoom_args.get('file', '?')}:{zoom_args.get('line', '?')}"
    view.zoom_begin(pos, small_model, max_context)

    # A resumed inner conversation ending on an assistant message means the
    # small model's last turn called no tools (the crash hit the nudge window);
    # asking for another assistant turn straight after would be rejected by the
    # provider APIs, so nudge exactly as the in-loop no-tool-calls path does.
    nudge = "You must call the zoom_out tool (with a summary) to finish."
    if messages and messages[-1]["role"] == "assistant":
        messages.append({"role": "user", "content": nudge})
        session.record_turn("user", nudge, zoom=True)

    last_usage: Usage | None = None
    last_text = ""
    # Continue the turn count across a resume: the seeded history holds one
    # assistant message per inner turn already run. Unlike the outer loop's
    # display-only turn_offset, this counts against the cap -- the task prompt
    # promised the small model inner_max_turns turns for the whole sub-session,
    # and a resume must not silently refresh that budget.
    turn = sum(1 for m in messages if m.get("role") == "assistant")
    while turn < inner_max_turns:
        turn += 1
        view.turn_header(
            f"zoom turn {turn}/{inner_max_turns}", "bold", "magenta",
            max_context=max_context, usage=last_usage,
        )

        assistant_parts, tool_calls, text, usage = _run_turn_with_retry(
            small_model, system, messages, tools, provider, sandbox.read_file,
            session, view,
        )
        _accumulate(total, usage)
        view.turn_complete(usage, zoom=True)
        last_usage = usage
        if text:
            last_text = text

        messages.append({"role": "assistant", "content": assistant_parts})
        session.record_turn(
            "assistant", text, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens, zoom=True,
        )

        # Unlike the outer loop, "no tool calls" is not a valid exit here --
        # only zoom_out hands control back. Nudge (counted against the cap, so
        # a model that never complies still terminates).
        if not tool_calls:
            view.notice(style("[zoom: no tool calls; nudging toward zoom_out]",
                              "yellow"))
            messages.append({"role": "user", "content": nudge})
            session.record_turn("user", nudge, zoom=True)
            continue

        tool_results = []
        for tc in tool_calls:
            session.record_tool_call(
                tc["name"], tc["args"],
                thought_signature=_thought_sig(tc["raw_part"]), zoom=True,
            )
            if tc["name"] == "zoom_out":
                summary = str(tc["args"].get("summary", ""))
                # Answer the call so the inner event stream stays well-formed
                # (every recorded call has a result) for a later resume; the
                # outer zoom_in tool_result the caller records right after this
                # is the authoritative end-of-zoom marker. Sibling calls after
                # zoom_out in the same turn are dropped: the inner conversation
                # is discarded wholesale, so no provider ever sees them dangling.
                session.record_tool_result(
                    "zoom_out", "[zoom_out received; sub-session ended]",
                    zoom=True,
                )
                view.zoom_end(style(f"===== zoom out ({turn} turns) =====",
                                    "bold", "magenta"))
                return summary, total

            result = toolset.dispatch(tc["name"], tc["args"])
            content = truncate_tool_output(result.content)
            session.record_tool_result(tc["name"], content, zoom=True)

            view.tool_result(tc["name"], content, result.content, result.is_error)

            tr = {
                "type": "tool_result",
                "tool_name": tc["name"],
                "content": content,
            }
            if tc["id"]:
                tr["tool_use_id"] = tc["id"]
                tr["tool_call_id"] = tc["id"]
            tool_results.append(tr)

        messages.append({"role": "user", "content": tool_results})

        # The sub-session mutates the same working tree; keep the crash
        # snapshot as fresh as the outer loop does -- and warn the small
        # model just like the big one when it strays onto a frozen path.
        wip_text = _update_wip_patch(
            sandbox, wip_patch_path, session.base_commit, view)
        _warn_off_limits(off_limits_check, wip_text, messages, session, view)

    # Cap hit without a zoom_out: don't just abandon the sub-session -- demand
    # the summary. One final turn where zoom_out is the only tool on offer and
    # the prompt orders it, mirroring how the outer loop's commit-message turn
    # constrains its final exchange. If the model still won't call zoom_out,
    # its final text stands in; either way the big model gets a real report,
    # prefixed with a note that the cap was hit.
    view.notice(style(
        f"[zoom: reached the {inner_max_turns}-turn cap without zoom_out; "
        "forcing a summary]", "bold", "yellow",
    ))
    request = (
        f"You have run out of turns ({inner_max_turns}). Stop working now. "
        "You MUST call zoom_out immediately with a summary of what you did, "
        "what state the sorry is in, and anything else the outer model needs "
        "to know."
    )
    messages.append({"role": "user", "content": request})
    session.record_turn("user", request, zoom=True)

    view.turn_header(
        f"zoom turn {inner_max_turns + 1}/{inner_max_turns} (forced summary)",
        "bold", "magenta", max_context=max_context, usage=last_usage,
    )
    _parts, tool_calls, text, usage = _run_turn_with_retry(
        small_model, system, messages, [ZOOM_OUT_TOOL], provider,
        sandbox.read_file, session, view,
    )
    _accumulate(total, usage)
    view.turn_complete(usage, zoom=True)
    session.record_turn(
        "assistant", text, usage.input_tokens, usage.output_tokens,
        usage.thinking_tokens, usage.cache_read_tokens,
        usage.cache_write_tokens, zoom=True,
    )

    zo = next((tc for tc in tool_calls if tc["name"] == "zoom_out"), None)
    if zo is not None:
        session.record_tool_call(
            "zoom_out", zo["args"],
            thought_signature=_thought_sig(zo["raw_part"]), zoom=True,
        )
        session.record_tool_result(
            "zoom_out", "[zoom_out received; sub-session ended]", zoom=True
        )
        summary = str(zo["args"].get("summary", ""))
    else:
        summary = text  # best effort: its final text stands in for the summary
    if not summary.strip():
        summary = (
            "(the smaller model produced no summary; its last message was: "
            f"{_clip(last_text, 500)})"
        )

    view.zoom_end(style(
        f"===== zoom out (forced after {inner_max_turns} turns) =====",
        "bold", "magenta",
    ))
    note = (
        f"[zoom_in: the smaller model used all {inner_max_turns} turns without "
        "calling zoom_out; this summary was demanded at cutoff. Its edits (if "
        "any) are in the working tree.]"
    )
    return f"{note}\n{summary}", total


def run_session(
    sandbox: LeanSandbox,
    session: Session,
    prompt: str,
    model: str,
    toolset: Toolset,
    provider: str | None = None,
    max_turns: int | None = None,
    messages: list | None = None,
    wip_patch_path: Path | None = None,
    small_model: str | None = None,
    inner_max_turns: int = DEFAULT_INNER_MAX_TURNS,
    pending_zoom: dict | None = None,
    view: SessionView | None = None,
    off_limits_check=None,
) -> SessionResult:
    """Run the agent loop until the model stops calling tools (or max_turns).

    When the model finishes the task naturally, one more turn is appended to the
    same conversation asking it to write a commit message for its diff; that turn
    is recorded and counted like any other.

    `messages`, when given, is a pre-built conversation to continue from (used by
    --resume to pick up a crashed session); the initial prompt is then assumed to
    already live in that history and is not re-recorded. `wip_patch_path`, when
    given, receives the working-tree diff after every turn, so a crash leaves a
    patch that reconstructs the tree exactly.

    `small_model`, when given, enables big-small mode: `model` (the big model)
    is offered the zoom_in tool, which hands a single sorry to a sub-session of
    the small model (see _run_zoom); each sub-session is capped at
    `inner_max_turns` turns. `pending_zoom` is a mid-zoom resume's state (see
    resume.ParsedSession.pending_zoom): the crashed session died inside a
    sub-session, so finish that sub-session first, then continue the outer
    conversation with its summary.

    `view` is where the session's output goes (see view.SessionView); it
    defaults to the classic scrolling PrintView, so every existing caller and
    test behaves exactly as before the view abstraction existed. The view is
    display-only -- what is recorded to `session` never depends on it.

    `off_limits_check`, when given (--fill-sorry with off-limits paths, see
    fill_sorry.off_limits_monitor), is fed each turn's wip-patch text; a
    warning it returns is delivered into the conversation immediately, so
    the model learns it strayed onto a frozen path the turn it happens.
    """
    view = view if view is not None else PrintView()
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
        session.record_turn("user", prompt)

    system = build_system_prompt(
        bool(toolset.mcp_tool_names()), ralph=toolset.ralph,
        base_commit=session.base_commit,
        zoom_in_available=small_model is not None,
        submodules=sandbox.submodule_paths,
        check_goal_available=bool(getattr(toolset, "check_script", None)),
    )
    tools = toolset.schemas(zoom="outer" if small_model is not None else None)

    # Query the model's maximum context window once at session start; every turn
    # then reports how close the running conversation is to filling it. None means
    # the provider doesn't report it (OpenAI), so we show raw token totals instead.
    max_context = get_context_window(model, provider)
    if max_context:
        banner = f"[context window: {max_context:,} tokens ({model})]"
    else:
        banner = f"[context window: unknown for {model}; reporting token totals only]"
    view.banner(style(banner, "gray"))

    total = Usage()
    # The small model's share of `total` (big-small mode), priced separately at
    # the end -- the two models have different rates.
    inner_total = Usage()
    turn = 0
    # Continue the turn counter across a resume: the seeded history already holds
    # this many assistant turns, so the displayed count picks up where it left off
    # instead of restarting at 1. It's display-only -- the max_turns budget below
    # still counts just this run's new turns. Zero for a fresh (non-resumed) run.
    turn_offset = sum(1 for m in messages if m.get("role") == "assistant")
    final_text = ""
    stopped_at_max = False
    # Context-pressure state. `warned_level` is the highest threshold the model
    # has already been told about: each note is delivered once, on the turn that
    # crosses its threshold. Repeating it every turn afterwards would spend the
    # very room it is warning about, and would push the earlier history further
    # out of the model's attention just as it needs to summarize it.
    warned_level = 0.0
    out_of_context = False
    # The most recent turn's usage, shown in the next turn's header so it reflects
    # how full the context is *entering* the turn. None until the first turn lands.
    last_usage: Usage | None = None

    # In a ralph loop, tag every turn header with the session counter so it stays
    # visible while scrolling, not just in the once-per-session banner.
    ralph_tag = (
        f"[ralph {session.ralph['iteration']}/{session.ralph['total']}] "
        if session.ralph else ""
    )

    # A mid-zoom resume: the crashed session died inside a zoomed-in
    # sub-session, so the rebuilt outer conversation ends on the assistant turn
    # that called zoom_in, with that call still unanswered. Finish the
    # sub-session first, then feed its summary back as the zoom_in result so
    # the outer loop below continues normally. This must run BEFORE the
    # done_already check: ending on an assistant message here means "zoom in
    # progress", not "the model was done".
    if pending_zoom is not None and small_model is not None:
        if pending_zoom.get("summary") is not None:
            # The small model had already zoomed out; only the outer zoom_in
            # result was never recorded. Nothing to re-run.
            summary = pending_zoom["summary"]
        else:
            summary, zoom_usage = _run_zoom(
                sandbox, session, toolset, provider, small_model,
                pending_zoom.get("args") or {}, inner_max_turns, wip_patch_path,
                seed_messages=pending_zoom.get("messages") or None, view=view,
                off_limits_check=off_limits_check,
            )
            _accumulate(total, zoom_usage)
            _accumulate(inner_total, zoom_usage)
        content = truncate_tool_output(summary)
        session.record_tool_result("zoom_in", content)
        # Results held from the crashed outer turn (calls dispatched before the
        # zoom_in) must share the zoom_in result's user message -- the providers
        # reject a tool_use answered across two messages.
        results = list(pending_zoom.get("held_results") or [])
        results.append({
            "type": "tool_result",
            "tool_name": "zoom_in",
            "content": content,
            "tool_use_id": pending_zoom["call_id"],
            "tool_call_id": pending_zoom["call_id"],
        })
        messages.append({"role": "user", "content": results})
        _update_wip_patch(sandbox, wip_patch_path, session.base_commit, view)

    # A resumed conversation that already ends on an assistant message had no
    # pending tool calls -- the model was done. Skip the loop (calling the model
    # again would append a second assistant turn, which the APIs reject) and go
    # straight to the commit-message phase.
    done_already = messages[-1]["role"] == "assistant" if messages else False
    if done_already:
        final_text = "".join(
            p.get("text", "")
            for p in messages[-1]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        )

    while not done_already:
        if max_turns and turn >= max_turns:
            stopped_at_max = True
            break
        turn += 1

        view.turn_header(
            f"{ralph_tag}turn {turn + turn_offset}", "bold", "dark_red",
            max_context=max_context, usage=last_usage,
        )

        assistant_parts, tool_calls, final_text, usage = _run_turn_with_retry(
            model, system, messages, tools, provider, sandbox.read_file, session,
            view,
        )
        _accumulate(total, usage)
        view.turn_complete(usage, zoom=False)
        last_usage = usage

        messages.append({"role": "assistant", "content": assistant_parts})
        session.record_turn(
            "assistant", final_text, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens,
        )

        # No tool calls => the model is done with the task.
        if not tool_calls:
            view.turn_end()
            break

        # Execute tool calls and feed results back.
        tool_results = []
        for tc in tool_calls:
            session.record_tool_call(
                tc["name"], tc["args"], thought_signature=_thought_sig(tc["raw_part"])
            )
            if small_model is not None and tc["name"] == "zoom_in":
                # Big-small mode: set the big conversation aside and run the
                # small model's sub-session; its zoom_out summary becomes this
                # call's result. The record_tool_call above / record_tool_result
                # below bracket the sub-session's zoom-tagged events in the log
                # -- resume relies on that ordering.
                summary, zoom_usage = _run_zoom(
                    sandbox, session, toolset, provider, small_model,
                    tc["args"], inner_max_turns, wip_patch_path, view=view,
                    off_limits_check=off_limits_check,
                )
                _accumulate(total, zoom_usage)
                _accumulate(inner_total, zoom_usage)
                content = truncate_tool_output(summary)
                session.record_tool_result("zoom_in", content)
                tr = {
                    "type": "tool_result",
                    "tool_name": "zoom_in",
                    "content": content,
                }
                if tc["id"]:
                    tr["tool_use_id"] = tc["id"]
                    tr["tool_call_id"] = tc["id"]
                tool_results.append(tr)
                continue
            result = toolset.dispatch(tc["name"], tc["args"])
            # Truncate once and use the same text everywhere: the session log
            # records exactly what the model sees, no more, no less.
            content = truncate_tool_output(result.content)
            session.record_tool_result(tc["name"], content)

            view.tool_result(tc["name"], content, result.content, result.is_error)

            tr = {
                "type": "tool_result",
                "tool_name": tc["name"],
                "content": content,
            }
            # Anthropic keys on tool_use_id, OpenAI on tool_call_id; set both.
            if tc["id"]:
                tr["tool_use_id"] = tc["id"]
                tr["tool_call_id"] = tc["id"]
            tool_results.append(tr)

        messages.append({"role": "user", "content": tool_results})

        # Context pressure. Measured on the turn that just landed, and acted on
        # here -- after its tool calls have been answered, because a provider
        # rejects an assistant turn whose tool_use blocks go unanswered. So even
        # the terminal threshold lets this turn's tools finish; what it stops is
        # the *next* one.
        level = _context_level(last_usage, max_context)
        if level and level > warned_level:
            warned_level = level
            note = context_pressure_note(
                level, last_usage.context_tokens, max_context
            )
            _append_user_text(messages, note)
            session.record_warning(note)
            view.notice(style(
                f"[context {last_usage.context_tokens / max_context:.0%} full: "
                + ("ending the session" if level >= CONTEXT_TERMINAL
                   else "told the model to wrap up now" if level >= CONTEXT_URGENT
                   else "told the model to start wrapping up"),
                "bold", "red" if level >= CONTEXT_URGENT else "yellow",
            ) + "]")
        if level and level >= CONTEXT_TERMINAL:
            out_of_context = True
            _update_wip_patch(sandbox, wip_patch_path, session.base_commit, view)
            break

        # Snapshot the working tree after every turn that ran tools, so an
        # interruption before the next turn leaves a patch that recreates it.
        wip_text = _update_wip_patch(
            sandbox, wip_patch_path, session.base_commit, view)
        _warn_off_limits(off_limits_check, wip_text, messages, session, view)

    if stopped_at_max:
        view.notice(style(f"[stopped: reached max_turns={max_turns}]",
                          "bold", "yellow"))

    if out_of_context:
        view.notice(style(
            f"[stopped: context window {CONTEXT_TERMINAL:.0%} full -- "
            "forcing the commit message and ending the session]", "bold", "red",
        ))

    # Final turn: ask for a commit message as a true continuation of the
    # conversation. Skipped if nothing changed, or if we bailed on max_turns
    # (the work is incomplete and the history ends on a tool result). The diff is
    # taken from the session base, so the message describes the whole session even
    # when the agent committed some of it internally.
    #
    # A context-exhausted session is the opposite case: it is cut short too, but
    # the whole point of stopping at CONTEXT_TERMINAL rather than at the API's
    # hard limit is to keep enough room for this turn -- otherwise the session's
    # work lands as an unexplained patch. The request merges into the pending
    # tool-result message (see _append_user_text), since the history ends on one.
    diff = (
        sandbox.diff_since(session.base_commit)
        if session.base_commit else sandbox.get_diff()
    )
    commit_message = ""
    if diff.strip() and not stopped_at_max:
        turn += 1
        view.turn_header(
            f"{ralph_tag}turn {turn + turn_offset} (commit message)",
            "bold", "dark_red", max_context=max_context, usage=last_usage,
        )

        request = commit_request(_commit_diff(diff, max_context, last_usage))
        _append_user_text(messages, request)
        session.record_turn("user", request)

        parts, _calls, text, usage = _run_turn_with_retry(
            model, system, messages, [], provider, session=session, view=view
        )
        _accumulate(total, usage)
        view.turn_complete(usage, zoom=False)
        last_usage = usage

        messages.append({"role": "assistant", "content": parts})
        session.record_turn(
            "assistant", text, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens,
        )
        commit_message = text.strip()
        view.turn_end()

    if _usage_any(inner_total):
        # Price the two models separately: the outer bucket is everything the
        # big model spent, the inner bucket everything the sub-sessions did.
        # Either bucket unpriced makes the whole estimate N/A (never a made-up
        # partial number).
        outer_cost = estimate_cost(
            model,
            total.input_tokens - inner_total.input_tokens,
            total.output_tokens - inner_total.output_tokens,
            total.cache_read_tokens - inner_total.cache_read_tokens,
            total.cache_write_tokens - inner_total.cache_write_tokens,
        )
        inner_cost = estimate_cost(
            small_model, inner_total.input_tokens, inner_total.output_tokens,
            inner_total.cache_read_tokens, inner_total.cache_write_tokens,
        )
        cost = (
            None if outer_cost is None or inner_cost is None
            else outer_cost + inner_cost
        )
        inner_tokens = (
            inner_total.input_tokens + inner_total.cache_read_tokens
            + inner_total.cache_write_tokens + inner_total.output_tokens
        )
        view.notice(style(
            f"[zoom sub-sessions on {small_model}: {inner_tokens:,} of the "
            "session's tokens]", "gray",
        ), newline_before=False)
    else:
        cost = estimate_cost(
            model, total.input_tokens, total.output_tokens,
            total.cache_read_tokens, total.cache_write_tokens,
        )
    view.usage_summary(turn + turn_offset, total, cost)
    return SessionResult(
        final_text=final_text, diff=diff, commit_message=commit_message
    )
