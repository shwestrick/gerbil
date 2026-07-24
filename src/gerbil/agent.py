"""Agent loop — drives turns between the LLM and the sandbox until the model
stops requesting tools.

run_session() is called by the gerbil CLI once the sandbox is ready and the
mathlib cache is warm. It uses the unified provider stream (providers.stream)
and the sandbox tools (tools.dispatch), recording every turn, tool call, and
tool result to the Session as it goes.
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .pricing import estimate_cost
from .prompts import build_system_prompt, commit_request, zoom_task_prompt
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
from .render import (
    _clip,
    context_suffix,
    format_tool_call,
    format_tool_result,
    print_usage,
    style,
)
from .tools import Toolset, truncate_tool_output


@dataclass
class SessionResult:
    final_text: str       # the model's last task-phase message
    diff: str             # git patch of all changes made during the session
    commit_message: str   # generated commit title + body ("" if no changes)


# Turn cap for each zoomed-in sub-session in big-small mode (--zoom-max-turns
# overrides). Unlike the outer loop, the inner one has no natural "stopped
# calling tools" exit -- only zoom_out returns control -- so it always needs a cap.
DEFAULT_INNER_MAX_TURNS = 25


def _accumulate(dst: Usage, src: Usage) -> None:
    """Fold one turn's usage into a running total, field by field."""
    dst.input_tokens += src.input_tokens
    dst.output_tokens += src.output_tokens
    dst.thinking_tokens += src.thinking_tokens
    dst.cache_read_tokens += src.cache_read_tokens
    dst.cache_write_tokens += src.cache_write_tokens


def _usage_any(u: Usage) -> bool:
    return bool(
        u.input_tokens or u.output_tokens or u.thinking_tokens
        or u.cache_read_tokens or u.cache_write_tokens
    )


def _run_turn(model, system, messages, tools, provider, read_file=None):
    """Stream a single turn, printing text and tool calls live.

    Returns (assistant_parts, tool_calls, text, usage).
    """
    assistant_parts = []
    current_text = ""
    tool_calls = []  # {name, args, id, raw_part}
    usage = Usage()

    for event in stream(model, system, messages, tools, provider):
        if isinstance(event, TextDelta):
            sys.stdout.write(event.text)
            sys.stdout.flush()
            current_text += event.text
        elif isinstance(event, ToolCall):
            if current_text:
                assistant_parts.append({"type": "text", "text": current_text})
                current_text = ""
            print(
                "\n" + format_tool_call(event.name, event.args, read_file),
                flush=True,
            )
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
)


def _is_transient_error(exc: BaseException) -> bool:
    """Whether `exc` is a transient provider failure worth retrying (503/
    UNAVAILABLE, rate limits, 5xx, dropped connections). Conservative: anything
    not recognized returns False so permanent errors still abort the run."""
    # A malformed/empty Gemini turn surfaces as no SDK exception, only a finish
    # reason; providers.py converts it to this, which is always retryable.
    if isinstance(exc, TransientProviderError):
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
    model, system, messages, tools, provider, read_file=None, session=None
):
    """Run one streamed turn, retrying transient provider failures every
    RETRY_DELAY_SECONDS until the call succeeds. Everything stays warm across the
    wait -- the session, the sandbox, and the conversation are untouched; only this
    one model call is repeated (from the same `messages`, so no state is lost).
    Non-transient errors propagate immediately, so the run still aborts (and stays
    resumable) on a real failure. A Ctrl-C during the wait aborts cleanly (it is a
    BaseException, not caught here)."""
    attempt = 0
    while True:
        try:
            return _run_turn(model, system, messages, tools, provider, read_file)
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            attempt += 1
            detail = _clip(f"{type(exc).__name__}: {exc}", 200)
            msg = (
                f"[provider unavailable: {detail}; retrying in "
                f"{RETRY_DELAY_SECONDS}s (attempt {attempt})]"
            )
            print("\n" + style(msg, "bold", "yellow"), flush=True)
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


def _update_wip_patch(sandbox: LeanSandbox, path: Path | None, base: str) -> None:
    """Refresh the live resume snapshot -- a format-patch from `base` to the
    current state (committed + uncommitted), so a crash can be resumed or the
    patch applied directly. Best-effort: a checkpoint must never be the thing that
    crashes the session."""
    if path is None:
        return
    try:
        path.write_text(sandbox.wip_patch(base))
    except Exception:
        pass


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
) -> tuple[str, Usage]:
    """Run a zoomed-in sub-session: the small model works on the single sorry
    named by `zoom_args` (a zoom_in call's arguments) until it calls zoom_out.

    Returns (summary, usage): the zoom_out summary -- or a synthesized one if
    the sub-session hit `inner_max_turns` -- and the tokens the small model
    spent. The caller feeds the summary back to the big model as the zoom_in
    tool result; the inner conversation itself is discarded (its file edits
    live on in the shared working tree, and its events live on in the log,
    tagged `zoom` so resume/summarize can tell them apart).

    `seed_messages` is a mid-zoom resume's rebuilt inner conversation; when
    given, it is continued as-is (its events were already replayed into the log,
    so nothing is re-recorded here)."""
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
        bool(toolset.mcp_tool_names()), base_commit=session.base_commit
    )
    tools = toolset.schemas(zoom="inner")
    max_context = get_context_window(small_model, provider)

    pos = f"{zoom_args.get('file', '?')}:{zoom_args.get('line', '?')}"
    print(
        "\n" + style(f"===== zoom in: {pos} ({small_model}) =====", "bold", "magenta"),
        flush=True,
    )

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
    turn = 0
    while turn < inner_max_turns:
        turn += 1
        header = style(
            f"--- zoom turn {turn}/{inner_max_turns} ---", "bold", "magenta"
        )
        print("\n" + header + context_suffix(max_context, last_usage), flush=True)

        assistant_parts, tool_calls, text, usage = _run_turn_with_retry(
            small_model, system, messages, tools, provider, sandbox.read_file,
            session,
        )
        _accumulate(total, usage)
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
            print(
                "\n" + style("[zoom: no tool calls; nudging toward zoom_out]",
                             "yellow"),
                flush=True,
            )
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
                print(
                    "\n" + style(f"===== zoom out ({turn} turns) =====",
                                 "bold", "magenta"),
                    flush=True,
                )
                return summary, total

            result = toolset.dispatch(tc["name"], tc["args"])
            content = truncate_tool_output(result.content)
            session.record_tool_result(tc["name"], content, zoom=True)

            print(
                format_tool_result(
                    tc["name"], content, result.content, result.is_error
                ),
                flush=True,
            )

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
        # snapshot as fresh as the outer loop does.
        _update_wip_patch(sandbox, wip_patch_path, session.base_commit)

    print(
        "\n" + style(
            f"[zoom: reached the {inner_max_turns}-turn cap without zoom_out; "
            "returning to the outer model]", "bold", "yellow",
        ),
        flush=True,
    )
    summary = (
        f"[zoom_in aborted: the smaller model used all {inner_max_turns} turns "
        "without calling zoom_out. Its edits (if any) are in the working tree. "
        f"Its last message was: {_clip(last_text, 500)}]"
    )
    return summary, total


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
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
        session.record_turn("user", prompt)

    system = build_system_prompt(
        bool(toolset.mcp_tool_names()), ralph=toolset.ralph,
        base_commit=session.base_commit,
        zoom_in_available=small_model is not None,
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
    print(style(banner, "gray"), flush=True)

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
                seed_messages=pending_zoom.get("messages") or None,
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
        _update_wip_patch(sandbox, wip_patch_path, session.base_commit)

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

        header = style(
            f"--- {ralph_tag}turn {turn + turn_offset} ---", "bold", "dark_red"
        )
        print("\n" + header + context_suffix(max_context, last_usage), flush=True)

        assistant_parts, tool_calls, final_text, usage = _run_turn_with_retry(
            model, system, messages, tools, provider, sandbox.read_file, session
        )
        _accumulate(total, usage)
        last_usage = usage

        messages.append({"role": "assistant", "content": assistant_parts})
        session.record_turn(
            "assistant", final_text, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens,
        )

        # No tool calls => the model is done with the task.
        if not tool_calls:
            print(flush=True)
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
                    tc["args"], inner_max_turns, wip_patch_path,
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

            print(
                format_tool_result(
                    tc["name"], content, result.content, result.is_error
                ),
                flush=True,
            )

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

        # Snapshot the working tree after every turn that ran tools, so an
        # interruption before the next turn leaves a patch that recreates it.
        _update_wip_patch(sandbox, wip_patch_path, session.base_commit)

    if stopped_at_max:
        print(
            "\n" + style(f"[stopped: reached max_turns={max_turns}]", "bold", "yellow"),
            flush=True,
        )

    # Final turn: ask for a commit message as a true continuation of the
    # conversation. Skipped if nothing changed, or if we bailed on max_turns
    # (the work is incomplete and the history ends on a tool result). The diff is
    # taken from the session base, so the message describes the whole session even
    # when the agent committed some of it internally.
    diff = (
        sandbox.diff_since(session.base_commit)
        if session.base_commit else sandbox.get_diff()
    )
    commit_message = ""
    if diff.strip() and not stopped_at_max:
        turn += 1
        header = style(
            f"--- {ralph_tag}turn {turn + turn_offset} (commit message) ---",
            "bold", "dark_red",
        )
        print("\n" + header + context_suffix(max_context, last_usage), flush=True)

        request = commit_request(diff)
        messages.append({"role": "user", "content": request})
        session.record_turn("user", request)

        parts, _calls, text, usage = _run_turn_with_retry(
            model, system, messages, [], provider, session=session
        )
        _accumulate(total, usage)
        last_usage = usage

        messages.append({"role": "assistant", "content": parts})
        session.record_turn(
            "assistant", text, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens, usage.cache_read_tokens,
            usage.cache_write_tokens,
        )
        commit_message = text.strip()
        print(flush=True)

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
        print(style(
            f"[zoom sub-sessions on {small_model}: {inner_tokens:,} of the "
            "session's tokens]", "gray",
        ), flush=True)
    else:
        cost = estimate_cost(
            model, total.input_tokens, total.output_tokens,
            total.cache_read_tokens, total.cache_write_tokens,
        )
    print_usage(turn + turn_offset, total, cost)
    return SessionResult(
        final_text=final_text, diff=diff, commit_message=commit_message
    )
