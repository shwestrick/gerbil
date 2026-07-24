"""Tests for empty-turn protection in the providers layer.

A glitched provider turn (seen from a gateway-served Gemini: no text, no tool
calls, zero usage) used to leave {"role": "assistant", "content": []} in the
conversation; the next request then serialized it to a parts-less model turn,
which Vertex rejects with 400 INVALID_ARGUMENT -- killing the session (and any
resume of its log, which faithfully replays the empty message). Two defenses:

  1. stream() drops part-less assistant messages before any provider sees them
     (covers live conversations AND resumed logs from before the fix).
  2. _stream_openai_chat raises TransientProviderError on an empty stream with
     no finish_reason (parity with _check_gemini_finish on the native path),
     so a glitched turn is retried instead of entering the conversation.

Pure/fast: fake chunks, no network. Run with: uv run python tests/test_empty_turn.py
"""

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import gerbil.providers as providers
from gerbil.providers import (
    Done,
    TextDelta,
    ToolCall,
    TransientProviderError,
    Usage,
    _drop_empty_assistant,
    _stream_openai_chat,
)


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


def fake_client(chunks, captured):
    def create(**kwargs):
        captured.update(kwargs)
        return iter(chunks)
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def chunk(content=None, tool_calls=None, finish=None, usage=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        usage=usage, choices=[SimpleNamespace(delta=delta, finish_reason=finish)]
    )


def run(chunks, messages=None, captured=None):
    return list(_stream_openai_chat(
        fake_client(chunks, captured if captured is not None else {}),
        "m", "sys", messages or [{"role": "user", "content": "hi"}], [],
    ))


def test_drop_empty_assistant() -> None:
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": []},               # the glitched turn
        {"role": "user", "content": "nudge"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_name": "bash",
                                      "content": "", "tool_use_id": "1",
                                      "tool_call_id": "1"}]},
    ]
    out = _drop_empty_assistant(msgs)
    check("filter drops the empty assistant message",
          len(out) == 4 and all(
              not (m["role"] == "assistant" and m["content"] == []) for m in out
          ), str(out))
    check("filter keeps everything else",
          out == [msgs[0], msgs[2], msgs[3], msgs[4]])
    check("filter does not mutate the input", len(msgs) == 5)


def test_stream_applies_filter() -> None:
    """The filter is wired into stream() itself, so every provider is covered."""
    captured = {}

    def fake_provider(model, system, messages, tools):
        captured["messages"] = messages
        yield Done(Usage())

    orig = providers._stream_ollama
    providers._stream_ollama = fake_provider
    try:
        events = list(providers.stream(
            "ollama:foo", "sys",
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": []}],
            [],
        ))
    finally:
        providers._stream_ollama = orig
    check("stream() filters before dispatch",
          captured["messages"] == [{"role": "user", "content": "hi"}],
          str(captured))
    check("stream() still yields events", isinstance(events[-1], Done))


def test_empty_stream_is_transient() -> None:
    """No chunks at all (no content, no finish_reason) => retryable, not a
    silently-completed empty turn."""
    try:
        run([])
        check("empty stream raises TransientProviderError", False, "no exception")
    except TransientProviderError:
        check("empty stream raises TransientProviderError", True)


def test_clean_finish_without_content_ok() -> None:
    """A clean finish with no content is a genuine (if empty) completion --
    same policy as the native Gemini path; the serialization filter handles the
    resulting empty message."""
    events = run([chunk(finish="stop")])
    check("clean empty finish yields Done", len(events) == 1
          and isinstance(events[0], Done), str(events))


def test_normal_turns_unaffected() -> None:
    events = run([chunk(content="hel"), chunk(content="lo"), chunk(finish="stop")])
    check("text turn streams normally",
          [type(e).__name__ for e in events] == ["TextDelta", "TextDelta", "Done"],
          str(events))
    check("text preserved", events[0].text == "hel" and events[1].text == "lo")

    tc = SimpleNamespace(
        index=0, id="call-1",
        function=SimpleNamespace(name="bash", arguments='{"command": "ls"}'),
    )
    events = run([chunk(tool_calls=[tc]), chunk(finish="tool_calls")])
    check("tool-call turn streams normally",
          [type(e).__name__ for e in events] == ["ToolCall", "_ToolMeta", "Done"],
          str(events))
    check("tool call parsed", events[0].name == "bash"
          and events[0].args == {"command": "ls"})


def test_crashed_log_resumes_clean() -> None:
    """Regression for the real-world crash: a log whose zoom sub-session
    recorded an empty assistant turn (no text, no tool calls) rebuilds into a
    conversation containing an empty assistant message -- the filter must
    remove it so the resumed session's first request is valid."""
    from gerbil.resume import parse_session

    events = [
        {"event": "session_start", "model": "portkey:@x/anthropic.claude-opus-4-8",
         "small_model": "portkey:@x/gemini-3.5-flash", "inner_max_turns": 100,
         "base_commit": "abc123", "project_dir": "/repo", "prompt_file": "/p.md",
         "gerbil_version": "v1"},
        {"event": "turn", "role": "user", "content": "Prove foo."},
        {"event": "turn", "role": "assistant", "content": "Zooming."},
        {"event": "tool_call", "name": "zoom_in",
         "args": {"prompt": "fix it", "file": "Foo.lean", "line": 3}},
        {"event": "turn", "zoom": True, "role": "user", "content": "fix it..."},
        {"event": "turn", "zoom": True, "role": "assistant", "content": ""},
        {"event": "tool_call", "zoom": True, "name": "lean_goal",
         "args": {"file_path": "Foo.lean", "line": 3}},
        {"event": "tool_result", "zoom": True, "name": "lean_goal", "result": "ok"},
        # The glitched turn: empty content, no tool calls -- then the nudge.
        {"event": "turn", "zoom": True, "role": "assistant", "content": ""},
        {"event": "turn", "zoom": True, "role": "user",
         "content": "You must call the zoom_out tool (with a summary) to finish."},
    ]
    d = Path(tempfile.mkdtemp())
    p = d / "crashed.jsonl"
    with p.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")

    ps = parse_session(p)
    inner = ps.pending_zoom["messages"]
    check("rebuilt inner conversation has the empty assistant message",
          any(m["role"] == "assistant" and m["content"] == [] for m in inner),
          str(inner))
    filtered = _drop_empty_assistant(inner)
    check("filter makes the resumed conversation valid",
          not any(m["role"] == "assistant" and m["content"] == [] for m in filtered)
          and len(filtered) == len(inner) - 1,
          str(filtered))


def main() -> None:
    test_drop_empty_assistant()
    test_stream_applies_filter()
    test_empty_stream_is_transient()
    test_clean_finish_without_content_ok()
    test_normal_turns_unaffected()
    test_crashed_log_resumes_clean()
    print("\nAll empty-turn tests passed.")


if __name__ == "__main__":
    main()
