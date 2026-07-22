"""Unit tests for the ollama local-LLM provider plumbing.

Pure functions, no server needed: provider detection, model-name handling, base
URL resolution, pricing, and that the OpenAI streaming core was extracted without
changing the existing OpenAI path. If an ollama server happens to be reachable,
a tiny live smoke section also runs; otherwise it is skipped.

Run: uv run python tests/test_ollama.py
"""

import os
import types

from gerbil.agent import model_pricing
from gerbil.ollama import (
    _server_up,
    is_ollama_model,
    ollama_base_url,
    ollama_model_name,
)
from gerbil.providers import detect_provider


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


def test_detect_provider() -> None:
    check("ollama: prefix -> ollama", detect_provider("ollama:qwen2.5-coder") == "ollama")
    check("tagged name still ollama", detect_provider("ollama:qwen2.5-coder:3b") == "ollama")
    # Existing detections are untouched.
    check("claude still anthropic", detect_provider("claude-opus-4-7") == "anthropic")
    check("gpt still openai", detect_provider("gpt-4o") == "openai")
    check("gemini still gemini", detect_provider("gemini-2.5-pro") == "gemini")


def test_model_name() -> None:
    check("is_ollama_model true", is_ollama_model("ollama:foo"))
    check("is_ollama_model false", not is_ollama_model("gpt-4o"))
    check("strip prefix", ollama_model_name("ollama:qwen2.5-coder") == "qwen2.5-coder")
    # Only the leading ollama: is removed; an embedded :tag survives.
    check("keeps tag", ollama_model_name("ollama:qwen2.5-coder:3b") == "qwen2.5-coder:3b")
    check("non-ollama passthrough", ollama_model_name("gpt-4o") == "gpt-4o")


def test_base_url() -> None:
    saved = os.environ.get("OLLAMA_HOST")
    try:
        os.environ.pop("OLLAMA_HOST", None)
        check("default host", ollama_base_url() == "http://localhost:11434")
        os.environ["OLLAMA_HOST"] = "http://box:1234/"
        check("full url, trailing slash trimmed", ollama_base_url() == "http://box:1234")
        os.environ["OLLAMA_HOST"] = "box:1234"
        check("bare host:port normalized", ollama_base_url() == "http://box:1234")
    finally:
        if saved is None:
            os.environ.pop("OLLAMA_HOST", None)
        else:
            os.environ["OLLAMA_HOST"] = saved


def test_pricing() -> None:
    check("ollama is free (input)", model_pricing("ollama:anything")[0] == 0.0)
    check("ollama is free (output)", model_pricing("ollama:anything")[1] == 0.0)
    # A known cloud model keeps its real pricing; an unknown one has no price
    # (None -> reported as N/A), never a made-up default.
    check("known cloud priced", model_pricing("gpt-4o") == (2.50, 10.0))
    check("unknown cloud -> None", model_pricing("mystery-model") is None)


def test_openai_extraction() -> None:
    """The ollama provider and the OpenAI provider share _stream_openai_chat. A
    fake client lets us drive that core with no network: a single text delta then
    a tool call should surface as TextDelta -> ToolCall -> _ToolMeta -> Done, and
    the resolved (prefix-stripped) model name must reach the client."""
    from gerbil import providers
    from gerbil.providers import Done, TextDelta, ToolCall, _ToolMeta

    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        # Two chunks: a text delta, then a tool call closed by finish_reason.
        def fn(name, args):
            return types.SimpleNamespace(
                index=0, id="call_1",
                function=types.SimpleNamespace(name=name, arguments=args),
            )

        def choice(delta, finish=None):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=delta, finish_reason=finish)],
                usage=None,
            )

        yield choice(types.SimpleNamespace(content="hi", tool_calls=None))
        yield choice(
            types.SimpleNamespace(content=None, tool_calls=[fn("bash", '{"command":"ls"}')]),
            finish="tool_calls",
        )

    client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=fake_create)
        )
    )

    events = list(providers._stream_openai_chat(
        client, "qwen2.5-coder", "sys", [{"role": "user", "content": "hello"}],
        [{"name": "bash", "description": "run", "input_schema": {"type": "object"}}],
    ))

    check("model reached client", seen.get("model") == "qwen2.5-coder", str(seen.get("model")))
    check("tools forwarded", "tools" in seen and seen["tools"][0]["function"]["name"] == "bash")
    kinds = [type(e).__name__ for e in events]
    check("event order", kinds == ["TextDelta", "ToolCall", "_ToolMeta", "Done"], str(kinds))
    check("text delta", isinstance(events[0], TextDelta) and events[0].text == "hi")
    tc = events[1]
    check("tool call name+args", isinstance(tc, ToolCall) and tc.name == "bash"
          and tc.args == {"command": "ls"}, repr(tc))
    check("tool meta id", isinstance(events[2], _ToolMeta) and events[2].tool_use_id == "call_1")
    check("done usage", isinstance(events[3], Done))


def test_live_smoke() -> None:
    """If a server is reachable, exercise the real client path end to end for one
    short, tool-free completion. Skipped when no server is up."""
    if not _server_up():
        print("[SKIP] live smoke: no ollama server reachable")
        return
    from gerbil.ollama import _get_json
    from gerbil.providers import Done, stream

    tags = _get_json("/api/tags").get("models", [])
    if not tags:
        print("[SKIP] live smoke: server up but no models pulled")
        return
    model = "ollama:" + tags[0]["name"]
    saw_done = False
    for ev in stream(model, "Reply with exactly: ok",
                     [{"role": "user", "content": "say ok"}], []):
        if isinstance(ev, Done):
            saw_done = True
    check(f"live stream completed ({model})", saw_done)


def main() -> None:
    test_detect_provider()
    test_model_name()
    test_base_url()
    test_pricing()
    test_openai_extraction()
    test_live_smoke()
    print("\nAll ollama tests passed.")


if __name__ == "__main__":
    main()
