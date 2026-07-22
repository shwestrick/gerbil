"""Unit tests for the Portkey gateway provider plumbing.

Pure functions plus a faked-SDK streaming pass, no network needed: provider
detection (both the `portkey:` prefix and Portkey's bare `@provider/model`
catalog syntax), model-name stripping, and that _stream_portkey builds the
client from PORTKEY_API_KEY / PORTKEY_BASE_URL and hands the stripped model
name to the shared OpenAI-compatible streaming core. If PORTKEY_API_KEY (and a
PORTKEY_TEST_MODEL to name a routable model) is set, a tiny live smoke section
also runs; otherwise it is skipped.

Run: uv run python tests/test_portkey.py
"""

import contextlib
import os
import sys
import types

from gerbil.providers import detect_provider, portkey_model_name

CATALOG_MODEL = "@vertexai-foo/anthropic.claude-opus-4-8"
# A non-Anthropic catalog model: exercises the OpenAI-compatible route directly
# (Anthropic-family models now try the gateway's native /v1/messages first).
OPENAI_CATALOG_MODEL = "@openai-foo/gpt-4o"


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


@contextlib.contextmanager
def _env(**overrides):
    """Temporarily set (value) or unset (None) environment variables."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_detect_provider() -> None:
    check("portkey: prefix -> portkey", detect_provider("portkey:" + CATALOG_MODEL) == "portkey")
    check("bare @catalog name -> portkey", detect_provider(CATALOG_MODEL) == "portkey")
    # Existing detections are untouched.
    check("claude still anthropic", detect_provider("claude-opus-4-7") == "anthropic")
    check("gpt still openai", detect_provider("gpt-4o") == "openai")
    check("gemini still gemini", detect_provider("gemini-2.5-pro") == "gemini")
    check("ollama still ollama", detect_provider("ollama:qwen2.5-coder") == "ollama")


def test_model_name() -> None:
    check("strip prefix", portkey_model_name("portkey:" + CATALOG_MODEL) == CATALOG_MODEL)
    # A bare catalog name is already in gateway syntax and passes through.
    check("bare name passthrough", portkey_model_name(CATALOG_MODEL) == CATALOG_MODEL)
    check("non-portkey passthrough", portkey_model_name("gpt-4o") == "gpt-4o")


def test_pricing() -> None:
    """A catalog name embeds the real model name, so a unique MODEL_PRICING key
    found inside the string prices it; zero or ambiguous matches mean N/A."""
    from gerbil.agent import MODEL_PRICING, model_pricing, pricing_match

    priced = "portkey:@vertexai-foo/anthropic.claude-opus-4-7"
    check("unique substring match found", pricing_match(priced) == "claude-opus-4-7")
    check("unique substring match priced",
          model_pricing(priced) == MODEL_PRICING["claude-opus-4-7"])
    # Bare catalog syntax prices the same way.
    check("bare catalog name priced",
          model_pricing("@vertexai-foo/anthropic.claude-opus-4-7")
          == MODEL_PRICING["claude-opus-4-7"])
    # No embedded known model -> unknown (None), reported as N/A downstream.
    check("no match -> None", model_pricing("portkey:@foo/unheard-of-model") is None)
    # Several embedded known models -> ambiguous -> also None.
    ambiguous = "portkey:@foo/o3-versus-gpt-4o"
    check("ambiguous is detected", pricing_match(ambiguous) is None)
    check("ambiguous -> None", model_pricing(ambiguous) is None)
    # Nested family keys are NOT ambiguous: the longest match subsumes the
    # shorter ones riding along inside it, so the most specific key wins.
    check("nested: o3-mini beats o3",
          pricing_match("portkey:@foo/o3-mini") == "o3-mini")
    check("nested: mini not mispriced as base",
          model_pricing("@foo/gpt-4.1-mini") == MODEL_PRICING["gpt-4.1-mini"])
    check("nested: dated pro key beats gpt-5.4 and gpt-5.4-pro",
          pricing_match("@foo/gpt-5.4-pro-2026-03-05") == "gpt-5.4-pro-2026-03-05")
    check("nested: flash-lite beats flash",
          pricing_match("@foo/gemini-3.5-flash-lite") == "gemini-3.5-flash-lite")
    # Exact table entries and ollama's free pricing are untouched.
    check("exact model still priced", model_pricing("gpt-4o") == (2.50, 10.0))
    check("ollama still free", model_pricing("ollama:anything") == (0.0, 0.0))


def _fake_portkey_module(seen: dict):
    """A stand-in portkey_ai module whose Portkey() records its kwargs and whose
    chat.completions.create records the request and streams one text chunk --
    enough to prove client construction and the hand-off to the shared core."""

    def fake_create(**kwargs):
        seen["create"] = kwargs

        def choice(delta, finish=None):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(delta=delta, finish_reason=finish)],
                usage=None,
            )

        yield choice(types.SimpleNamespace(content="ok", tool_calls=None))

    def fake_portkey(**kwargs):
        seen["client"] = kwargs
        return types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=fake_create)
            )
        )

    return types.SimpleNamespace(Portkey=fake_portkey)


def test_client_construction() -> None:
    from gerbil import providers
    from gerbil.providers import Done, TextDelta

    def run_stream(model):
        return list(providers._stream_portkey(
            model, "sys", [{"role": "user", "content": "hello"}], [],
        ))

    saved = sys.modules.get("portkey_ai")
    try:
        # With a base URL: both key and URL must reach the client, and the
        # prefix-stripped model name must reach create().
        seen = {}
        sys.modules["portkey_ai"] = _fake_portkey_module(seen)
        with _env(PORTKEY_API_KEY="pk-test", PORTKEY_BASE_URL="https://gw.example/v1/"):
            events = run_stream("portkey:" + OPENAI_CATALOG_MODEL)
        check("api key reaches client", seen["client"].get("api_key") == "pk-test")
        check("base url reaches client",
              seen["client"].get("base_url") == "https://gw.example/v1/")
        check("strict compliance requested",
              seen["client"].get("strict_open_ai_compliance") is True)
        check("stripped model reaches create",
              seen["create"].get("model") == OPENAI_CATALOG_MODEL,
              str(seen["create"].get("model")))
        kinds = [type(e).__name__ for e in events]
        check("event stream", kinds == ["TextDelta", "Done"], str(kinds))
        check("text delta", isinstance(events[0], TextDelta) and events[0].text == "ok")
        check("done", isinstance(events[-1], Done))

        # Without PORTKEY_BASE_URL the SDK's hosted default must be left alone.
        seen = {}
        sys.modules["portkey_ai"] = _fake_portkey_module(seen)
        with _env(PORTKEY_API_KEY="pk-test", PORTKEY_BASE_URL=None):
            run_stream(OPENAI_CATALOG_MODEL)
        check("no base_url when env unset", "base_url" not in seen["client"])

        # Missing key must fail up front with a pointer at the env var, not
        # deep inside the SDK.
        with _env(PORTKEY_API_KEY=None):
            try:
                run_stream("portkey:" + CATALOG_MODEL)
                check("missing key raises", False)
            except RuntimeError as e:
                check("missing key raises", "PORTKEY_API_KEY" in str(e), str(e))
    finally:
        if saved is None:
            sys.modules.pop("portkey_ai", None)
        else:
            sys.modules["portkey_ai"] = saved


def _chunk(content=None, tool_calls=None, finish=None):
    delta = types.SimpleNamespace(content=content, tool_calls=tool_calls)
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(delta=delta, finish_reason=finish)],
        usage=None,
    )


def _tc_delta(index=0, id=None, name=None, arguments=None):
    fn = types.SimpleNamespace(name=name, arguments=arguments)
    return types.SimpleNamespace(index=index, id=id, function=fn)


def _fake_client(chunks):
    def create(**kwargs):
        yield from chunks

    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create)
        )
    )


def test_tool_call_flush() -> None:
    """The shared streaming core must surface accumulated tool calls whatever
    terminal finish_reason label the server used. Portkey's gateway (in its
    default non-strict mode) forwards Anthropic's raw "tool_use" verbatim, and
    the old exact-match gate on "tool_calls" silently dropped the call --
    making the agent look done after its very first turn."""
    from gerbil import providers
    from gerbil.providers import ToolCall, _ToolMeta

    def events_for(chunks):
        return list(providers._stream_openai_chat(
            _fake_client(chunks), "m", "sys",
            [{"role": "user", "content": "hi"}], [],
        ))

    call = [_tc_delta(0, id="c1", name="read_file",
                      arguments='{"path": "PLAN.md"}')]

    def check_flushed(label, events):
        kinds = [type(e).__name__ for e in events]
        check(label, kinds == ["TextDelta", "ToolCall", "_ToolMeta", "Done"],
              str(kinds))
        tc = events[1]
        check(label + ": call content",
              isinstance(tc, ToolCall) and tc.name == "read_file"
              and tc.args == {"path": "PLAN.md"})
        check(label + ": tool id",
              isinstance(events[2], _ToolMeta)
              and events[2].tool_use_id == "c1")

    # The spec-compliant label still works.
    check_flushed("finish=tool_calls", events_for([
        _chunk(content="reading"), _chunk(tool_calls=call),
        _chunk(finish="tool_calls"),
    ]))
    # Anthropic's raw label, as a non-strict gateway forwards it.
    check_flushed("finish=tool_use", events_for([
        _chunk(content="reading"), _chunk(tool_calls=call),
        _chunk(finish="tool_use"),
    ]))
    # Any other terminal label with calls accumulated must flush too.
    check_flushed("finish=stop with calls", events_for([
        _chunk(content="reading"), _chunk(tool_calls=call),
        _chunk(finish="stop"),
    ]))
    # Safety net: stream ends without any finish_reason at all.
    check_flushed("no finish_reason", events_for([
        _chunk(content="reading"), _chunk(tool_calls=call),
    ]))

    # A genuinely tool-free turn must not grow phantom events.
    kinds = [type(e).__name__ for e in
             events_for([_chunk(content="done"), _chunk(finish="stop")])]
    check("text-only turn unchanged", kinds == ["TextDelta", "Done"],
          str(kinds))


def _fake_anthropic_module(seen: dict, events=None, raise_status=None):
    """A stand-in anthropic module whose Anthropic() records its kwargs and
    whose messages.stream records the request and replays canned events --
    enough to prove the native-route client construction, cache-breakpoint
    injection, and event/usage parsing. raise_status simulates a gateway that
    rejects the /v1/messages route."""

    class APIStatusError(Exception):
        def __init__(self, status_code):
            super().__init__(f"status {status_code}")
            self.status_code = status_code

    class _Stream:
        def __init__(self, evs):
            self._evs = evs

        def __enter__(self):
            return self._evs

        def __exit__(self, *exc):
            return False

    def stream(**kwargs):
        seen["stream"] = kwargs
        if raise_status is not None:
            raise APIStatusError(raise_status)
        return _Stream(list(events or []))

    def fake_client(**kwargs):
        seen["client"] = kwargs
        return types.SimpleNamespace(
            messages=types.SimpleNamespace(stream=stream)
        )

    return types.SimpleNamespace(Anthropic=fake_client, APIStatusError=APIStatusError)


def _anthropic_events():
    ns = types.SimpleNamespace
    return [
        ns(type="message_start", message=ns(usage=ns(
            input_tokens=100,
            cache_read_input_tokens=5000,
            cache_creation_input_tokens=250,
        ))),
        ns(type="content_block_delta", delta=ns(type="text_delta", text="hi")),
        ns(type="content_block_stop"),
        ns(type="content_block_start",
           content_block=ns(type="tool_use", name="read_file", id="t1")),
        ns(type="content_block_delta",
           delta=ns(type="input_json_delta", partial_json='{"path": "PLAN.md"}')),
        ns(type="content_block_stop"),
        ns(type="message_delta",
           usage=ns(output_tokens=42, output_tokens_details=None)),
    ]


@contextlib.contextmanager
def _modules(**mods):
    """Temporarily install fake modules in sys.modules."""
    saved = {name: sys.modules.get(name) for name in mods}
    try:
        for name, mod in mods.items():
            sys.modules[name] = mod
        yield
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


def test_gateway_helpers() -> None:
    from gerbil.providers import _portkey_gateway_root, _portkey_split_catalog

    check("root strips /v1/",
          _portkey_gateway_root("https://gw.example/v1/") == "https://gw.example")
    check("root strips /v1",
          _portkey_gateway_root("https://gw.example/v1") == "https://gw.example")
    check("root without /v1 untouched",
          _portkey_gateway_root("https://gw.example") == "https://gw.example")
    check("root default is hosted portkey",
          _portkey_gateway_root("") == "https://api.portkey.ai")
    check("catalog splits",
          _portkey_split_catalog(CATALOG_MODEL)
          == ("@vertexai-foo", "anthropic.claude-opus-4-8"))
    check("non-catalog passthrough",
          _portkey_split_catalog("claude-opus-4-8") == (None, "claude-opus-4-8"))


def test_native_route() -> None:
    """An Anthropic-family catalog model takes the gateway's /v1/messages route:
    anthropic client pointed at the gateway root, provider header from the
    catalog name, cache breakpoints injected, cache usage parsed."""
    from gerbil import providers
    from gerbil.providers import Done, TextDelta, ToolCall, _ToolMeta

    seen = {}
    fake_an = _fake_anthropic_module(seen, events=_anthropic_events())
    with _modules(anthropic=fake_an, portkey_ai=_fake_portkey_module({})):
        with _env(PORTKEY_API_KEY="pk-test", PORTKEY_BASE_URL="https://gw.example/v1/"):
            events = list(providers._stream_portkey(
                CATALOG_MODEL, "sys",
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "reading"},
                        {"type": "tool_call", "id": "t0", "name": "read_file",
                         "args": {"path": "PLAN.md"}},
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "t0",
                         "content": "out"},
                    ]},
                ],
                [],
            ))

    client = seen["client"]
    check("gateway root strips /v1",
          client.get("base_url") == "https://gw.example")
    check("portkey key as auth token", client.get("auth_token") == "pk-test")
    headers = client.get("default_headers") or {}
    check("x-portkey-api-key header",
          headers.get("x-portkey-api-key") == "pk-test")
    check("provider header from catalog",
          headers.get("x-portkey-provider") == "@vertexai-foo")

    req = seen["stream"]
    check("bare model in request",
          req.get("model") == "anthropic.claude-opus-4-8",
          str(req.get("model")))
    sys_blocks = req.get("system")
    check("system is a block list with a breakpoint",
          isinstance(sys_blocks, list)
          and sys_blocks[-1].get("cache_control") == {"type": "ephemeral"})
    msgs = req.get("messages")
    check("moving breakpoint on last block",
          msgs[-1]["content"][-1].get("cache_control") == {"type": "ephemeral"})
    check("earlier messages unmarked", msgs[0]["content"] == "hello")

    kinds = [type(e).__name__ for e in events]
    check("native event stream",
          kinds == ["TextDelta", "ToolCall", "_ToolMeta", "Done"], str(kinds))
    tc = events[1]
    check("tool call parsed",
          isinstance(tc, ToolCall) and tc.name == "read_file"
          and tc.args == {"path": "PLAN.md"})
    check("tool id parsed",
          isinstance(events[2], _ToolMeta) and events[2].tool_use_id == "t1")
    usage = events[-1].usage
    check("uncached input", usage.input_tokens == 100)
    check("cache read tokens", usage.cache_read_tokens == 5000)
    check("cache write tokens", usage.cache_write_tokens == 250)
    check("output tokens", usage.output_tokens == 42)


def test_native_fallback() -> None:
    """A gateway without the /v1/messages route (404 on first use) falls back to
    the OpenAI-compatible dialect; a real error (500) propagates instead."""
    from gerbil import providers

    seen_pk = {}
    fake_an = _fake_anthropic_module({}, raise_status=404)
    with _modules(anthropic=fake_an, portkey_ai=_fake_portkey_module(seen_pk)):
        with _env(PORTKEY_API_KEY="pk-test", PORTKEY_BASE_URL="https://gw.example/v1/"):
            events = list(providers._stream_portkey(
                "portkey:" + CATALOG_MODEL, "sys",
                [{"role": "user", "content": "hello"}], [],
            ))
    check("fallback reaches chat completions",
          seen_pk.get("create", {}).get("model") == CATALOG_MODEL)
    kinds = [type(e).__name__ for e in events]
    check("fallback event stream", kinds == ["TextDelta", "Done"], str(kinds))

    fake_an = _fake_anthropic_module({}, raise_status=500)
    with _modules(anthropic=fake_an, portkey_ai=_fake_portkey_module({})):
        with _env(PORTKEY_API_KEY="pk-test", PORTKEY_BASE_URL="https://gw.example/v1/"):
            try:
                list(providers._stream_portkey(
                    CATALOG_MODEL, "sys",
                    [{"role": "user", "content": "hello"}], [],
                ))
                check("500 propagates", False)
            except Exception as e:
                check("500 propagates",
                      getattr(e, "status_code", None) == 500, repr(e))


def test_estimate_cost() -> None:
    """Cache reads/writes bill at their multiplier of the input rate, on top of
    the uncached input and output."""
    from gerbil.agent import estimate_cost

    # claude-opus-4-8 prices at (5.0, 25.0) per MTok.
    check("plain input+output",
          estimate_cost("claude-opus-4-8", 1_000_000, 1_000_000) == 30.0)
    check("cache read at 0.1x",
          abs(estimate_cost("claude-opus-4-8", 0, 0, cache_read_tokens=1_000_000)
              - 0.5) < 1e-9)
    check("cache write at 1.25x",
          abs(estimate_cost("claude-opus-4-8", 0, 0, cache_write_tokens=1_000_000)
              - 6.25) < 1e-9)
    check("all components",
          abs(estimate_cost("claude-opus-4-8", 100_000, 10_000,
                            cache_read_tokens=900_000, cache_write_tokens=20_000)
              - (0.5 + 0.25 + 0.45 + 0.125)) < 1e-9)
    check("unknown model -> None",
          estimate_cost("mystery-model-xyz", 1, 1) is None)


def test_live_smoke() -> None:
    """If PORTKEY_API_KEY and PORTKEY_TEST_MODEL are set, exercise the real SDK
    path end to end for one short, tool-free completion. Skipped otherwise."""
    if not os.environ.get("PORTKEY_API_KEY"):
        print("[SKIP] live smoke: PORTKEY_API_KEY not set")
        return
    model = os.environ.get("PORTKEY_TEST_MODEL")
    if not model:
        print("[SKIP] live smoke: PORTKEY_TEST_MODEL not set (e.g. @provider/model)")
        return
    from gerbil.providers import Done, stream

    saw_done = False
    for ev in stream("portkey:" + portkey_model_name(model), "Reply with exactly: ok",
                     [{"role": "user", "content": "say ok"}], []):
        if isinstance(ev, Done):
            saw_done = True
    check(f"live stream completed ({model})", saw_done)


def main() -> None:
    test_detect_provider()
    test_model_name()
    test_pricing()
    test_client_construction()
    test_tool_call_flush()
    test_gateway_helpers()
    test_native_route()
    test_native_fallback()
    test_estimate_cost()
    test_live_smoke()
    print("\nAll portkey tests passed.")


if __name__ == "__main__":
    main()
