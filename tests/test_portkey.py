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

    def run_stream():
        return list(providers._stream_portkey(
            "portkey:" + CATALOG_MODEL, "sys",
            [{"role": "user", "content": "hello"}], [],
        ))

    saved = sys.modules.get("portkey_ai")
    try:
        # With a base URL: both key and URL must reach the client, and the
        # prefix-stripped model name must reach create().
        seen = {}
        sys.modules["portkey_ai"] = _fake_portkey_module(seen)
        with _env(PORTKEY_API_KEY="pk-test", PORTKEY_BASE_URL="https://gw.example/v1/"):
            events = run_stream()
        check("api key reaches client", seen["client"].get("api_key") == "pk-test")
        check("base url reaches client",
              seen["client"].get("base_url") == "https://gw.example/v1/")
        check("stripped model reaches create",
              seen["create"].get("model") == CATALOG_MODEL,
              str(seen["create"].get("model")))
        kinds = [type(e).__name__ for e in events]
        check("event stream", kinds == ["TextDelta", "Done"], str(kinds))
        check("text delta", isinstance(events[0], TextDelta) and events[0].text == "ok")
        check("done", isinstance(events[-1], Done))

        # Without PORTKEY_BASE_URL the SDK's hosted default must be left alone.
        seen = {}
        sys.modules["portkey_ai"] = _fake_portkey_module(seen)
        with _env(PORTKEY_API_KEY="pk-test", PORTKEY_BASE_URL=None):
            run_stream()
        check("no base_url when env unset", "base_url" not in seen["client"])

        # Missing key must fail up front with a pointer at the env var, not
        # deep inside the SDK.
        with _env(PORTKEY_API_KEY=None):
            try:
                run_stream()
                check("missing key raises", False)
            except RuntimeError as e:
                check("missing key raises", "PORTKEY_API_KEY" in str(e), str(e))
    finally:
        if saved is None:
            sys.modules.pop("portkey_ai", None)
        else:
            sys.modules["portkey_ai"] = saved


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
    test_live_smoke()
    print("\nAll portkey tests passed.")


if __name__ == "__main__":
    main()
