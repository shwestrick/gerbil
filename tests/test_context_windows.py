#!/usr/bin/env python3
"""The static context-window fallback (no network, no Docker, no API key).

providers.get_context_window asks the provider first and falls back to
context_windows.CONTEXT_WINDOWS. What matters here is that the table is keyed
so the model strings gerbil actually receives -- dated Anthropic ids, Gemini
`-preview` suffixes, Portkey catalog names, ollama prefixes -- resolve to the
right entry, and that the live query still wins when it answers.

Run: uv run python tests/test_context_windows.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gerbil import providers  # noqa: E402
from gerbil.context_windows import CONTEXT_WINDOWS, context_window  # noqa: E402
from gerbil.model_match import table_match  # noqa: E402
from gerbil.pricing import MODEL_PRICING, pricing_match  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
failures = []


def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def test_table_shape():
    print("\n-- the table --")
    check("has entries", len(CONTEXT_WINDOWS) > 100, f"got {len(CONTEXT_WINDOWS)}")
    check(
        "all values are plausible token counts",
        all(isinstance(v, int) and 1000 <= v <= 20_000_000 for v in CONTEXT_WINDOWS.values()),
    )
    check(
        "keys are API-ID-shaped (lowercase, no spaces)",
        all(k == k.lower() and " " not in k for k in CONTEXT_WINDOWS),
    )
    # benchlm splits GPT-5 into 128K and 400K variants under one name; a table
    # that picked one would be guessing.
    check("ambiguous 'gpt-5' is absent", "gpt-5" not in CONTEXT_WINDOWS)


def test_real_model_strings():
    print("\n-- model strings gerbil actually receives --")
    cases = [
        # (--model string, expected window)
        ("claude-opus-4-8", 1_000_000),
        ("claude-sonnet-4-5", 200_000),
        ("claude-opus-5", 1_000_000),
        # Anthropic appends a release date to most ids.
        ("claude-haiku-4-5-20251001", 200_000),
        # ...and benchlm writes this one "Claude 4.1 Opus", the other word order.
        ("claude-opus-4-1-20250805", 200_000),
        # Gemini ids carry a channel suffix the table's keys don't.
        ("gemini-3-pro-preview", 2_000_000),
        ("gemini-2.5-flash", 1_000_000),
        # A family base key is a substring of its variants; the longest wins.
        ("gpt-4.1-mini", 1_000_000),
        ("gpt-5.4-pro-2026-03-05", 1_050_000),
        # The case the provider can never be asked about: the real model is
        # buried inside a Portkey catalog name.
        ("portkey:@vertexai-shw8119/anthropic.claude-opus-4-8", 1_000_000),
        ("@vertexai-shw8119/anthropic.claude-opus-4-8", 1_000_000),
        # ollama prefixes the local model name.
        ("ollama:deepseek-r1", 128_000),
    ]
    for model, expected in cases:
        got = context_window(model)
        check(f"{model} -> {expected:,}", got == expected, f"got {got}")

    print("\n-- honest None rather than a guess --")
    for model in ["gpt-5", "totally-made-up-model", "o3"]:
        check(f"{model} -> None", context_window(model) is None, f"got {context_window(model)}")


def test_family_subsumption():
    print("\n-- family keys don't shadow their variants --")
    # `claude-sonnet-4` is a substring of `claude-sonnet-4-5` and
    # `claude-sonnet-4-6`; each must resolve to itself, not the shorter base.
    for model, expected_key in [
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("claude-sonnet-4-20250514", "claude-sonnet-4"),
        ("gpt-4.1-nano", "gpt-4.1-nano"),
    ]:
        got = table_match(model, CONTEXT_WINDOWS)
        check(f"{model} -> {expected_key}", got == expected_key, f"got {got}")


def test_shared_matcher():
    print("\n-- pricing still matches the way it did --")
    # pricing_match now delegates to table_match; the behaviour it documented
    # must be unchanged.
    check("exact key", pricing_match("claude-opus-4-8") == "claude-opus-4-8")
    check(
        "portkey catalog name",
        pricing_match("@vertexai-foo/anthropic.claude-opus-4-7") == "claude-opus-4-7",
    )
    check("longest subsuming key", pricing_match("@x/o3-mini") == "o3-mini")
    check("unknown model", pricing_match("no-such-model") is None)
    check(
        "every pricing key still prices itself",
        all(pricing_match(k) == k for k in MODEL_PRICING),
    )


def test_fallback_order():
    print("\n-- the live query wins; the table is only a backup --")
    providers.get_context_window.cache_clear()
    calls = []

    class StubModels:
        def retrieve(self, model):
            calls.append(model)
            return type("M", (), {"max_input_tokens": 123_456})()

    class StubAnthropic:
        def __init__(self, **kwargs):
            self.models = StubModels()

    import os

    import anthropic

    real = anthropic.Anthropic
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    # get_context_window reads the key before constructing the client, and a
    # missing one is just another failed query -> the table.
    os.environ["ANTHROPIC_API_KEY"] = "test-key"
    anthropic.Anthropic = StubAnthropic
    try:
        got = providers.get_context_window("claude-opus-4-8", "anthropic")
        # 123456 is the stub's answer; 1000000 would be the table's.
        check("provider's answer is preferred", got == 123_456, f"got {got}")
        check("provider was actually asked", calls == ["claude-opus-4-8"])

        # Now make the provider fail: the table has to cover for it.
        providers.get_context_window.cache_clear()

        class BrokenAnthropic:
            def __init__(self, **kwargs):
                raise RuntimeError("no API key")

        anthropic.Anthropic = BrokenAnthropic
        got = providers.get_context_window("claude-opus-4-8", "anthropic")
        check("table covers a failed query", got == 1_000_000, f"got {got}")
    finally:
        anthropic.Anthropic = real
        if old_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        providers.get_context_window.cache_clear()

    # A provider with no model-info endpoint at all -- the case that used to
    # report nothing.
    check(
        "openai falls straight through to the table",
        providers.get_context_window("gpt-5.4", "openai") == 1_050_000,
    )
    check(
        "a portkey catalog name resolves",
        providers.get_context_window(
            "portkey:@vertexai-shw8119/anthropic.claude-opus-4-8", "portkey"
        ) == 1_000_000,
    )
    check(
        "an unrecognizable model is still None",
        providers.get_context_window("nonsense-model-xyz", None) is None,
    )
    providers.get_context_window.cache_clear()


if __name__ == "__main__":
    test_table_shape()
    test_real_model_strings()
    test_family_subsumption()
    test_shared_matcher()
    test_fallback_order()
    print()
    if failures:
        print(f"\033[31m{len(failures)} failed:\033[0m " + ", ".join(failures))
        sys.exit(1)
    print("\033[32mall passed\033[0m")
