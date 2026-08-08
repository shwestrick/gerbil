#!/usr/bin/env python3
"""Streaming idle-timeout plumbing (no network, no Docker, no API key).

The failure this guards against is expensive rather than loud: an AI gateway
holds a finished response, the stream goes silent, and the SDK's stock 600s read
timeout means gerbil waits ten minutes before retrying -- by which point the
Anthropic prompt cache the original request just paid to write (~5 minute TTL)
is gone, so the retry re-writes the whole conversation prefix at the 1.25x
premium instead of re-reading it at 0.1x. providers.STREAM_IDLE_TIMEOUT exists to
catch the stall inside the cache's lifetime, so what matters here is that the
value is under the TTL and that it actually reaches every metered client.

Run: uv run python tests/test_stream_timeout.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import httpx  # noqa: E402

from gerbil import providers  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
failures = []


def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


class _EnvVar:
    """Set (or clear) GERBIL_STREAM_TIMEOUT for the duration of a block."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.environ.get("GERBIL_STREAM_TIMEOUT")
        if self.value is None:
            os.environ.pop("GERBIL_STREAM_TIMEOUT", None)
        else:
            os.environ["GERBIL_STREAM_TIMEOUT"] = self.value
        return self

    def __exit__(self, *exc):
        if self.old is None:
            os.environ.pop("GERBIL_STREAM_TIMEOUT", None)
        else:
            os.environ["GERBIL_STREAM_TIMEOUT"] = self.old


def test_timeout_value():
    print("\n-- the timeout itself --")
    # The whole point of the override is landing inside Anthropic's ~5 minute
    # prompt-cache TTL, with room for the retry delay on top.
    check(
        "default is under the 5-minute cache TTL",
        providers.STREAM_IDLE_TIMEOUT < 300,
        f"got {providers.STREAM_IDLE_TIMEOUT}",
    )
    with _EnvVar(None):
        t = providers._stream_timeout()
        check("default read timeout", t.read == providers.STREAM_IDLE_TIMEOUT, f"got {t.read}")
        check("default write timeout", t.write == providers.STREAM_IDLE_TIMEOUT, f"got {t.write}")
        check("connect stays short", t.connect == 10.0, f"got {t.connect}")
        check("kwargs carry it", providers._timeout_kwargs() == {"timeout": t})

    with _EnvVar("45"):
        check("env override", providers._stream_timeout().read == 45.0)
    with _EnvVar("0"):
        # Absent, not None: `timeout=None` means "no timeout at all" to both SDKs,
        # which is the opposite of restoring their default.
        check("0 disables the override", providers._stream_timeout() is None)
        check("...and drops the kwarg", providers._timeout_kwargs() == {})
    with _EnvVar("-1"):
        check("negative disables the override", providers._stream_timeout() is None)
    with _EnvVar("not-a-number"):
        check(
            "garbage falls back to the default",
            providers._stream_timeout().read == providers.STREAM_IDLE_TIMEOUT,
        )


def _capture_client(module, attr):
    """Replace `module.attr` with a stub recording the client it is handed."""
    seen = {}

    def stub(client, *args, **kwargs):
        seen["client"] = client
        return iter(())

    setattr(module, attr, stub)
    return seen


def test_clients_get_the_timeout():
    print("\n-- the timeout reaches the clients --")
    real_anthropic = providers._stream_anthropic_chat
    real_openai = providers._stream_openai_chat
    seen_a = _capture_client(providers, "_stream_anthropic_chat")
    seen_o = _capture_client(providers, "_stream_openai_chat")
    old_env = {k: os.environ.get(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "PORTKEY_API_KEY", "PORTKEY_BASE_URL")}
    try:
        os.environ.update(
            ANTHROPIC_API_KEY="test-key",
            OPENAI_API_KEY="test-key",
            PORTKEY_API_KEY="test-key",
        )
        os.environ.pop("PORTKEY_BASE_URL", None)
        expected = providers.STREAM_IDLE_TIMEOUT

        list(providers._stream_anthropic("claude-x", "sys", [], []))
        check(
            "anthropic client",
            seen_a["client"].timeout.read == expected,
            f"got {seen_a['client'].timeout}",
        )

        list(providers._stream_openai("gpt-x", "sys", [], []))
        check(
            "openai client",
            seen_o["client"].timeout.read == expected,
            f"got {seen_o['client'].timeout}",
        )

        # Portkey's claude route goes through the anthropic client, pointed at
        # the gateway -- the case that motivated all of this.
        list(providers._stream_portkey("portkey:@vertexai-x/claude-opus-4-8", "sys", [], []))
        check(
            "portkey native /v1/messages client",
            seen_a["client"].timeout.read == expected,
            f"got {seen_a['client'].timeout}",
        )

        # Portkey's chat-completions route takes no timeout= kwarg, so the limit
        # has to arrive on an http_client instead.
        import portkey_ai

        captured = {}
        real_portkey = portkey_ai.Portkey

        class StubPortkey:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        portkey_ai.Portkey = StubPortkey
        try:
            list(providers._stream_portkey("portkey:gpt-4o", "sys", [], []))
        finally:
            portkey_ai.Portkey = real_portkey
        http_client = captured.get("http_client")
        check("portkey chat route gets an http_client", isinstance(http_client, httpx.Client))
        check(
            "...carrying the idle timeout",
            http_client is not None and http_client.timeout.read == expected,
            f"got {getattr(http_client, 'timeout', None)}",
        )
    finally:
        providers._stream_anthropic_chat = real_anthropic
        providers._stream_openai_chat = real_openai
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_read_timeout_is_transient():
    print("\n-- a stalled stream is retryable, not fatal --")
    from gerbil.agent import _is_transient_error

    # httpx raises this raw out of stream iteration (the SDKs only wrap timeouts
    # on the initial request), so it must be recognized without an SDK wrapper.
    check(
        "httpx.ReadTimeout retries",
        _is_transient_error(httpx.ReadTimeout("The read operation timed out")),
    )
    check(
        "httpx.ReadError retries",
        _is_transient_error(httpx.ReadError("Connection reset by peer")),
    )
    check(
        "a bad request still aborts",
        not _is_transient_error(ValueError("invalid model name")),
    )


if __name__ == "__main__":
    test_timeout_value()
    test_clients_get_the_timeout()
    test_read_timeout_is_transient()
    print()
    if failures:
        print(f"\033[31m{len(failures)} failed:\033[0m " + ", ".join(failures))
        sys.exit(1)
    print("\033[32mall passed\033[0m")
