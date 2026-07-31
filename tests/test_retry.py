"""Tests for transient provider-error retry (agent._run_turn_with_retry).

A 503/UNAVAILABLE (or rate limit, 5xx, dropped connection) from the model
provider should be retried in place -- keeping the session warm -- instead of
crashing the run. Permanent errors must still propagate.

Pure/fast: classifies fake exceptions and drives the retry loop with a stubbed
_run_turn (retry delay set to 0). No network.

Run with: uv run python tests/test_retry.py
"""

import gerbil.agent as agent
from gerbil.agent import _is_transient_error, _run_turn_with_retry


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


class ServerError(Exception):
    """Mimics google.genai's ServerError (carries .code / .status)."""
    def __init__(self, msg, code=None, status=None):
        super().__init__(msg)
        self.code = code
        self.status = status


class StatusError(Exception):
    """Mimics anthropic/openai APIStatusError (carries .status_code)."""
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


def main() -> None:
    # --- classification: transient, by exception class ---------------------
    # A stream cut off mid-body raises a transport exception with no status code
    # and a message matching none of the phrase markers. This is the real
    # httpx/anthropic/portkey failure that aborted a live session; it must be
    # retryable on class alone. Built from the real SDKs so the class names this
    # relies on cannot drift silently.
    import httpx
    import openai

    transport = [
        httpx.RemoteProtocolError(
            "peer closed connection without sending complete message body "
            "(incomplete chunked read)"
        ),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("connection failed"),
        openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
        ConnectionResetError("Connection reset by peer"),
    ]
    for e in transport:
        check(f"transient class: {type(e).__name__}", _is_transient_error(e), repr(e))

    # ...but a real HTTP response is NOT a transport error: httpx.HTTPStatusError
    # must stay subject to the status-code rules (permanent for a 401).
    unauthorized = httpx.HTTPStatusError(
        "401 Unauthorized",
        request=httpx.Request("POST", "http://x"),
        response=httpx.Response(401),
    )
    check("permanent: httpx.HTTPStatusError 401", not _is_transient_error(unauthorized))

    # --- classification: transient ---
    transient = [
        ServerError("503 UNAVAILABLE. {'error': {'code': 503, 'status': "
                    "'UNAVAILABLE'}}", code=503, status="UNAVAILABLE"),
        ServerError("429 RESOURCE_EXHAUSTED", code=429, status="RESOURCE_EXHAUSTED"),
        StatusError("overloaded_error", status_code=529),
        StatusError("internal server error", status_code=500),
        Exception("The service is currently unavailable"),
        Exception("Connection reset by peer"),
        type("RateLimitError", (Exception,), {})("rate limit exceeded, try again"),
    ]
    for e in transient:
        check(f"transient: {type(e).__name__}: {str(e)[:40]}", _is_transient_error(e), repr(e))

    # --- classification: permanent (must NOT retry) ---
    permanent = [
        StatusError("400 invalid request", status_code=400),
        StatusError("401 unauthorized", status_code=401),
        ServerError("404 NOT_FOUND", code=404, status="NOT_FOUND"),
        Exception("invalid api key"),
        Exception("context length exceeded: 1,200,000 tokens"),
        ValueError("Can't detect provider for model 'foo'"),
    ]
    for e in permanent:
        check(f"permanent: {type(e).__name__}: {str(e)[:40]}", not _is_transient_error(e), repr(e))

    # --- retry loop: fail twice (transient), then succeed ---
    agent.RETRY_DELAY_SECONDS = 0  # don't actually sleep in the test
    state = {"n": 0}

    def flaky(model, system, messages, tools, provider, read_file=None):
        state["n"] += 1
        if state["n"] < 3:
            raise ServerError("503 UNAVAILABLE", code=503, status="UNAVAILABLE")
        return (["parts"], [], "done", "usage")

    agent._run_turn = flaky
    result = _run_turn_with_retry("m", "sys", [], [], "prov")
    check("retried transient until success", state["n"] == 3 and result[2] == "done",
          f"calls={state['n']}, result={result}")

    # --- retry loop: permanent error propagates immediately (no retry) ---
    state["n"] = 0

    def permanent_fail(*a, **k):
        state["n"] += 1
        raise ValueError("permanent boom")

    agent._run_turn = permanent_fail
    try:
        _run_turn_with_retry("m", "sys", [], [], "prov")
        check("permanent error propagates", False, "no exception raised")
    except ValueError:
        check("permanent error propagates", state["n"] == 1, f"calls={state['n']}")

    print("\nretry tests passed.")


if __name__ == "__main__":
    main()
