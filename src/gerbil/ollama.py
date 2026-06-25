"""Host-side ollama server management for the local-LLM provider.

The ollama provider (`providers._stream_ollama`) talks to an ollama server over
its OpenAI-compatible HTTP API. That server runs on the *host* (not in the Lean
sandbox), so before the agent loop makes its first call we must ensure one is
reachable. If the user already has `ollama serve` running we reuse it untouched;
otherwise we start one as a child process and -- crucially -- stop only the one
we started, leaving a pre-existing server alone.

This module is host-only and never touches the container. It uses nothing beyond
the standard library; the openai SDK (already a core dep) does the actual talking.
"""

import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

# The conventional ollama listen address. `ollama serve` and the `ollama` CLI both
# honor the OLLAMA_HOST env var, so we mirror that here for the base URL.
OLLAMA_DEFAULT_HOST = "http://localhost:11434"

# How long to wait for a freshly-spawned `ollama serve` to start answering.
_STARTUP_TIMEOUT = 30.0
_POLL_INTERVAL = 0.4

# A single short timeout for the small JSON GETs we make (health + tag list).
_HTTP_TIMEOUT = 2.0


def is_ollama_model(model: str) -> bool:
    """True for the `ollama:<NAME>` model syntax that selects this provider."""
    return model.startswith("ollama:")


def ollama_model_name(model: str) -> str:
    """The bare model name ollama expects, with the `ollama:` prefix stripped.

    Only the first `ollama:` is removed, so a tagged name survives intact:
    `ollama:qwen2.5-coder:3b` -> `qwen2.5-coder:3b`."""
    return model[len("ollama:") :] if is_ollama_model(model) else model


def ollama_base_url() -> str:
    """Base URL of the ollama server, honoring OLLAMA_HOST.

    OLLAMA_HOST may be a full URL (`http://host:port`) or a bare `host:port`
    (ollama's own convention); we normalize the latter to an http URL. Any
    trailing slash is trimmed so callers can append `/v1` or `/api/...` cleanly."""
    import os

    host = os.environ.get("OLLAMA_HOST", "").strip()
    if not host:
        return OLLAMA_DEFAULT_HOST
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/")


def _get_json(path: str):
    """GET a small JSON document from the ollama server, or raise on any failure.

    `path` is appended to the base URL (e.g. "/api/tags")."""
    url = ollama_base_url() + path
    with urllib.request.urlopen(url, timeout=_HTTP_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _server_up() -> bool:
    """Whether an ollama server is answering at the configured base URL.

    Mirrors the `_require_docker` "ping the service" pattern: any successful
    response to the cheap /api/tags endpoint means it's up; any error means it
    isn't (yet)."""
    try:
        _get_json("/api/tags")
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


class OllamaServer:
    """Context manager ensuring an ollama server is reachable for the session.

    On enter: reuse a running server if one answers; otherwise spawn `ollama serve`
    and wait until it does. On exit: terminate only a server we ourselves started
    (`_owned`), so a pre-existing one the user is running is never disturbed.

    Shaped like McpClient (`__enter__`/`__exit__` + `close`) so it slots onto the
    same `contextlib.ExitStack` the CLI already uses for teardown ordering."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._owned = False

    def __enter__(self) -> "OllamaServer":
        if _server_up():
            # Someone already has a server up -- reuse it and leave it be.
            self._owned = False
            return self

        exe = shutil.which("ollama")
        if exe is None:
            sys.exit(
                "error: --model ollama:... needs the `ollama` CLI, but it is not on "
                "PATH.\nInstall it from https://ollama.com/download, then retry."
            )

        # No server running -- start one. Discard its output (it logs verbosely);
        # the JSON-RPC we care about goes over HTTP, not stdio.
        self._proc = subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._owned = True

        deadline = time.monotonic() + _STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                # `ollama serve` exited immediately -- almost always "address
                # already in use" (a server we couldn't reach, or a race). Stop
                # claiming ownership and fail clearly.
                self._owned = False
                sys.exit(
                    "error: `ollama serve` exited immediately "
                    f"(code {self._proc.returncode}). Is another ollama already "
                    f"bound to {ollama_base_url()}?"
                )
            if _server_up():
                return self
            time.sleep(_POLL_INTERVAL)

        # Timed out waiting for readiness -- tear down our child and bail.
        self.close()
        sys.exit(
            f"error: ollama server did not become ready within "
            f"{_STARTUP_TIMEOUT:.0f}s at {ollama_base_url()}."
        )

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        """Stop the server if (and only if) we started it."""
        if not self._owned or self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def ensure_model_available(model: str) -> None:
    """Exit with a `ollama pull` hint unless the requested model is pulled locally.

    Must be called after the server is up. Queries /api/tags and checks the bare
    model name against the installed tags, tolerating the implicit `:latest` tag
    (so `ollama:qwen2.5-coder` matches an installed `qwen2.5-coder:latest`)."""
    name = ollama_model_name(model)
    try:
        tags = _get_json("/api/tags")
    except (urllib.error.URLError, OSError, ValueError):
        # If we can't list models, don't block the run on a flaky check -- the
        # provider call will surface a clear error if the model is truly absent.
        return

    installed = {m.get("name", "") for m in tags.get("models", [])}
    # Match exactly, or modulo the implicit :latest tag in either direction.
    candidates = {name}
    if ":" not in name:
        candidates.add(name + ":latest")
    elif name.endswith(":latest"):
        candidates.add(name[: -len(":latest")])
    if candidates & installed:
        return

    available = ", ".join(sorted(installed)) or "(none)"
    sys.exit(
        f"error: ollama model '{name}' is not available locally.\n"
        f"  run: ollama pull {name}\n"
        f"  installed models: {available}"
    )
