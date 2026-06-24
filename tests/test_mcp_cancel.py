"""Test that a timed-out MCP tool call is actually cancelled server-side.

Drives the real McpClient against a local mock stdio MCP server (no Docker, no
Lean). The mock's `slow` tool sleeps; when McpClient times out, it must send the
server an MCP `notifications/cancelled`, which cancels the tool -- the mock
records that by writing "cancelled" to a marker file. We also confirm the session
stays usable afterward (a follow-up `quick` call still works).

This is the regression guard for the lean_run_code wedge: a slow call (e.g. an
`import Mathlib` snippet) must not keep running on the server after we give up on
it, or it starves every later tool call.

Run with: uv run python tests/test_mcp_cancel.py
"""

import os
import sys
import time
from pathlib import Path

from mcp import StdioServerParameters

from gerbil import mcp_client
from gerbil.mcp_client import McpClient

MOCK = str(Path(__file__).with_name("mock_mcp_server.py"))


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


def main() -> None:
    marker = Path(os.environ.get("TMPDIR", "/tmp")) / f"gerbil-cancel-marker-{os.getpid()}"
    marker.unlink(missing_ok=True)

    params = StdioServerParameters(
        command=sys.executable, args=[MOCK, str(marker)], env=dict(os.environ)
    )

    # Make the `slow` tool time out fast so the test is quick.
    mcp_client.TOOL_TIMEOUTS["slow"] = 2

    with McpClient(sandbox=None, server_params=params) as mcp:
        names = {t["name"] for t in mcp.list_tools()}
        check("mock tools listed", {"slow", "quick"} <= names, str(sorted(names)))

        # Baseline: a normal call round-trips.
        r = mcp.call_tool("quick", {})
        check("quick call works", not r.is_error and "ok" in r.content, r.content)

        # The slow call must time out at ~2s (not 30s) ...
        t0 = time.time()
        r = mcp.call_tool("slow", {})
        elapsed = time.time() - t0
        check("slow call times out", r.is_error and "timed out" in r.content, r.content)
        check("slow call returns promptly (~timeout, not 30s)", elapsed < 8,
              f"{elapsed:.1f}s")
        check("timeout suggests reset_lean_server", "reset_lean_server" in r.content,
              r.content)

        # ... and the server must have CANCELLED the job (the heart of the test):
        # poll the marker the mock writes from its CancelledError handler.
        for _ in range(50):
            if marker.is_file():
                break
            time.sleep(0.1)
        check("server cancelled the in-flight job (marker written)",
              marker.is_file() and marker.read_text() == "cancelled",
              marker.read_text() if marker.is_file() else "(no marker)")

        # The session must remain usable after a cancellation.
        r = mcp.call_tool("quick", {})
        check("session still works after cancel", not r.is_error and "ok" in r.content,
              r.content)

    marker.unlink(missing_ok=True)
    print("\nMCP cancel test passed.")


if __name__ == "__main__":
    main()
