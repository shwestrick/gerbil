"""Test the reset_lean_server recovery tool.

When the Lean language server wedges, the agent can call reset_lean_server, which
restarts the MCP server (McpClient.restart). Drives the real McpClient + Toolset
against a local mock stdio MCP server (no Docker, no Lean). The mock appends its
PID to a pidfile at startup, so we can prove the restart spawned a genuinely new
server process -- and that tools still work afterward.

Run with: uv run python tests/test_reset_lean_server.py
"""

import os
import sys
import tempfile
from pathlib import Path

from mcp import StdioServerParameters

from gerbil.mcp_client import McpClient
from gerbil.tools import Toolset

MOCK = str(Path(__file__).with_name("mock_mcp_server.py"))


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


def pids(pidfile: Path) -> list[str]:
    return [ln for ln in pidfile.read_text().splitlines() if ln.strip()]


def main() -> None:
    # reset tool is only offered when MCP is on.
    no_mcp = Toolset(sandbox=None, mcp=None)
    check("no reset tool without MCP",
          "reset_lean_server" not in {t["name"] for t in no_mcp.schemas()})

    marker = Path(tempfile.mktemp())
    pidfile = Path(tempfile.mktemp())
    params = StdioServerParameters(
        command=sys.executable, args=[MOCK, str(marker), str(pidfile)],
        env=dict(os.environ),
    )

    with McpClient(sandbox=None, server_params=params) as mcp:
        toolset = Toolset(sandbox=None, mcp=mcp)
        check("reset tool offered with MCP",
              "reset_lean_server" in {t["name"] for t in toolset.schemas()})

        check("one server process so far", len(pids(pidfile)) == 1, pidfile.read_text())
        r = toolset.dispatch("quick", {})
        check("tool works before reset", not r.is_error and "ok" in r.content, r.content)

        # The recovery action: restart the server.
        r = toolset.dispatch("reset_lean_server", {})
        check("reset reports success", not r.is_error and "restarted" in r.content, r.content)

        # A genuinely new server process must have started ...
        after = pids(pidfile)
        check("a new server process was spawned", len(after) == 2 and after[0] != after[1],
              str(after))
        # ... and tools work again on the fresh server.
        r = toolset.dispatch("quick", {})
        check("tool works after reset", not r.is_error and "ok" in r.content, r.content)

    marker.unlink(missing_ok=True)
    pidfile.unlink(missing_ok=True)
    print("\nreset_lean_server test passed.")


if __name__ == "__main__":
    main()
