"""A tiny stdio MCP server used by test_mcp_cancel.py.

It exposes two tools:
  - slow:  sleeps for a long time; if its request is cancelled by the client
           (via an MCP `notifications/cancelled`), it records that fact by
           writing "cancelled" to the marker file passed as argv[1], then
           re-raises. This lets the test prove the server actually tore the job
           down rather than letting it linger.
  - quick: returns immediately, so the test can confirm the session is still
           usable after a cancellation.

Run as: python mock_mcp_server.py <marker_path> [pidfile]   (stdio transport)

If a pidfile is given, this process appends its PID to it at startup -- so a test
can prove a restart spawned a genuinely new server process.
"""

import asyncio
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

MARKER = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mock-mcp-marker")

# Record this process's PID so the reset test can see a new one after a restart.
if len(sys.argv) > 2:
    with open(sys.argv[2], "a") as f:
        f.write(f"{os.getpid()}\n")

mcp = FastMCP("mock")


@mcp.tool()
async def slow() -> str:
    """Sleep a long time; record + propagate if the request is cancelled."""
    try:
        await asyncio.sleep(30)
        return "completed"
    except asyncio.CancelledError:
        MARKER.write_text("cancelled")
        raise


@mcp.tool()
async def quick() -> str:
    """Return immediately."""
    return "ok"


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
