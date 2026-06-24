"""Synchronous client for an MCP server running inside the sandbox container.

The Lean toolchain lives inside the Docker container, so the lean-lsp-mcp server
must run there too. We connect from the host over stdio by spawning
`docker exec -i <container> lean-lsp-mcp --transport stdio` as the MCP subprocess
(no `-t`: a TTY would corrupt the JSON-RPC framing).

The `mcp` SDK is asyncio-based but gerbil's agent loop is synchronous, so we run
the client's event loop in a background daemon thread and submit calls to it via
run_coroutine_threadsafe. The session is opened once and reused for the whole
gerbil session (lean-lsp-mcp wraps `lake serve` and is stateful + slow to init).
"""

import asyncio
import os
import threading
from concurrent.futures import TimeoutError as FuturesTimeout

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .sandbox import WORKSPACE_DIR
from .tools import ToolResult

# Per-tool timeouts (seconds). lake build and the rate-limited external search
# tools are slow; everything else uses the default.
DEFAULT_TIMEOUT = 60.0
INIT_TIMEOUT = 120.0

# lean-lsp-mcp tools that reach EXTERNAL, internet-hosted services (the mathlib
# search / premise-selection engines). The sandbox is meant to be hermetic -- the
# agent must not leave it -- and in practice these calls also trigger host
# network-permission prompts and can wedge the whole MCP server when one stalls
# (a stall there starved even local tools like lean_run_code). So gerbil never
# exposes them to the agent. lean_local_search is deliberately NOT here: it
# searches the LOCAL Lean/mathlib source with ripgrep, no network.
NETWORK_TOOLS = frozenset({
    "lean_leansearch",      # natural language -> mathlib (leansearch.net)
    "lean_loogle",          # type/name pattern -> mathlib (loogle.lean-lang.org)
    "lean_leanfinder",      # semantic/conceptual search (external service)
    "lean_state_search",    # goal -> closing lemmas (external premise search)
    "lean_hammer_premise",  # goal -> premises for simp/aesop (external selector)
})

# Slow LOCAL tools that need a longer timeout than the default. (The external
# search tools are intentionally absent -- they are filtered out, never run.)
TOOL_TIMEOUTS = {
    "lean_build": 300.0,
    "lean_profile_proof": 180.0,
}


class McpClient:
    """Synchronous façade over an asyncio MCP stdio client.

    Use as a context manager:

        with McpClient(sandbox) as mcp:
            schemas = mcp.list_tools()      # gerbil-format tool schemas
            result = mcp.call_tool(name, args)   # -> ToolResult
    """

    def __init__(self, sandbox, project_path: str = WORKSPACE_DIR):
        self._params = StdioServerParameters(
            command="docker",
            args=[
                "exec",
                "-i",  # keep stdin open; NO -t (a TTY corrupts JSON-RPC framing)
                "-w", project_path,
                "-e", f"LEAN_PROJECT_PATH={project_path}",
                sandbox.container_id,
                "lean-lsp-mcp", "--transport", "stdio",
            ],
            # Inherit the host environment so `docker` resolves exactly as it does
            # for the rest of gerbil (PATH, DOCKER_HOST, contexts, ...).
            env=dict(os.environ),
        )
        # The MCP server logs to stderr; keep it out of gerbil's clean output.
        self._errlog = open(os.devnull, "w")

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: ClientSession | None = None
        self._stop: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None
        self._schemas: list[dict] = []
        self._disabled: list[str] = []  # network tools filtered out at list time

    # ---- lifecycle -------------------------------------------------------

    def __enter__(self) -> "McpClient":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=INIT_TIMEOUT):
            self.close()
            raise TimeoutError("lean-lsp-mcp did not initialize in time")
        if self._startup_error is not None:
            self.close()
            raise self._startup_error

    def close(self) -> None:
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=15)
        try:
            self._errlog.close()
        except Exception:
            pass

    # ---- background event loop ------------------------------------------

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        # One coroutine owns both async context managers for the whole lifetime,
        # so they are entered and exited in the same task (the mcp SDK's anyio
        # cancel scopes break if entered/exited across tasks).
        self._stop = asyncio.Event()
        try:
            async with stdio_client(self._params, errlog=self._errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    # Drop the network tools here, at the source: the agent never
                    # sees them in its schema list, so it cannot select (or even
                    # attempt) a call that would leave the sandbox. Keep a record of
                    # what was dropped for the startup banner.
                    self._disabled = sorted(
                        t.name for t in listed.tools if t.name in NETWORK_TOOLS
                    )
                    self._schemas = [
                        _to_schema(t)
                        for t in listed.tools
                        if t.name not in NETWORK_TOOLS
                    ]
                    self._session = session
                    self._ready.set()      # unblock start()
                    await self._stop.wait()  # park until close()
        except Exception as e:
            self._startup_error = e
            self._ready.set()              # unblock start() with the error

    # ---- synchronous API -------------------------------------------------

    def list_tools(self) -> list[dict]:
        """Tool schemas in gerbil format ({name, description, input_schema}).
        Network tools are already filtered out (see NETWORK_TOOLS)."""
        return self._schemas

    @property
    def disabled_tools(self) -> list[str]:
        """Names of the network tools that were filtered out (never exposed to
        the agent), for reporting at startup."""
        return self._disabled

    def call_tool(self, name: str, args: dict) -> ToolResult:
        """Invoke an MCP tool. Never raises -- failures become error ToolResults."""
        if name in NETWORK_TOOLS:
            # Defense in depth: these are filtered out of the advertised schemas,
            # so the agent should never reach here -- but refuse outright if it
            # somehow does, rather than leaving the sandbox.
            return ToolResult(
                f"[{name} is disabled: it requires external network access, which "
                "is not allowed inside the sandbox]",
                is_error=True,
            )
        if self._session is None or self._loop is None:
            return ToolResult("mcp session is not available", is_error=True)
        timeout = TOOL_TIMEOUTS.get(name, DEFAULT_TIMEOUT)
        coro = self._session.call_tool(name, arguments=args)
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            result = fut.result(timeout=timeout)
        except FuturesTimeout:
            fut.cancel()
            return ToolResult(
                f"[mcp tool {name} timed out after {timeout:.0f}s]", is_error=True
            )
        except Exception as e:
            return ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        return _to_result(result)


def _to_schema(tool) -> dict:
    """Convert an MCP Tool to gerbil's {name, description, input_schema} shape."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema or {"type": "object", "properties": {}},
    }


def _to_result(result) -> ToolResult:
    """Convert an MCP CallToolResult to a gerbil ToolResult."""
    parts = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(text)
        else:
            parts.append(f"[{getattr(block, 'type', 'content')} block]")
    body = "\n".join(parts) if parts else "(no content)"
    return ToolResult(body, is_error=bool(getattr(result, "isError", False)))
