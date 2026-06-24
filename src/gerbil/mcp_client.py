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
import re
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import (
    CancelledNotification,
    CancelledNotificationParams,
    ClientNotification,
)

from .sandbox import WORKSPACE_DIR
from .tools import RESET_HINT, ToolResult

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

# A standalone `import Mathlib` (the whole library) on its own line. Loading all
# of Mathlib from scratch in a snippet is the heavy operation that hangs the LSP,
# so we reject it in lean_run_code. A specific import (e.g. `import
# Mathlib.Data.Nat.Basic`) is NOT matched -- the trailing `.Foo` fails `\s`.
_BARE_MATHLIB_IMPORT = re.compile(r"^\s*import\s+Mathlib\s*(--.*)?$")


def _has_bare_mathlib_import(code: str) -> bool:
    """Whether any line of `code` is a standalone `import Mathlib`."""
    return any(_BARE_MATHLIB_IMPORT.match(line) for line in code.splitlines())


class McpClient:
    """Synchronous façade over an asyncio MCP stdio client.

    Use as a context manager:

        with McpClient(sandbox) as mcp:
            schemas = mcp.list_tools()      # gerbil-format tool schemas
            result = mcp.call_tool(name, args)   # -> ToolResult
    """

    def __init__(
        self,
        sandbox,
        project_path: str = WORKSPACE_DIR,
        server_params: StdioServerParameters | None = None,
    ):
        # server_params lets tests point the client at a local mock MCP server
        # over stdio; production always connects to lean-lsp-mcp in the container.
        self._params = server_params or StdioServerParameters(
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
        # Kept so restart() can sweep orphaned Lean processes in the container.
        self._sandbox = sandbox
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
            self._stop_loop()  # leave errlog open so the client can be retried
            raise TimeoutError("lean-lsp-mcp did not initialize in time")
        if self._startup_error is not None:
            self._stop_loop()
            raise self._startup_error

    def _stop_loop(self) -> None:
        """Signal the background loop to exit and join its thread, leaving the
        client able to start() again. Does NOT close errlog (close() does that)."""
        if self._loop and self._stop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread:
            self._thread.join(timeout=15)
        self._thread = None
        self._loop = None
        self._session = None
        self._stop = None

    def close(self) -> None:
        self._stop_loop()
        try:
            self._errlog.close()
        except Exception:
            pass

    def restart(self) -> int:
        """Tear down and restart the lean-lsp-mcp server -- the recovery valve for
        a wedged language server (calls timing out, server unresponsive). Stops the
        current connection, sweeps orphaned Lean processes the dead server may have
        left behind (so a stuck worker stops pegging the container), then starts a
        fresh server. Returns the number of tools available afterward. Raises if the
        fresh server fails to come up."""
        self._stop_loop()
        self._sweep_orphans()
        # Fresh per-connection state for start().
        self._ready = threading.Event()
        self._startup_error = None
        self._schemas = []
        self._disabled = []
        self.start()
        return len(self._schemas)

    def _sweep_orphans(self) -> None:
        """Best-effort: kill Lean processes left running in the container. When the
        MCP server is killed, its child `lake serve`/REPL and the `lean` file
        workers it spawned can be orphaned (the container's PID 1 is `sleep`, which
        doesn't reap them) and keep pegging CPU. Match on /proc/<pid>/comm (the
        image has no pkill/ps) for exactly `lean`/`lake`/`repl`, so the sweep can't
        hit gerbil's git, the shell, or the container's init. Nothing else of
        gerbil's runs concurrently at restart time, so this is safe."""
        if self._sandbox is None:
            return
        script = (
            "for d in /proc/[0-9]*; do "
            "c=$(cat \"$d/comm\" 2>/dev/null) || continue; "
            "case \"$c\" in lean|lake|repl) kill -9 \"${d##*/}\" 2>/dev/null;; esac; "
            "done; true"
        )
        try:
            self._sandbox.run(script, timeout=15)
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
        """Invoke an MCP tool. Never raises -- failures become error ToolResults.

        On timeout we don't just abandon the call: the timeout is enforced inside
        the event loop (asyncio.wait_for cancels the in-flight request) and we send
        the server an MCP `notifications/cancelled` for that request, so a heavy
        job it is still running (e.g. a Lean worker loading Mathlib) is actually
        torn down instead of lingering and starving later tool calls."""
        if name in NETWORK_TOOLS:
            # Defense in depth: these are filtered out of the advertised schemas,
            # so the agent should never reach here -- but refuse outright if it
            # somehow does, rather than leaving the sandbox.
            return ToolResult(
                f"[{name} is disabled: it requires external network access, which "
                "is not allowed inside the sandbox]",
                is_error=True,
            )
        if name == "lean_run_code" and _has_bare_mathlib_import(
            str(args.get("code", ""))
        ):
            # `import Mathlib` loads the whole library from scratch in the snippet,
            # which routinely hangs the LSP. Reject it outright so the agent gets a
            # clear, immediate error instead of a 60s timeout (and a wedged server).
            return ToolResult(
                "[`import Mathlib` is not allowed in lean_run_code: importing the "
                "whole library loads it from scratch and hangs the sandbox. Import "
                "only the specific modules you need (e.g. "
                "`import Mathlib.Data.Nat.Basic`), or edit a project file -- which "
                "already has Mathlib built -- and check it with "
                "lean_diagnostic_messages.]",
                is_error=True,
            )
        if self._session is None or self._loop is None:
            return ToolResult("mcp session is not available." + RESET_HINT, is_error=True)
        timeout = TOOL_TIMEOUTS.get(name, DEFAULT_TIMEOUT)
        fut = asyncio.run_coroutine_threadsafe(
            self._call_with_cancel(name, args, timeout), self._loop
        )
        try:
            # The inner wait_for bounds the call at `timeout`; allow a little extra
            # for the cancellation round-trip before we abandon the future itself.
            result = fut.result(timeout=timeout + 10)
        except TimeoutError:
            # asyncio.TimeoutError (inner wait_for) and concurrent.futures'
            # TimeoutError are both builtins.TimeoutError on 3.11+. Either way the
            # call timed out; _call_with_cancel has already asked the server to
            # cancel. Drop the future defensively and report the timeout.
            fut.cancel()
            return ToolResult(
                f"[mcp tool {name} timed out after {timeout:.0f}s]" + RESET_HINT,
                is_error=True,
            )
        except Exception as e:
            return ToolResult(f"{type(e).__name__}: {e}" + RESET_HINT, is_error=True)
        return _to_result(result)

    async def _call_with_cancel(self, name: str, args: dict, timeout: float):
        """Run one tool call bounded by `timeout` (runs on the loop thread). On
        timeout, asyncio.wait_for cancels the local request; we then notify the
        server to cancel it too, so its work actually stops."""
        # The request id the SDK will assign to this call. gerbil issues one tool
        # call at a time, so the next send_request -- call_tool's -- uses this id.
        req_id = getattr(self._session, "_request_id", None)
        try:
            return await asyncio.wait_for(
                self._session.call_tool(name, arguments=args), timeout
            )
        except asyncio.TimeoutError:
            if req_id is not None:
                await self._cancel_request(
                    req_id, f"gerbil client timeout after {timeout:.0f}s"
                )
            raise

    async def _cancel_request(self, req_id, reason: str) -> None:
        """Send the server an MCP `notifications/cancelled` for `req_id` so it
        aborts the in-flight request and frees its work. Best effort -- a failure
        here must not mask the timeout the caller is already handling."""
        try:
            await self._session.send_notification(
                ClientNotification(
                    CancelledNotification(
                        method="notifications/cancelled",
                        params=CancelledNotificationParams(
                            requestId=req_id, reason=reason
                        ),
                    )
                )
            )
        except Exception:
            pass


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
