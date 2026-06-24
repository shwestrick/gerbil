"""Tools the gerbil agent can use inside the sandbox.

Each tool has an Anthropic tool-use schema (TOOLS) and is executed by dispatch().
All file paths are relative to the Lean project root. dispatch() never raises:
errors are returned as strings (with is_error=True) so the model can react and
retry rather than crashing the session.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .sandbox import CommandResult, LeanSandbox

if TYPE_CHECKING:
    from .mcp_client import McpClient


TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command in the Lean project directory. Use this for "
            "lake build, lake exe, ls, grep, and any other shell operations. "
            "Returns stdout, stderr, and a nonzero exit code if the command failed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": (
            "Read a file's contents. The path is relative to the project root. "
            "Returns the raw file text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write (creating or overwriting) a file with the given contents. "
            "The path is relative to the project root; parent directories are "
            "created as needed. Prefer edit_file for small changes to existing files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
                "content": {
                    "type": "string",
                    "description": "The full contents to write.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file with a new string. old_string must "
            "match the file contents exactly and appear exactly once; include "
            "enough surrounding context to make it unique. Use this for targeted "
            "edits to existing files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the project root.",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to replace (must be unique in the file).",
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to replace it with.",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
]


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


# Maximum size (in characters) of tool output fed back to the model. A single
# tool call can produce enormous output (e.g. a build that prints megabytes of
# logs); sending all of it can blow past the model's context window and crash
# the session. Output larger than this is truncated, keeping the head and tail
# (errors often land at the end) with a summary of what was omitted in between.
MAX_TOOL_OUTPUT_CHARS = 10000


def truncate_tool_output(content: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Cap oversized tool output, appending a summary of what was omitted.

    Returns content unchanged if it is within `limit`. Otherwise keeps the first
    half and last half of `limit` characters (each trimmed back to a line
    boundary so we don't cut mid-line) and drops the middle, since the most
    relevant lines -- especially build/test errors -- tend to be at the start or
    end. A note giving the full size is inserted where the omission happened.
    """
    if len(content) <= limit:
        return content
    total_chars = len(content)
    total_lines = content.count("\n") + 1

    half = limit // 2
    head = content[:half]
    tail = content[-half:]
    # Trim each piece to a line boundary so we don't cut mid-line.
    nl = head.rfind("\n")
    if nl > 0:
        head = head[:nl]
    nl = tail.find("\n")
    if nl >= 0:
        tail = tail[nl + 1:]

    return (
        f"{head}\n"
        f"...\n"
        f"(Output truncated. Total length of tool output: "
        f"{total_lines} lines, {total_chars} characters. "
        f"Showing the first and last {half} characters.)\n"
        f"...\n"
        f"{tail}"
    )


def dispatch(sandbox: LeanSandbox, name: str, args: dict) -> ToolResult:
    """Execute a tool call against the sandbox. Never raises."""
    try:
        if name == "bash":
            return _bash(sandbox, args["command"])
        if name == "read_file":
            return _read_file(sandbox, args["path"])
        if name == "write_file":
            return _write_file(sandbox, args["path"], args["content"])
        if name == "edit_file":
            return _edit_file(
                sandbox, args["path"], args["old_string"], args["new_string"]
            )
        return ToolResult(f"unknown tool: {name}", is_error=True)
    except Exception as e:
        return ToolResult(f"{type(e).__name__}: {e}", is_error=True)


# A gerbil-provided tool (not from the sandbox or the MCP server) that restarts
# the lean-lsp language server. Offered to the agent only when MCP is enabled, and
# handled directly by the Toolset (see _reset_lean_server).
RESET_LEAN_SERVER_TOOL = {
    "name": "reset_lean_server",
    "description": (
        "Restart the Lean language server that backs the lean_* tools. Use this if "
        "the lean_* tools start timing out or behave as if the server is stuck or "
        "hung. It tears the server down (clearing any stuck Lean processes) and "
        "starts a fresh one; the next lean_* call re-initializes it, which may be "
        "slow. This does not touch your files or your edits -- it only restarts the "
        "server."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


class Toolset:
    """Unified tool registry passed to the agent loop.

    Combines gerbil's sandbox-bound built-in tools with optional MCP-server tools
    (lean-lsp). Exposes a flat schema list for the provider and a single dispatch
    entry point that routes by name. dispatch() never raises.

    `ralph` records whether this is a --ralph session, so the agent loop can append
    the repeating-loop note to the system prompt. (Whether the loop terminates is
    decided by the --ralph_done check script, not by any tool the model can call.)
    """

    def __init__(
        self,
        sandbox: LeanSandbox,
        mcp: "McpClient | None" = None,
        ralph: bool = False,
    ):
        self._sandbox = sandbox
        self._mcp = mcp
        self.ralph = ralph
        self._mcp_schemas: list[dict] = []
        self._mcp_names: set[str] = set()
        if mcp is not None:
            builtin = {t["name"] for t in TOOLS}
            # Built-in names win over any colliding MCP tool (today: none collide).
            self._mcp_schemas = [
                t for t in mcp.list_tools() if t["name"] not in builtin
            ]
            self._mcp_names = {t["name"] for t in self._mcp_schemas}

    def schemas(self) -> list[dict]:
        """Built-in schemas, the reset tool (only when MCP is on), then MCP schemas."""
        reset = [RESET_LEAN_SERVER_TOOL] if self._mcp is not None else []
        return TOOLS + reset + self._mcp_schemas

    def mcp_tool_names(self) -> set[str]:
        return set(self._mcp_names)

    def dispatch(self, name: str, args: dict) -> ToolResult:
        """Route to the reset tool, a built-in, or an MCP handler. Never raises."""
        if name == "reset_lean_server":
            return self._reset_lean_server()
        if name in self._mcp_names:
            try:
                return self._mcp.call_tool(name, args)
            except Exception as e:
                return ToolResult(f"{type(e).__name__}: {e}", is_error=True)
        return dispatch(self._sandbox, name, args)

    def _reset_lean_server(self) -> ToolResult:
        """Restart the lean-lsp server (see RESET_LEAN_SERVER_TOOL). Never raises."""
        if self._mcp is None:
            return ToolResult(
                "the Lean language server is not enabled (running without MCP); "
                "there is nothing to restart",
                is_error=True,
            )
        try:
            n = self._mcp.restart()
            return ToolResult(
                f"restarted the Lean language server; {n} lean tools available "
                "again. The next lean_* call will re-initialize it (may be slow)."
            )
        except Exception as e:
            return ToolResult(
                f"failed to restart the Lean language server: {type(e).__name__}: {e}",
                is_error=True,
            )


def _bash(sandbox: LeanSandbox, command: str) -> ToolResult:
    result = sandbox.run(command)
    return ToolResult(_format_command(result), is_error=result.exit_code != 0)


def _read_file(sandbox: LeanSandbox, path: str) -> ToolResult:
    try:
        return ToolResult(sandbox.read_file(path))
    except Exception:
        return ToolResult(f"could not read file: {path}", is_error=True)


def _write_file(sandbox: LeanSandbox, path: str, content: str) -> ToolResult:
    sandbox.write_file(path, content)
    return ToolResult(f"wrote {len(content)} bytes to {path}")


def _edit_file(
    sandbox: LeanSandbox, path: str, old_string: str, new_string: str
) -> ToolResult:
    if old_string == new_string:
        return ToolResult("old_string and new_string are identical", is_error=True)
    try:
        content = sandbox.read_file(path)
    except Exception:
        return ToolResult(f"could not read file: {path}", is_error=True)

    count = content.count(old_string)
    if count == 0:
        return ToolResult("old_string not found in file", is_error=True)
    if count > 1:
        return ToolResult(
            f"old_string appears {count} times; add more context to make it unique",
            is_error=True,
        )

    sandbox.write_file(path, content.replace(old_string, new_string))
    return ToolResult(f"edited {path}")


def _format_command(r: CommandResult) -> str:
    parts = []
    if r.stdout:
        parts.append(r.stdout.rstrip("\n"))
    if r.stderr:
        label = "[stderr]\n" if r.stdout else ""
        parts.append(label + r.stderr.rstrip("\n"))
    body = "\n".join(parts) if parts else "(no output)"
    if r.timeout_occurred:
        body += "\n[command timed out]"
    elif r.exit_code != 0:
        body += f"\n[exit code: {r.exit_code}]"
    return body
