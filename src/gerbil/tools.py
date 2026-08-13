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


# Appended to gerbil-side MCP failures (timeouts, transport errors like
# ClosedResourceError, a lost session) to nudge the model toward recovery instead
# of retrying into the same wall. Deliberately NOT added to policy rejections
# (network tools, `import Mathlib`) or to normal tool-level errors, where
# restarting the server would not help.
RESET_HINT = (
    " [hint: if the Lean server keeps timing out or failing like this, call the "
    "reset_lean_server tool to restart it, then retry.]"
)


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


# A gerbil-provided tool available whenever the session has a termination
# check installed (--ralph_done, or the check --fill-sorry generates). It
# lets the model run the EXACT script the harness runs between sessions --
# closing the gap between "I believe the task is done" and "the harness
# agrees": without it, an agent that considers itself finished can only
# watch the loop restart with no idea which condition it is failing.
# Handled directly by the Toolset (see _check_goal).
CHECK_GOAL_TOOL = {
    "name": "check_goal",
    "description": (
        "Run this task's termination check: the exact script the harness runs "
        "between sessions to decide whether the task is complete (exit 0 ends "
        "the loop). Returns the check's verdict and output. The check is the "
        "definition of done -- if it says the goal is not met, the task is "
        "not done, whatever you believe. It typically runs a full build, so "
        "call it at natural checkpoints (e.g. when you think you are "
        "finished), not after every edit."
    ),
    "input_schema": {"type": "object", "properties": {}},
}


# The big-small mode tools. Both are gerbil-provided and neither is ever
# dispatched: agent.py intercepts them by name before Toolset.dispatch is
# reached (zoom_in launches the inner small-model loop; zoom_out ends it).
# zoom_in is advertised only to the big (outer) model, zoom_out only to the
# small (inner) model -- see Toolset.schemas(zoom=...).
ZOOM_IN_TOOL = {
    "name": "zoom_in",
    "description": (
        "Delegate the mechanical proof details of a single sorry to a smaller "
        "model. Give the exact position of the sorry and a task prompt with "
        "everything the smaller model needs (context, rules, guidelines -- it "
        "cannot ask you questions); a focused sub-session works on just that "
        "sorry, and this call returns a summary of what it accomplished. The "
        "sub-session shares your working tree, so its edits are visible to "
        "you afterward."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task prompt for the smaller model: precise "
                               "instructions, all necessary context, and the "
                               "rules it must follow.",
            },
            "file": {
                "type": "string",
                "description": "Path of the file containing the sorry, "
                               "relative to the project root.",
            },
            "line": {
                "type": "integer",
                "description": "1-indexed line of the sorry.",
            },
            "column": {
                "type": "integer",
                "description": "1-indexed column of the sorry (optional).",
            },
        },
        "required": ["prompt", "file", "line"],
    },
}

ZOOM_OUT_TOOL = {
    "name": "zoom_out",
    "description": (
        "End this zoomed-in sub-session and report back to the outer model. "
        "Call this exactly once, when the sorry is resolved or you are stuck: "
        "the summary is all the outer model will see of your work besides the "
        "file changes themselves."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "What you did, what state the sorry is in now, "
                               "and anything the outer model must know.",
            },
        },
        "required": ["summary"],
    },
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
        check_script: str | None = None,
        check_timeout: float = 300.0,
    ):
        self._sandbox = sandbox
        self._mcp = mcp
        self.ralph = ralph
        # The termination check's script text and timeout, when the session
        # has one (--ralph_done or --fill-sorry). Advertises and backs the
        # check_goal tool; may also be assigned after construction (cli.py
        # generates the --fill-sorry check only once the sandbox is up).
        self.check_script = check_script
        self.check_timeout = check_timeout
        self._mcp_schemas: list[dict] = []
        self._mcp_names: set[str] = set()
        if mcp is not None:
            builtin = {t["name"] for t in TOOLS}
            # Built-in names win over any colliding MCP tool (today: none collide).
            self._mcp_schemas = [
                t for t in mcp.list_tools() if t["name"] not in builtin
            ]
            self._mcp_names = {t["name"] for t in self._mcp_schemas}

    def schemas(self, zoom: str | None = None) -> list[dict]:
        """Built-in schemas, the reset tool (only when MCP is on), then MCP schemas.

        `zoom` selects the big-small mode variant: "outer" additionally
        advertises zoom_in (the big model), "inner" advertises zoom_out (the
        small model's sub-session). None -- the default, and the only value used
        outside big-small mode -- advertises neither."""
        reset = [RESET_LEAN_SERVER_TOOL] if self._mcp is not None else []
        check = [CHECK_GOAL_TOOL] if self.check_script else []
        base = TOOLS + check + reset + self._mcp_schemas
        if zoom == "outer":
            return base + [ZOOM_IN_TOOL]
        if zoom == "inner":
            return base + [ZOOM_OUT_TOOL]
        return base

    def mcp_tool_names(self) -> set[str]:
        return set(self._mcp_names)

    def dispatch(self, name: str, args: dict) -> ToolResult:
        """Route to the reset tool, a built-in, or an MCP handler. Never raises."""
        if name == "reset_lean_server":
            return self._reset_lean_server()
        if name == "check_goal":
            return self._check_goal()
        if name in self._mcp_names:
            try:
                return self._mcp.call_tool(name, args)
            except Exception as e:
                # call_tool already turns failures into ToolResults; this is a
                # belt-and-suspenders path for anything that still escapes.
                return ToolResult(
                    f"{type(e).__name__}: {e}" + RESET_HINT, is_error=True
                )
        return dispatch(self._sandbox, name, args)

    def _check_goal(self) -> ToolResult:
        """Run the session's termination check for the model (see
        CHECK_GOAL_TOOL). The verdict line comes first so it survives even a
        truncated transcript; is_error stays False on a failing CHECK -- the
        tool ran fine, and the failure detail is the content the model asked
        for. Never raises."""
        if not self.check_script:
            return ToolResult(
                "no termination check is installed for this session",
                is_error=True,
            )
        try:
            result = self._sandbox.run_script(
                self.check_script, timeout=self.check_timeout
            )
        except Exception as e:
            return ToolResult(
                f"failed to run the termination check: {type(e).__name__}: {e}",
                is_error=True,
            )
        out = (result.stdout + result.stderr).strip()
        verdict = (
            "CHECK PASSED (exit 0): the goal is met; the session loop will stop."
            if result.exit_code == 0
            else f"CHECK NOT PASSED (exit {result.exit_code}): the goal is "
            "not met yet. The output below says which condition failed."
        )
        return ToolResult(f"{verdict}\n\n{truncate_tool_output(out)}".rstrip())

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
