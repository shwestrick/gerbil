"""Tools the gerbil agent can use inside the sandbox.

Each tool has an Anthropic tool-use schema (TOOLS) and is executed by dispatch().
All file paths are relative to the Lean project root. dispatch() never raises:
errors are returned as strings (with is_error=True) so the model can react and
retry rather than crashing the session.
"""

from dataclasses import dataclass

from .sandbox import CommandResult, LeanSandbox


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
