"""Test the lean_run_code guard that rejects a standalone `import Mathlib`.

Importing the whole Mathlib library in a snippet loads it from scratch and hangs
the LSP, so call_tool rejects it up front with an error the agent can react to.
A specific import (import Mathlib.Data.Nat.Basic) must still be allowed.

Pure/fast: the detector is tested directly, and the call_tool guard returns
before any MCP session is needed, so no Docker/server is required.

Run with: uv run python tests/test_run_code_guard.py
"""

from mcp import StdioServerParameters

from gerbil.mcp_client import McpClient, _has_bare_mathlib_import


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


def main() -> None:
    # --- the detector ---------------------------------------------------------
    blocked = [
        "import Mathlib",
        "import Mathlib\n#check Nat.repr_inj",       # the exact reported case
        "  import Mathlib  ",                         # leading/trailing space
        "import   Mathlib",                           # extra spaces
        "import Mathlib -- everything",               # trailing comment
        "import Lean\nimport Mathlib\n#check x",      # not the first line
    ]
    for code in blocked:
        check(f"blocks: {code!r}", _has_bare_mathlib_import(code), code)

    allowed = [
        "import Mathlib.Data.Nat.Basic",              # specific module is fine
        "import Mathlib.Tactic\nimport Mathlib.Order.Basic",
        "import Lean\n#check Nat.succ",
        "-- import Mathlib",                          # commented out
        "#check Mathlib",                             # not an import line
        "",
    ]
    for code in allowed:
        check(f"allows: {code!r}", not _has_bare_mathlib_import(code), code)

    # --- the call_tool guard (no session needed; it returns first) ------------
    dummy = StdioServerParameters(command="true", args=[])
    mcp = McpClient(sandbox=None, server_params=dummy)  # not started

    r = mcp.call_tool("lean_run_code", {"code": "import Mathlib\n#check Nat.repr_inj"})
    check("call_tool rejects import Mathlib",
          r.is_error and "not allowed" in r.content, r.content)

    # A specific import passes the guard (falls through to the not-started session).
    r = mcp.call_tool("lean_run_code", {"code": "import Mathlib.Data.Nat.Basic\n#check Nat"})
    check("call_tool allows a specific Mathlib import past the guard",
          "import Mathlib" not in r.content and "session is not available" in r.content,
          r.content)

    print("\nrun_code guard test passed.")


if __name__ == "__main__":
    main()
