"""Smoke test for the lean-lsp MCP integration.

Phase 1 (fast): validates the new infrastructure — the docker-exec stdio
transport, the async->sync bridge, the MCP handshake, and schema conversion.
list_tools() does not require the Lean toolchain, so this is quick and reliable.

Phase 2 (slow): the staleness check from the plan — write a file with an error
via gerbil's tool path, then ask the LSP for diagnostics and confirm it sees the
new content. This boots `lake serve` and installs the Lean toolchain on first
use, so it can take a few minutes.

Requires Docker and the lean-sandbox image (built from src/lean-sandbox). No API
key needed. Run:  uv run python tests/test_mcp.py
"""

import subprocess
import tempfile
from pathlib import Path

from gerbil.mcp_client import McpClient
from gerbil.sandbox import LeanSandbox

TOOLCHAIN = "leanprover/lean4:v4.15.0"


def make_project(root: Path) -> None:
    (root / "lean-toolchain").write_text(TOOLCHAIN + "\n")
    (root / "lakefile.toml").write_text(
        'name = "mcptest"\n'
        'defaultTargets = ["McpTest"]\n\n'
        "[[lean_lib]]\n"
        'name = "McpTest"\n'
    )
    (root / "McpTest.lean").write_text("def hello : Nat := 1\n")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"mcp test failed at: {label}\n{detail}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root)

        with LeanSandbox(project_dir=root, fetch_cache=False) as sb:
            with McpClient(sb) as mcp:
                # -- Phase 1: transport + handshake + schema conversion --------
                schemas = mcp.list_tools()
                names = {t["name"] for t in schemas}
                check("list_tools returned tools", len(schemas) > 0, str(len(schemas)))
                check("has lean_diagnostic_messages", "lean_diagnostic_messages" in names, str(sorted(names)))
                check("has lean_goal", "lean_goal" in names)
                shape_ok = all(
                    set(t) >= {"name", "description", "input_schema"} for t in schemas
                )
                check("schemas in gerbil format", shape_ok)
                print(f"  {len(schemas)} tools: {', '.join(sorted(names))}")

                # -- Phase 2: staleness check (slow; boots the LSP) -----------
                print("\n[phase 2] warming up the toolchain + build (slow)...")
                build = sb.run("lake build", timeout=600.0)
                check("lake build succeeded", build.exit_code == 0, build.stderr[-500:])

                # Overwrite the file with a type error via gerbil's tool path
                # (put_archive) -- out of band from the LSP.
                sb.write_file("McpTest.lean", 'def hello : Nat := "not a nat"\n')

                res = mcp.call_tool("lean_diagnostic_messages", {"file_path": "McpTest.lean"})
                print(f"  diagnostics -> {res.content[:300]}")
                # The LSP should report a type-mismatch error for the new content.
                saw_error = ("error" in res.content.lower()) or res.is_error
                check("LSP sees gerbil-written change (no staleness)", saw_error, res.content)

    print("\nMCP smoke test passed.")


if __name__ == "__main__":
    main()
