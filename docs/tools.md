# Tools

The agent works through two sets of tools: gerbil's built-ins, and the Lean
language-server tools provided over MCP.

## Built-in tools

| Tool | What it does |
| --- | --- |
| `bash` | run a shell command in the project directory (this is how the agent builds: `lake build`) |
| `read_file` | read a file, relative to the project root |
| `write_file` | write a file's full contents |
| `edit_file` | replace an exact, unique string in a file |

These are always available, including with `--no-mcp`.

Tool output is truncated once at 10k characters, keeping the head and the tail.
The model and the session log see the *same* truncated text, so the log is an
accurate record of what the model actually saw.

## Lean LSP tools (MCP)

By default gerbil enables [lean-lsp-mcp](https://github.com/oOo0oOo/lean-lsp-mcp),
which gives the agent real interaction with the Lean language server rather
than just compiler output:

- `lean_goal` -- the proof state at a position
- `lean_diagnostic_messages` -- errors and warnings for a file
- `lean_hover_info` -- type signature and docs for an identifier
- `lean_multi_attempt` -- try several tactics at a position without editing
- `lean_local_search` -- search declarations in the project
- ...and the rest of the lean-lsp toolset

The MCP server runs **inside the sandbox container**, where the Lean toolchain
lives; gerbil connects to it over `docker exec` (or `podman exec`).

gerbil also offers a `reset_lean_server` tool alongside them, so a stuck or
hung language server can be torn down and restarted mid-session without losing
any of the agent's work. It is advertised only when MCP is enabled.

`gerbil run --no-mcp` disables the MCP tools entirely and runs with the
built-ins only. If the MCP server fails to start, gerbil warns and continues
that way automatically.

### Disabled network tools

A few lean-lsp tools are intentionally unavailable, because they call out to
external services and would take the agent's work outside the sandbox:

`lean_leansearch`, `lean_loogle`, `lean_leanfinder`, `lean_state_search`, and
`lean_hammer_premise`.

They are filtered out of the schemas the model is shown -- so it does not try
-- and refused if invoked anyway.

## Name collisions

Where a built-in tool name and an MCP tool name collide, the built-in wins.
(Today none collide.)

## Big-small mode tools

With `--small-model`, the big model additionally gets `zoom_in` and the small
model gets `zoom_out`. See [Big-small mode](big-small.md).
