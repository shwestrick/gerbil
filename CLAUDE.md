# CLAUDE.md

Guidance for working in this repository.

## What gerbil is

gerbil is a teensy, self-contained autonomous coding agent for **Lean 4 / Lake
projects**. You give it a prompt describing a task (typically "prove this" or
"fill in these `sorry`s"), and it drives a loop between an LLM and a Docker
sandbox until the model stops requesting tools. It is inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover) but adds Docker-based
sandboxing, a git-based workflow, and built-in Ralph loops.

The defining design choices:

- **Sandboxed**: every session runs inside a Docker container (the
  `lean-sandbox` image). The host never executes agent-authored commands.
- **Git-native I/O**: gerbil uploads the *real* repo (with full `.git` history)
  into the container, lets the agent work, then squashes the session's work into
  a single commit and emits a `git format-patch` (`.patch`). Nothing touches the
  host repo until the user runs `gerbil commit` (`git am`). The agent's output is
  read *purely* as `git format-patch <base>..HEAD` — anything not reachable from
  that range is lost.
- **Provider-agnostic**: one unified streaming interface over Gemini, Anthropic,
  OpenAI, ollama (local models, `--model ollama:<NAME>`), and Portkey AI
  gateways (`--model portkey:<MODEL>` or a bare `@provider/model` catalog name;
  auth via `PORTKEY_API_KEY`, self-hosted gateway via `PORTKEY_BASE_URL`).
  Model is selected with `--model`; provider is auto-detected from the name.
  ollama runs on the host (not in the sandbox) and gerbil starts `ollama serve`
  itself if needed.
- **Resumable**: a crashed session can be continued from its append-only `.jsonl`
  log; a `.wip.patch` snapshot next to the log is refreshed every turn.

This is a **Python project that operates on Lean projects** — there is no Lean
code in this repo itself. The Lean toolchain lives only inside the sandbox image.

## Layout

```
bin/gerbil              self-contained bash launcher (installed on PATH; fetches
                        its own source from GitHub, builds the image, runs via uv)
src/gerbil/
  cli.py                argparse entry point + all subcommands (run, commit,
                        summarize, reconstruct-patch) and the resume orchestration
  agent.py              the agent loop (run_session), system prompts, pricing
                        table, and pretty terminal rendering of tool calls
  providers.py          unified LLM streaming over gemini/anthropic/openai/
                        ollama/portkey
  ollama.py             host-side ollama server detect/start/stop + model check
                        (local provider; reuses the OpenAI-compatible stream core)
  sandbox.py            LeanSandbox — Docker container lifecycle + all git plumbing
  tools.py              built-in tools (bash/read_file/write_file/edit_file) and
                        the Toolset that merges them with MCP tools
  mcp_client.py         sync façade over the lean-lsp-mcp server (runs in-container,
                        reached via `docker exec -i`)
  session.py            append-only JSONL session recorder
  resume.py             parse a (crashed) session log back into a conversation
  term.py               tiny ANSI color helper (respects NO_COLOR / non-TTY)
src/lean-sandbox/
  Dockerfile            debian + elan (no default toolchain) + lean-lsp-mcp venv
tests/                  standalone scripts (see Testing), not a pytest suite
pyproject.toml          packaging; entry point is gerbil.cli:main
```

## How a run works (the core flow)

1. **Preflight** (`cli.cmd_run`): require a git repo with ≥1 commit, a lakefile,
   a clean working tree, and a reachable Docker daemon.
2. **Sandbox boot** (`sandbox.LeanSandbox.__enter__`): start the container, upload
   all git-tracked files + the `.git` dir, configure a `gerbil` committer
   identity, and `lake exe cache get` (skip with `--skip-cache`).
3. **MCP start** (`cli._start_mcp`): launch lean-lsp-mcp inside the container; on
   failure, warn and continue with built-in tools only. The network-backed search
   tools (`mcp_client.NETWORK_TOOLS`) are filtered out of the advertised schemas
   and refused if invoked, so the agent can't (and won't try to) leave the sandbox.
4. **Agent loop** (`agent.run_session`): stream turns; execute tool calls via
   `Toolset.dispatch`; record everything to the `Session`; refresh the
   `.wip.patch` snapshot after each tool-running turn. When the model stops
   calling tools, one extra turn asks it to write a commit message.
5. **Finalize** (`cli._finalize_session`): `squash_commit(base)` collapses all of
   the session's work (the agent's own intermediate commits *plus* uncommitted
   changes) into one commit, then `format_patch(base)` writes the `.patch`.

Outputs: the live `.jsonl` log lands in `~/.gerbil/sessions/`; the `.patch` lands
in the project's `.gerbil/` (and is copied to the archive). The log only reaches
the project commit if `--include-session` is passed.

## Key invariants — read before changing git/sandbox logic

- **The base commit is the contract.** The agent's result is `git format-patch
  base..HEAD`. The system prompt (`agent.GIT_STATE_NOTE`) forbids the agent from
  running `git reset`/`checkout`/`stash`/`init`. gerbil's own git always goes
  through `sandbox._git`, which pins `GIT_DIR`/`GIT_WORK_TREE` so a stray nested
  repo the agent creates can't hijack gerbil's bookkeeping.
- **Built-in tool names win** over colliding MCP tool names (today none collide).
- **Tool output is truncated once** (`tools.truncate_tool_output`, 10k chars,
  head+tail) and the *same* truncated text is what the model sees and what the log
  records — keep that property.
- **Container uid/gid (1000) must match** `SANDBOX_UID`/`SANDBOX_GID` in
  sandbox.py and the `useradd` in the Dockerfile.
- The terminal rendering in agent.py (`_format_tool_call` and friends) is purely
  cosmetic — it must never change what is dispatched or recorded.

## Subcommands

- `gerbil run --prompt FILE [--model M] [--ralph N] [--ralph_done SCRIPT]
  [--max-turns N] [--skip-cache] [--no-mcp] [--include-session]`
- `gerbil resume LOG [--at DIR] [--max-turns N] [--skip-cache] [--no-mcp]
  [--ralph_done SCRIPT] [--include-session]` — continue a crashed/interrupted
  session (model and prompt come from the log).
- `gerbil commit` — `git am` the project's `.gerbil/*.patch` in order, skipping
  already-applied (by stable patch-id) and stale (non-applying) patches.
- `gerbil summarize` — token/cost/tool/status stats across `.gerbil/*.jsonl`.
- `gerbil reconstruct-patch LOG` — rebuild a `.patch` by *replaying the logged
  tool calls* (`bash`/`write_file`/`edit_file`; read-only/`lean_*` skipped) in a
  fresh sandbox, no LLM involved.

**Ralph loops** (`--ralph N`): run the same prompt across N back-to-back sessions
in one sandbox, each building on the previous one's commit. Each session records
its `chain_base` + ordered `ancestors` (prior patches) so any mid-chain session
is independently resumable. `--ralph_done SCRIPT` runs in-container after each
session; exit 0 stops the loop.

**Resume** (`gerbil resume LOG`): recreate the session's git starting point (for
a ralph chain, `git am` the recorded ancestor patches), reapply the `.wip.patch`,
replay the logged conversation, and continue. Model/prompt come from the log (so
`gerbil resume` takes neither `--prompt` nor `--model`, and no `--ralph` — a
resumed ralph chain continues on its own). The continuation is written as its own
fresh, resumable log/patch. `cmd_resume` in cli.py handles it.

## Development

The launcher (`bin/gerbil`) is what users install; it runs the versioned source
via `uv`. For local development run the module directly:

```bash
uv run python -m gerbil run --prompt prompt.md --at /path/to/lake/project
```

When invoked this way (no launcher), the sandbox image defaults to
`lean-sandbox:latest` and `GERBIL_VERSION` is `unknown`. Build the image with:

```bash
docker build -t lean-sandbox:latest src/lean-sandbox
```

Dependencies (managed by `uv`, see pyproject.toml): `docker`, `mcp`, and all
four provider SDKs (`anthropic`, `openai`, `google-genai`, `portkey-ai`) are
core deps so any `--model` works out of the box. `providers.py` imports only
the selected SDK at runtime. Requires Python ≥ 3.12.

## Testing

Tests are **standalone scripts**, not a pytest suite — run each directly:

```bash
uv run python tests/smoke_test.py        # Docker plumbing (needs Docker; stubs cache)
uv run python tests/test_mcp.py          # lean-lsp MCP integration (Docker; slow phase 2)
uv run python tests/test_reconstruct.py  # reconstruct-patch end-to-end (Docker)
uv run python tests/test_resume.py       # resume logic
uv run python tests/test_render.py       # terminal rendering
uv run python tests/test_ollama.py       # ollama provider plumbing (no Docker; live smoke if a server is up)
uv run python tests/test_portkey.py      # portkey provider plumbing (no Docker/key; live smoke if PORTKEY_API_KEY + PORTKEY_TEST_MODEL set)
GOOGLE_API_KEY=... uv run python tests/test_gemini.py   # live Gemini backend
```

Most require Docker and the `lean-sandbox` image; `test_gemini.py` needs a real
API key. `test_ollama.py` and `test_portkey.py` need neither Docker nor a key
(each runs a live smoke only if its backend is already reachable/configured).

## Conventions

- Match the existing style: heavy explanatory docstrings/comments on the *why*
  (especially around the git and sandbox plumbing — the subtleties are the point),
  small focused helpers, no external formatter config.
- Provider streaming yields a fixed event vocabulary (`TextDelta`, `ToolCall`,
  `_ToolMeta`, `Done`); keep new providers conforming to it.
- Session events are append-only JSONL; never rewrite a log in place.
