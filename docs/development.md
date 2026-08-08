# Development

This is a **Python project that operates on Lean projects** -- there is no Lean
code in this repository. The Lean toolchain lives only inside the sandbox
image.

## Layout

```
bin/gerbil              self-contained bash launcher (installed on PATH; fetches
                        its own source from GitHub, builds the image, runs via uv)
src/gerbil/
  cli.py                argparse entry point + all subcommands and resume orchestration
  agent.py              the agent loop (run_session) and transient-error retry
  prompts.py            system/task prompt text
  pricing.py            per-model pricing table and cost estimation
  render.py             all human-readable terminal rendering (leaf module)
  providers.py          unified LLM streaming over gemini/anthropic/openai/ollama/portkey
  ollama.py             host-side ollama server detect/start/stop + model check
  runtime.py            container-runtime selection + the podman client
  sandbox.py            LeanSandbox -- container lifecycle, image check, git plumbing
  tools.py              built-in tools and the Toolset that merges them with MCP tools
  mcp_client.py         sync facade over the lean-lsp-mcp server (runs in-container)
  session.py            append-only JSONL session recorder
  resume.py             parse a (crashed) session log back into a conversation
src/lean-sandbox/
  Dockerfile            debian + elan (no default toolchain) + lean-lsp-mcp venv
tests/                  standalone scripts (see below), not a pytest suite
pyproject.toml          packaging; entry point is gerbil.cli:main
```

## Running from source

The launcher (`bin/gerbil`) is what users install; it runs the versioned source
via `uv`. For local development, run the module directly:

```bash
uv run python -m gerbil run --prompt prompt.md --at /path/to/lake/project
```

Invoked this way (no launcher), the sandbox image defaults to
`gerbil-lean-sandbox:latest` and the recorded `GERBIL_VERSION` is `unknown`.

Build the image with:

```bash
docker build -t gerbil-lean-sandbox:latest src/lean-sandbox
# or, with GERBIL_SANDBOX=podman:
podman build -t gerbil-lean-sandbox:latest src/lean-sandbox
# ...and on a host where rootless podman has no /etc/subuid range (uid 1000 can
# be neither chowned to at build time nor mapped at run time):
podman build --build-arg SANDBOX_UID=0 -t gerbil-lean-sandbox:latest src/lean-sandbox
```

Dependencies are managed by `uv` (see `pyproject.toml`). `docker`, `mcp`, and
all four provider SDKs (`anthropic`, `openai`, `google-genai`, `portkey-ai`)
are core deps so any `--model` works out of the box; `providers.py` imports
only the selected SDK at runtime. Requires Python >= 3.12.

## Testing

Tests are **standalone scripts**, not a pytest suite -- run each directly:

```bash
uv run python tests/smoke_test.py           # container plumbing (needs Docker; stubs cache)
uv run python tests/test_runtime.py         # runtime selection + podman client (no Docker)
uv run python tests/test_mcp.py             # lean-lsp MCP integration (Docker; slow phase 2)
uv run python tests/test_reconstruct.py     # reconstruct-patch end-to-end (Docker)
uv run python tests/test_commit.py          # gerbil commit end-to-end (Docker)
uv run python tests/test_resume.py          # resume logic
uv run python tests/test_zoom.py            # big-small inner loop + zoom schemas (no Docker)
uv run python tests/test_zoom_resume.py     # big-small resume + summarize accounting (no Docker)
uv run python tests/test_submodule.py       # submodule upload + containment (phase 2 needs Docker)
uv run python tests/test_image_config.py    # image selection + compatibility check (phase 2 needs Docker)
uv run python tests/test_mathlib_detect.py  # mathlib dependency detection (no Docker)
uv run python tests/test_render.py          # terminal rendering
uv run python tests/test_empty_turn.py      # glitched empty-turn guard + filter (no network)
uv run python tests/test_sandbox_cleanup.py # container cleanup on interrupt (no Docker)
uv run python tests/test_ollama.py          # ollama provider plumbing (live smoke if a server is up)
uv run python tests/test_portkey.py         # portkey provider plumbing (live smoke if configured)
GOOGLE_API_KEY=... uv run python tests/test_gemini.py   # live Gemini backend
```

Most require Docker and the `gerbil-lean-sandbox` image; `test_gemini.py` needs
a real API key. `test_ollama.py` and `test_portkey.py` need neither (each runs
a live smoke only if its backend is already reachable).

Container-backed tests run against whichever runtime `GERBIL_SANDBOX` selects,
so

```bash
GERBIL_SANDBOX=podman uv run python tests/smoke_test.py
```

exercises the whole sandbox/git plumbing through podman.

## Invariants worth knowing before changing things

`CLAUDE.md` in the repository root carries the full list, but the load-bearing
ones are:

- **The base commit is the contract.** The result of a session is
  `git format-patch base..HEAD`; anything unreachable from that range is lost.
  gerbil's own git always pins `GIT_DIR`/`GIT_WORK_TREE`, so a stray nested repo
  the agent creates can't hijack the bookkeeping.
- **Only the current branch enters the sandbox.** Never reintroduce a raw copy
  of the host `.git`.
- **Submodule state is reset before the patch is taken.** A patch cannot carry
  submodule work; see [Sandboxing](sandbox.md#submodules).
- **Tool output is truncated once**, and the same truncated text is what the
  model sees and what the log records.
- **Engine differences live only in `runtime.py`.** `sandbox.py` talks to one
  Docker-SDK-shaped interface.
- **`render.py` is purely cosmetic** and imports nothing from gerbil. It must
  never change what is dispatched or recorded.

## Conventions

- Heavy explanatory comments on the *why*, especially around git and sandbox
  plumbing -- the subtleties are the point.
- Provider streaming yields a fixed event vocabulary (`TextDelta`, `ToolCall`,
  `_ToolMeta`, `Done`); new providers must conform.
- Session events are append-only JSONL; never rewrite a log in place.
