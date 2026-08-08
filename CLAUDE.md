# CLAUDE.md

Guidance for working in this repository.

## What gerbil is

gerbil is a teensy, self-contained autonomous coding agent for **Lean 4 / Lake
projects**. You give it a prompt describing a task (typically "prove this" or
"fill in these `sorry`s"), and it drives a loop between an LLM and a
containerized sandbox until the model stops requesting tools. It is inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover) but adds container-based
sandboxing, a git-based workflow, and built-in Ralph loops.

The defining design choices:

- **Sandboxed**: every session runs inside a container (the
  `gerbil-lean-sandbox` image). The host never executes agent-authored commands.
  The container engine is Docker by default, or podman when
  `GERBIL_SANDBOX=podman` (see `runtime.py`).
- **Git-native I/O**: gerbil uploads the *real* repo into the container — the
  tracked files plus a *sanitized* `.git` holding only the current branch's
  history (no other branches, tags, remotes, or reflogs: the agent sees nothing
  beyond the branch) — lets the agent work, then squashes the session's work into
  a single commit and emits a `git format-patch` (`.patch`). Nothing touches the
  host repo until the user runs `gerbil commit` (`git am`). The agent's output is
  read *purely* as `git format-patch <base>..HEAD` — anything not reachable from
  that range is lost. Submodules are uploaded fully populated so the project
  builds, but are read-only to the agent: a patch cannot carry submodule work, so
  gerbil strips any submodule change before emitting one (see the invariants).
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
  agent.py              the agent loop (run_session) and transient-error retry
  prompts.py            system/task prompt text + build_system_prompt, plus the
                        context-pressure thresholds and their notes
  pricing.py            MODEL_PRICING table and cost estimation (N/A when unknown)
  context_windows.py    everything about a model's context window: the static
                        CONTEXT_WINDOWS table (snapshot of benchlm.ai/llm-pricing)
                        plus ~/.gerbil/context-windows.jsonl, the append-only log
                        of what providers have actually reported. Resolution is
                        live query > observation > table; table_drift compares
                        the last two
  model_match.py        table_match — the one rule both tables are keyed by
                        (longest substring match, but only when it subsumes the
                        others; ambiguity -> None). A leaf module.
  render.py             ALL human-readable terminal rendering: the ANSI style()
                        helper (respects NO_COLOR / non-TTY) and the pretty
                        tool-call/result, context, and usage formatting
  providers.py          unified LLM streaming over gemini/anthropic/openai/
                        ollama/portkey
  ollama.py             host-side ollama server detect/start/stop + model check
                        (local provider; reuses the OpenAI-compatible stream core)
  runtime.py            container-runtime selection (GERBIL_SANDBOX=docker|podman)
                        + the podman client (a Docker-SDK-shaped façade over the
                        podman CLI, so sandbox.py never branches on the engine)
  sandbox.py            LeanSandbox — container lifecycle, the sandbox-image
                        compatibility check, and all git plumbing
  tools.py              built-in tools (bash/read_file/write_file/edit_file) and
                        the Toolset that merges them with MCP tools
  mcp_client.py         sync façade over the lean-lsp-mcp server (runs in-container,
                        reached via `docker`/`podman exec -i`)
  session.py            append-only JSONL session recorder
  resume.py             parse a (crashed) session log back into a conversation
src/lean-sandbox/
  Dockerfile            debian + elan (no default toolchain) + lean-lsp-mcp venv
tests/                  standalone scripts (see Testing), not a pytest suite
pyproject.toml          packaging; entry point is gerbil.cli:main
```

## How a run works (the core flow)

1. **Preflight** (`cli.cmd_run`): require a git repo with ≥1 commit, a lakefile,
   a clean working tree, submodules (if any) initialized and clean
   (`cli._require_clean_submodules`), and a usable container runtime
   (`runtime.check_available` — a reachable Docker daemon, or a working `podman`).
2. **Sandbox boot** (`sandbox.LeanSandbox.__enter__`): start the container, upload
   all git-tracked files + a sanitized single-branch `.git`
   (`sandbox._sanitized_git_dir`), upload each submodule fully populated
   (`sandbox._upload_submodule`), configure a `gerbil` committer identity, and
   `lake exe cache get` — only when the project actually depends on mathlib
   (`sandbox.uses_mathlib`); skip it outright with `--skip-cache`.
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
in the project's `.gerbil/` (and is copied to the archive). The log is folded
into the project commit by default; `--omit-session-log` keeps it out (the
archive copy in `~/.gerbil/sessions/` is kept regardless).

## Key invariants — read before changing git/sandbox logic

- **The base commit is the contract.** The agent's result is `git format-patch
  base..HEAD`. The system prompt (`prompts.GIT_STATE_NOTE`) forbids the agent from
  running `git reset`/`checkout`/`stash`/`init`. gerbil's own git always goes
  through `sandbox._git`, which pins `GIT_DIR`/`GIT_WORK_TREE` so a stray nested
  repo the agent creates can't hijack gerbil's bookkeeping.
- **Only the current branch enters the sandbox.** The uploaded `.git` is built
  by `sandbox._sanitized_git_dir` — a fetch of HEAD into a fresh repo, so no
  other branches, tags, remotes, reflogs, or unreachable objects reach the
  agent. Never reintroduce a raw copy of the host `.git`.
- **Submodules go in whole and never come out.** gerbil supports repos that *use*
  submodules; the agent does no submodule manipulation. Contents are uploaded
  from the host's already-initialized working tree (cloning in the container
  would need the `.gitmodules` URLs' credentials and availability, and gerbil
  never writes to the host repo — preflight refuses an uninitialized, moved, or
  dirty submodule instead), each with a `.git` of its own built by the
  same `_sanitized_git_dir`. That `.git` is deliberately a real *directory* at
  `<sub>/.git` — the pre-1.7.8 layout — not the `.git/modules` + gitdir-file
  arrangement: nested submodules then need no module-path juggling. On the way
  out, `sandbox._reset_submodule_state` restores every gitlink (and
  `.gitmodules`) to its base state in the index before `squash_commit` and
  `wip_patch` snapshot the tree, so no submodule change can reach a patch. This
  is enforced, not merely asked for in `prompts.SUBMODULE_NOTE`, because a patch
  *cannot* carry submodule work: format-patch renders a gitlink as one
  `Subproject commit <sha>` line, the objects behind it die with the container,
  and `git am` would silently commit a pointer to a commit that exists nowhere.
- **Context pressure escalates, and the terminal threshold must leave room to
  land.** When the window size is known, `agent.run_session` advises the model to
  wrap up at `CONTEXT_WIND_DOWN` (75%), orders it at `CONTEXT_URGENT` (85%), and
  at `CONTEXT_TERMINAL` (95%) stops the loop itself — but still runs the
  commit-message turn, shortening the diff to fit (`_commit_diff`) so the
  session's work lands explained rather than as a bare patch. Each note is
  delivered once, on the turn that crosses its threshold; repeating it would
  spend the very room it warns about. Every check is measured on
  `Usage.context_tokens`, the one definition of "how full is the context" —
  render.py's percentage and agent.py's thresholds must never disagree.
  The guards are inert when the window is unknown, so a provider that reports no
  context window behaves exactly as it did before. The zoom sub-session loop is
  *not* covered (it is bounded by `--zoom-max-turns` and its forced `zoom_out`).
- **A note can only reach the model inside the pending tool-result message.**
  The providers reject two user messages in a row and require an assistant turn's
  tool calls to be answered immediately, so `_append_user_text` merges text into
  that message rather than following it — which is also how the forced commit
  request gets in. This put a `{"type": "text"}` item into a user list for the
  first time: the Anthropic converter already passed it through, but the OpenAI
  one silently *dropped* non-tool_result items and Gemini stringified them into a
  printed dict. Both were fixed; a converter that loses this item disables the
  guard silently, so `tests/test_context_pressure.py` asserts each one transmits it.
- **A guessed number is worse than an honest "unknown."** Both model tables
  (`MODEL_PRICING`, `CONTEXT_WINDOWS`) resolve a `--model` string through
  `model_match.table_match`, which returns None when several keys match and none
  subsumes the rest; callers then report N/A rather than invent a figure. Keep
  new entries API-ID-shaped, and never hand-enter a value the upstream snapshot
  doesn't list — it would silently survive every refresh.
- **The context-window observation log records changes, not runs.**
  `record_observation` appends to `~/.gerbil/context-windows.jsonl` only when the
  value differs from the last one logged for that model, so a model whose window
  never moves has exactly one line ever and the log stays readable as a history.
  It is append-only (last line wins) like a session log, and every read is
  failure-tolerant — a corrupt line is skipped and an unwritable home costs the
  observation, never the session.
- **Built-in tool names win** over colliding MCP tool names (today none collide).
- **Tool output is truncated once** (`tools.truncate_tool_output`, 10k chars,
  head+tail) and the *same* truncated text is what the model sees and what the log
  records — keep that property.
- **Container uid/gid (1000) must match** `SANDBOX_UID`/`SANDBOX_GID` in
  sandbox.py and the `useradd` in the Dockerfile. Under podman it is also the
  uid uploads end up owned by (they are unpacked by the container's own `tar`,
  running as the image's user). The one exception is the image built with
  `--build-arg SANDBOX_UID=0`, which runs as container-root: the launcher picks
  it (tag suffix `-rootuser`) when rootless podman has no subuid range to map
  uid 1000 with, and there the uploads are simply root-owned inside the
  container. Nothing on the host is affected either way — gerbil never bind
  mounts; files enter as a tar and leave as `git format-patch` text.
- **Engine differences live only in runtime.py.** sandbox.py talks to one
  Docker-SDK-shaped interface; `runtime.PodmanClient` reimplements exactly the
  methods it uses over `podman` subprocesses. Two podman quirks are already
  handled there and are easy to regress: podman is muzzled with
  `--log-level=error` (its warnings otherwise land in the stderr of every tool
  result, since `podman exec` cannot separate them), and uploads never go
  through `podman cp -` (its stdin copier fails at particular payload sizes).
  The image must also `mkdir` the workspace explicitly — buildah does not
  materialize a trailing `WORKDIR`, so `podman run --workdir` would fail.
- The terminal rendering in render.py (`format_tool_call` and friends) is purely
  cosmetic — it must never change what is dispatched or recorded. Keep render.py
  free of gerbil imports (it is a leaf module; agent.py calls in with real data).
  `turn_header` draws its rule with box-drawing characters only where the output
  stream can *encode* them (`_supports_unicode` asks the stream, never the
  locale; `GERBIL_ASCII=1` forces the fallback), and the two layouts are the same
  width column for column so nothing shifts between them. Anything else drawn
  with non-ASCII must degrade the same way — a `print` that raises
  UnicodeEncodeError would take a session down for a decoration.

## Subcommands

- `gerbil run --prompt FILE [--model M] [--small-model M] [--zoom-max-turns N]
  [--ralph N] [--ralph_done SCRIPT] [--max-turns N] [--image IMAGE]
  [--skip-cache] [--no-mcp] [--omit-session-log]`
- `gerbil resume LOG [--at DIR] [--max-turns N] [--zoom-max-turns N] [--image
  IMAGE] [--skip-cache] [--no-mcp] [--ralph_done SCRIPT] [--omit-session-log]` —
  continue a crashed/interrupted session (model and prompt come from the log).
- `gerbil commit` — `git am` the project's `.gerbil/*.patch` in order, skipping
  already-applied (by stable patch-id) and stale (non-applying) patches.
- `gerbil cleanup` — delete the project's `.gerbil/*.patch` files whose changes
  are already committed (same stable patch-id test as `gerbil commit`); patches
  not yet committed are never touched, and the `~/.gerbil/sessions/` archive
  copies are kept.
- `gerbil summarize` — token/cost/tool/status stats across `.gerbil/*.jsonl`,
  plus a per-session table (cost, `.lean` lines +/- from the commit each log
  was folded into or its uncommitted `.patch`, running lean-code total
  excluding `.lake`).
- `gerbil reconstruct-patch LOG [--image IMAGE]` — rebuild a `.patch` by *replaying the logged
  tool calls* (`bash`/`write_file`/`edit_file`; read-only/`lean_*` skipped) in a
  fresh sandbox, no LLM involved.

**Big-small mode** (`--small-model M`): the big model (`--model`) drives the
session and gets a `zoom_in(prompt, file, line[, column])` tool; calling it sets
the big conversation aside and runs an inner sub-session of the small model
(full toolset plus `zoom_out(summary)`, no `zoom_in` — depth is at most 1) on
that one sorry until it calls `zoom_out`, whose summary comes back as the
`zoom_in` tool result. The sub-session's initial prompt is the big model's
supplied `prompt` plus one appended "YOUR TASK" line naming the sorry and turn
budget (`prompts.zoom_task_prompt`); its system prompt is the plain gerbil one.
Each sub-session is capped at `--zoom-max-turns` (default 100; on the cap one
final forced turn — zoom_out as the only tool — demands the summary, so the
big model always gets a real report). Inner events are recorded
in the same `.jsonl` tagged `"zoom": true` — that tag is what lets resume
rebuild the two conversations separately (a mid-zoom crash resumes *inside* the
sub-session via `ParsedSession.pending_zoom`) and lets summarize price each
model's bucket at its own rates.

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
fresh, resumable log/patch; that log opens with the parent log re-emitted **in
full** (every event, tagged `"replayed": true`) followed by a `resumed` boundary
marker, so the one log folded into the eventual commit records the whole session
from the beginning — the crashed parent never commits, so this replay is the only
copy of its history that reaches the project's `.gerbil/`. `cmd_resume` in cli.py
handles it.

**Sandbox image** (`--image IMAGE`): a project may need pre-built artifacts, extra
packages, or a pinned toolchain that gerbil's stock image does not carry, so the
image is selectable. `cli._resolve_image` picks it, highest precedence first:
`--image`, then `image = "..."` in `<project>/.gerbil/config.toml` (stdlib
`tomllib`; unknown keys ignored so the file can grow), then
`GERBIL_SANDBOX_IMAGE` (what the launcher sets to its version-matched build),
then `gerbil-lean-sandbox:latest` for direct dev use. The project config
deliberately outranks the environment — the launcher *always* exports
`GERBIL_SANDBOX_IMAGE`, so a project could otherwise never state its own image.
The resolved image is recorded on `session_start` for provenance, but `gerbil
resume` re-resolves rather than reusing it: gerbil's own tag is version-pinned
and the launcher's `remove_old_images` prunes superseded ones, so a recorded
default would go stale the moment the user updates.

`sandbox._check_image` vets the image at boot, between `_wait_running` and the
upload, so a bad one fails before any work. Every problem is collected and
reported at once (a hand-rolled image is usually wrong in several ways), from a
single POSIX-`sh` probe — deliberately *not* `LeanSandbox.run`, which wraps
everything in `timeout ... bash -c`, two of the very things under test. The
contract an image must satisfy:

| Requirement | Why |
| --- | --- |
| `sh`, `bash`, `timeout` | every `LeanSandbox.run` is `timeout N bash -c CMD` |
| `git`, `tar`, `chown`, `id`, `mktemp` | git plumbing; podman's `put_archive` unpacks with the container's own `tar`; the post-upload `chown -R`; `--ralph_done` uses `mktemp` |
| `lake` on `PATH` | `_fetch_mathlib_cache` and every build |
| writable `/workspace/project` | `WORKSPACE_DIR`; podman will not create a missing `--workdir` |
| runs as uid 1000 or 0 | uploads are tarred as `SANDBOX_UID` and `chown -R`ed to it |
| root `exec` works | the post-upload `chown` runs with `user="root"` |
| no `ENTRYPOINT` swallowing the command | the container runs `sleep infinity` and must stay up |
| `lean-lsp-mcp` on `PATH` | **optional** — warns, then runs built-in tools only |

`sandbox._image_problems` is a pure function over the probe output, so the whole
matrix is testable without building deliberately-broken images.

**Mathlib cache** (`sandbox.uses_mathlib`): `cache` is an executable *mathlib*
provides, so `lake exe cache get` does not merely waste time in a project without
mathlib — it fails, and used to take the whole session down unless the user knew
to pass `--skip-cache`. The fetch is therefore gated on detection, which reads
`lake-manifest.json` (Lake's *resolved* graph, so mathlib arriving transitively
counts) plus the `require`s in `lakefile.toml`/`lakefile.lean`, and takes the
union — a manifest is generated by `lake update` and can lag a lakefile that just
gained mathlib. When nothing can be read the answer is True, i.e. gerbil's
original unconditional fetch: the skip is only taken on positive evidence that
mathlib is absent.

## Development

The launcher (`bin/gerbil`) is what users install; it runs the versioned source
via `uv`. For local development run the module directly:

```bash
uv run python -m gerbil run --prompt prompt.md --at /path/to/lake/project
```

When invoked this way (no launcher), the sandbox image defaults to
`gerbil-lean-sandbox:latest` and `GERBIL_VERSION` is `unknown`. Build the image with:

```bash
docker build -t gerbil-lean-sandbox:latest src/lean-sandbox
# or, with GERBIL_SANDBOX=podman:
podman build -t gerbil-lean-sandbox:latest src/lean-sandbox
# ...and on a host where rootless podman has no /etc/subuid range (uid 1000
# can be neither chowned to at build time nor mapped at run time), the image
# has to run as container-root instead:
podman build --build-arg SANDBOX_UID=0 -t gerbil-lean-sandbox:latest src/lean-sandbox
```

Dependencies (managed by `uv`, see pyproject.toml): `docker`, `mcp`, and all
four provider SDKs (`anthropic`, `openai`, `google-genai`, `portkey-ai`) are
core deps so any `--model` works out of the box. `providers.py` imports only
the selected SDK at runtime. Requires Python ≥ 3.12.

## Testing

Tests are **standalone scripts**, not a pytest suite — run each directly:

```bash
uv run python tests/smoke_test.py        # container plumbing (needs Docker; stubs cache)
uv run python tests/test_runtime.py      # runtime selection + podman client (no Docker;
                                         # live phase only if podman is installed)
uv run python tests/test_mcp.py          # lean-lsp MCP integration (Docker; slow phase 2)
uv run python tests/test_reconstruct.py  # reconstruct-patch end-to-end (Docker)
uv run python tests/test_commit.py       # gerbil commit end-to-end (Docker)
uv run python tests/test_resume.py       # resume logic
uv run python tests/test_zoom.py         # big-small inner loop + zoom schemas (no Docker)
uv run python tests/test_zoom_resume.py  # big-small resume + summarize accounting (no Docker)
uv run python tests/test_submodule.py    # submodule upload + containment (phase 2 needs Docker)
uv run python tests/test_image_config.py # image selection + compatibility check (phase 2 needs Docker)
uv run python tests/test_mathlib_detect.py  # mathlib dependency detection (no Docker)
uv run python tests/test_render.py       # terminal rendering
uv run python tests/test_empty_turn.py   # glitched empty-turn guard + filter (no network)
uv run python tests/test_sandbox_cleanup.py  # container cleanup on interrupt (no Docker)
uv run python tests/test_ollama.py       # ollama provider plumbing (no Docker; live smoke if a server is up)
uv run python tests/test_portkey.py      # portkey provider plumbing (no Docker/key; live smoke if PORTKEY_API_KEY + PORTKEY_TEST_MODEL set)
uv run python tests/test_stream_timeout.py  # stalled-stream idle timeout reaches every metered client (no network)
uv run python tests/test_context_windows.py # context-window fallback table + model matching (no network)
uv run python tests/test_context_pressure.py # context-exhaustion guards: escalation, message shape, forced ending (no network)
GOOGLE_API_KEY=... uv run python tests/test_gemini.py   # live Gemini backend
```

Most require Docker and the `gerbil-lean-sandbox` image; `test_gemini.py` needs a real
API key. `test_ollama.py` and `test_portkey.py` need neither Docker nor a key
(each runs a live smoke only if its backend is already reachable/configured).

The container-backed tests run against whichever runtime `GERBIL_SANDBOX`
selects, so `GERBIL_SANDBOX=podman uv run python tests/smoke_test.py` exercises
the whole sandbox/git plumbing through podman.

## Conventions

- Match the existing style: heavy explanatory docstrings/comments on the *why*
  (especially around the git and sandbox plumbing — the subtleties are the point),
  small focused helpers, no external formatter config.
- Provider streaming yields a fixed event vocabulary (`TextDelta`, `ToolCall`,
  `_ToolMeta`, `Done`); keep new providers conforming to it.
- **Every metered provider client gets `providers._timeout_kwargs()`** (or, where
  the SDK takes no `timeout=`, an `httpx.Client` carrying `_stream_timeout()`).
  The SDKs default to a 600s read timeout, and a stalled stream — a gateway
  sitting on a finished response, the observed Portkey failure — is not merely
  slow but *expensive*: `_run_turn_with_retry` re-runs the same turn, and
  Anthropic's prompt cache expires in ~5 minutes, so a 600s stall guarantees the
  retry re-writes the whole conversation prefix at 1.25x instead of re-reading it
  at 0.1x. `STREAM_IDLE_TIMEOUT` (120s, `GERBIL_STREAM_TIMEOUT` to override) is
  set to catch the stall while the cache is still alive; it bounds the *gap
  between chunks*, never the length of a response. ollama is deliberately exempt
  (a local server goes quiet while loading a model, and nothing is metered).
- Session events are append-only JSONL; never rewrite a log in place.
