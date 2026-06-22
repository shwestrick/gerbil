# Gerbil

:warning: work-in-progress :warning:

A teensy tiny agent for Lean projects, inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover), but with
Docker-based sandboxing, a git-based workflow, and built-in support for
Ralph loops.

Gerbil sessions are self-contained and sandboxed: each session is run
in a container and produces a git commit.

### Setup
```bash
$ ... # git clone this repo, cd gerbil

# build the lean-sandbox Docker image (only have to do this once)
$ docker build -t lean-sandbox:latest src/lean-sandbox
```

### Run a session
Use `--at PATH` to specify where the Lake project is. This path
needs to be inside of a git repo, and should be the root of the Lake
project.

Use `--prompt FILE` to pass an initial prompt.

```bash
# run a session
$ uv run gerbil --at /path/to/lake/project --prompt prompt.md

# this produces two timestamped files in the project's .gerbil/ directory:
# the jsonl session log, and a git format-patch of the session's commit
$ ls /path/to/lake/project/.gerbil/
gerbil-260621-190350.jsonl
gerbil-260621-190350.patch
```

The agent works on the real repository (with full history) inside the container,
and its changes are committed there. The `.patch` is a `git format-patch` —
title, message, and diff in one file — so you apply it with `git am`:
```bash
$ cd /path/to/lake/project
$ git am .gerbil/gerbil-260621-190350.patch
```

Pass `--include-session` to fold the `.jsonl` session log into the commit itself
(so applying the patch also records how the change was produced). In that case
the loose `.jsonl` is dropped, since it lives in the patch.

Every session log and patch is also archived to `~/.gerbil/` (same filenames),
so you keep a full record no matter what you do with the project-level files.

### Ralph loops

`--ralph N` runs N sessions back-to-back on the *same* prompt, each building on
the last as a series of commits.

```bash
$ uv run gerbil --at /path/to/lake/project --prompt prompt.md --ralph 5
```

- The sandbox (and the lean-lsp MCP server) is reused across
  all sessions, so the mathlib cache is fetched only once.
- After each session gerbil commits the changes *inside* the container, so the
  next session starts from the previous one's result.
- Each session writes its own numbered output set,
  `.gerbil/gerbil-<ts>-NN.{jsonl,patch}`.

Apply the whole series into your repo in order with the helper script, which
`git am`s each format-patch as a real commit:
```bash
$ cd /path/to/lake/project && /path/to/gerbil/scripts/apply-gerbil.sh
```

**Stopping early.** In ralph mode the agent has a `ralph_done` tool; when it
calls it, the loop stops after that session instead of running the rest. The
tool just signals "done" — your prompt should explain *when* the task counts as
complete (e.g. "once `lake build` succeeds with no `sorry`, call ralph_done").

### Lean LSP tools (MCP)

By default gerbil also gives the agent the
[lean-lsp-mcp](https://github.com/oOo0oOo/lean-lsp-mcp) tools — proof state
(`lean_goal`), diagnostics, hover info, tactic trials (`lean_multi_attempt`),
and mathlib search — alongside the built-in `bash`/`read_file`/`write_file`/
`edit_file` tools. The MCP server runs inside the sandbox container (where the
Lean toolchain lives); gerbil connects to it over `docker exec`.

Pass `--no-mcp` to disable it and use only the built-in tools. If the MCP server
fails to start, gerbil warns and continues with the built-in tools.