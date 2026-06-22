# Gerbil

:warning: work-in-progress :warning:

A teensy tiny agent for Lean projects, inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover), but with
Docker-based sandboxing, a git-based workflow, and built-in support for
Ralph loops.

Gerbil sessions are self-contained and sandboxed: each session is run
in a container and produces a git commit.

### Setup

`gerbil` is a single self-contained launcher script. Put it on your `PATH`; it
fetches its own source from GitHub on first use (requires `git`, `docker`, and
[`uv`](https://astral.sh/uv)):
```bash
$ curl -fsSL https://raw.githubusercontent.com/shwestrick/gerbil/main/bin/gerbil \
      -o ~/.local/bin/gerbil && chmod +x ~/.local/bin/gerbil
```
The sandbox Docker image is built automatically on the first `gerbil run` (and
rebuilt by `gerbil update`), tagged to match the gerbil version — no manual
`docker build` needed.

Managing the launcher itself:
```bash
$ gerbil --version    # current version (commit hash)
$ gerbil update       # update to the latest commit on main (+ rebuild the image)
```

### Run a session

Run from inside your Lake project (a git repo); `--at` defaults to the current
directory. Use `--prompt FILE` for the task.
```bash
$ cd my/lake/project
$ gerbil run --prompt prompt.md
```

The agent works on the real repository (with full history) inside the container,
and its changes are committed there. The live session log is written to
`~/.gerbil/` (the true session file); the project's `.gerbil/` only receives the
patch — a `git format-patch` (title, message, diff in one file):
```bash
$ ls ~/.gerbil/                  # session log + a copy of every patch
gerbil-260621-190350.jsonl
gerbil-260621-190350.patch
$ ls .gerbil/                    # patch only
gerbil-260621-190350.patch
```

Apply the patch(es) into your repo as real commits with `gerbil apply` (it
`git am`s every `.gerbil/*.patch` in order):
```bash
$ gerbil apply
```

The session log stays out of your project by default. Pass `--include-session`
to fold it into the commit itself (so applying the patch also records how the
change was produced).

`~/.gerbil/` is a user-level archive of all gerbil data — the live session logs
plus a copy of every patch — kept regardless of what you do with the
project-level files.

### Ralph loops

`--ralph N` runs N sessions back-to-back on the *same* prompt, each building on
the last as a series of commits.

```bash
$ cd my/lake/project
$ gerbil run --ralph 5 --prompt prompt.md
$ gerbil apply        # applies the whole numbered series in order
```

- The sandbox (and the lean-lsp MCP server) is reused across
  all sessions, so the mathlib cache is fetched only once.
- After each session gerbil commits the changes *inside* the container, so the
  next session starts from the previous one's result.
- Each session writes its own numbered output set,
  `.gerbil/gerbil-<ts>-NN.patch` (plus the log in `~/.gerbil/`).

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