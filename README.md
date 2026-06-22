# Gerbil

:warning: work-in-progress :warning

A teensy tiny agent for Lean projects, inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover), but with
Docker-based sandboxing and a built-in git-based workflow.

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

# this produces three timestamped files in the project's .gerbil/ directory:
# jsonl session data, and a git patch/commit
$ ls /path/to/lake/project/.gerbil/
gerbil-260621-190350.jsonl
gerbil-260621-190350.patch
gerbil-260621-190350.commit
```

You can then apply and commit if desired
```bash
$ cd /path/to/lake/project
$ git apply .gerbil/gerbil-260621-190350.patch
$ git commit -F .gerbil/gerbil-260621-190350.commit
```

To run many sessions back-to-back on the same prompt (each building on the last
as a series of commits), use `--ralph N`. Outputs are numbered per session, and
`scripts/apply-gerbil.sh` will apply + commit them in order:
```bash
$ uv run gerbil --at /path/to/lake/project --prompt prompt.md --ralph 5
$ cd /path/to/lake/project && /path/to/gerbil/scripts/apply-gerbil.sh
```

### Lean LSP tools (MCP)

By default gerbil also gives the agent the
[lean-lsp-mcp](https://github.com/oOo0oOo/lean-lsp-mcp) tools — proof state
(`lean_goal`), diagnostics, hover info, tactic trials (`lean_multi_attempt`),
and mathlib search — alongside the built-in `bash`/`read_file`/`write_file`/
`edit_file` tools. The MCP server runs inside the sandbox container (where the
Lean toolchain lives); gerbil connects to it over `docker exec`.

Pass `--no-mcp` to disable it and use only the built-in tools. If the MCP server
fails to start, gerbil warns and continues with the built-in tools.