# gerbil

:warning: work-in-progress, but fairly stable :warning:

A teensy tiny agent for Lean projects, inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover), but with
Docker-based sandboxing, a git-based workflow, and built-in support for
Ralph loops.

gerbil sessions are self-contained and sandboxed: each session is run
in a container and produces a git commit.

## Example Usage

Run at your Lake project root. This needs to be in a git repo, with a clean
state (no unstaged changes). gerbil produces two files: a `.jsonl` session
log, and a `.patch` that can be applied and pushed.

```console
$ cd /path/to/my/lake/project

$ export GOOGLE_API_KEY=...

$ gerbil run --model gemini-3.1-pro-preview --prompt prompt.md
...
--- 134 turns, 6,541,423 tokens (in: 6,517,329, out: 24,094), ~$8.3876 ---
session: /Users/shwestrick/.gerbil/sessions/gerbil-260623-235800.jsonl
patch:   /path/to/my/lake/project/.gerbil/gerbil-260623-235800.patch (git am)

$ ls .gerbil
gerbil-260623-235800.jsonl
gerbil-260623-235800.patch

$ gerbil apply

$ git push
```


You can also run a multi-session Ralph loop, applying the same prompt
repeatedly. The option `--ralph N` runs at most `N` sessions. This produces
session logs and patches for every session. Run `gerbil apply` a single
time will commit each of these.

```console
$ cd my/lake/project

$ export GOOGLE_API_KEY=...

$ gerbil run --model gemini-3.1-pro-preview --prompt ralph-prompt.md --ralph 3
...

$ ls .gerbil
gerbil-260623-235900-01.jsonl
gerbil-260623-235900-01.patch
gerbil-260623-235900-02.jsonl
gerbil-260623-235900-02.patch
gerbil-260623-235900-03.jsonl
gerbil-260623-235900-03.patch

$ gerbil apply

$ git push
```

In Ralph mode, gerbil has access to an additional tool called `ralph_done`,
which can be used to terminate the Ralph loop early. In the prompt, we
recommend specifying *very precisely* under what conditions the model is
allowed to invoke `ralph_done` (otherwise, the model might spuriously decide to
terminate the Ralph loop).

## Setup and Install

`gerbil` is a single self-contained launcher script. Put it on your `PATH`; it
fetches its own source from GitHub on first use (requires `git`, `docker`, and
[`uv`](https://astral.sh/uv)):
```console
$ curl -fsSL https://raw.githubusercontent.com/shwestrick/gerbil/main/bin/gerbil \
      -o ~/.local/bin/gerbil && chmod +x ~/.local/bin/gerbil
```
The sandbox Docker image is built automatically on the first `gerbil run` (and
rebuilt by `gerbil update`), tagged to match the gerbil version.

Docker must be usable **without sudo** (gerbil talks to the daemon via the
Docker SDK). On Linux, add yourself to the `docker` group
(`sudo usermod -aG docker $USER`, then re-login) or use
[rootless Docker](https://docs.docker.com/engine/security/rootless/).

Managing the launcher itself:
```console
$ gerbil --version    # current version (commit hash)
$ gerbil update       # update to the latest commit on main (+ rebuild the image)
```

## API Keys and Backend Models

Use `gerbil run --help` to see the list of backend models. Set the appropriate
API key for the model you wish to use:

```console
$ export GOOGLE_API_KEY=...

$ export ANTHROPIC_API_KEY=...

$ export OPENAI_API_KEY=...
```

## The `~/.gerbil` directory

gerbil maintains a `$HOME/.gerbil` directory that contains an archive of
all recent sessions and patches, and versions of the gerbil driver itself.
This directory is safe to delete at any time (but note that this will delete
all archived data).

gerbil also maintains a per-project `.gerbil/` directory inside of the
project where it is run, to store project-specific session data and patches.

## Lean LSP tools (MCP)

By default, gerbil enables all
[lean-lsp-mcp](https://github.com/oOo0oOo/lean-lsp-mcp) tools — proof state
(`lean_goal`), diagnostics, hover info, tactic trials (`lean_multi_attempt`),
and mathlib search — alongside the built-in `bash`/`read_file`/`write_file`/
`edit_file` tools. The MCP server runs inside the sandbox container (where the
Lean toolchain lives); gerbil connects to it over `docker exec`.

Use `gerbil run --no-mcp` to disable it and use only the built-in tools.
If the MCP server fails to start, gerbil warns and continues with the built-in
tools.

## Mathlib caching

By default, gerbil assumes the Lake project includes Mathlib, and starts
every sandbox session with `lake exe cache get`. Use `gerbil run --skip-cache`
to disable the initial `lake exe cache get`.

## Turn limits

Use `gerbil run --max-turns N` to forcibly terminate sessions after `N` turns.

## Include session data in commits

Use `gerbil run --include-session` to include the `.jsonl` session data in
the generated patch.

## Resuming a crashed session

If a session dies partway through -- a transient API error ("service
unavailable"), a lost connection, a Ctrl-C -- you can resume
it from its session log:

```console
$ gerbil run --resume ~/.gerbil/sessions/gerbil-260623-235800.jsonl
```

gerbil boots a fresh sandbox, recreates the git state the session started
from, replays the conversation up to the crash, and continues from there.
The model and prompt are taken from the log, so `--resume` takes neither
`--prompt` nor `--model` (and is not combined with `--ralph`).

The working tree is recovered from a `*.wip.patch` file that is kept live
next to the session log, refreshed after every turn. This patch is also
a normal git patch -- if you'd rather not resume, you can just `git apply`
it yourself. A clean finish deletes the `.wip.patch`; a crash leaves it in
place for `--resume`.

The continuation is written as its own session log and patch (named
`...-resume-<timestamp>`), carrying the full prior history forward, so it is
itself resumable if it too is interrupted. The original crashed log is left
untouched.

Resume needs the same repository that produced the session, with `base_commit`
still in its history. (Multi-session `--ralph` resume is not yet supported.)

## Applying commits with `gerbil apply`

After running sessions, `gerbil apply` looks in the project `.gerbil/` folder
and identifies patches that can be applied at the current git `HEAD`. Stale
and already-applied patches are ignored; these are safe to leave around
or delete.