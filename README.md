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

$ gerbil commit

$ git push
```


You can also run a multi-session Ralph loop, applying the same prompt
repeatedly. The option `--ralph N` runs at most `N` sessions. This produces
session logs and patches for every session. Running `gerbil commit` a single
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

$ gerbil commit

$ git push
```

To terminate the Ralph loop early, pass `--ralph_done SCRIPT`. After each
session completes, gerbil runs that script inside the container on the session's
committed working tree (from the project directory). If it exits `0`, the loop
stops; any non-zero exit means keep going.

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
$ gerbil update       # update to latest main: refreshes the source, rebuilds the
                      # image, and overwrites this launcher script in place
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

## Summarizing usage with `gerbil summarize`

`gerbil summarize` scans the project's `.gerbil/*.jsonl` session logs and reports
total token usage, an estimated cost (from the per-model pricing table), and
breakdowns by session status, tool call, and model:

```
$ gerbil summarize
gerbil summary -- 28 session(s) in /path/to/project/.gerbil

Tokens
  input:   141,094,502
  output:      367,974
  total:   141,462,476

Estimated cost
  ~$180.0479
...
```

Session logs are only included in the project `.gerbil/` when
`gerbil run --include-session` is used; otherwise they live in
`~/.gerbil/sessions/` (and `summarize` points you there if the project has
none).

## Committing patches with `gerbil commit`

After running sessions, `gerbil commit` looks in the project `.gerbil/` folder
and identifies patches that can be applied at the current git `HEAD`. Stale
and already-committed patches are ignored; these are safe to leave around
or delete.

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
next to the session log, refreshed after every turn. It is a `git
format-patch` from the session's base to the current state -- including any
commits the agent made itself, not just uncommitted changes -- so it never
loses work. It is a normal git patch: if you'd rather not resume, you can
just `git apply` it yourself. A clean finish deletes the `.wip.patch`; a
crash leaves it in place for `--resume`.

The continuation is written as its own session log and patch (named
`...-resume-<timestamp>`), carrying the full prior history forward, so it is
itself resumable if it too is interrupted. The original crashed log is left
untouched.

Ralph chains are supported. Point `--resume` at the crashed session's log
(e.g. `gerbil-<ts>-03.jsonl`) and gerbil rebuilds that session's starting
point by replaying the earlier sessions' patches on top of the chain's base
commit, reapplies the crashed session's working-tree patch, and then runs the
remaining iterations. Each ralph session's header records the chain's base
commit and the ordered list of ancestor patches needed to rebuild it, so the
sibling `.patch` files (found by the `gerbil-<ts>-NN` naming convention) are all
that's required -- including across a resume-of-a-resume. If the chain used a
`--ralph_done` check, its script is recorded in the session log, so the resumed
chain keeps the same termination check automatically (pass `--ralph_done` again
to override it).

Resume needs the same repository that produced the session, with the base
commit still in its history.

### Reconstructing a patch by replaying tool calls

Where `--resume` restores the working tree from the live `.wip.patch` snapshot,
`gerbil reconstruct-patch` rebuilds a session's `.patch` by *actually replaying
the session's tool calls* in a fresh sandbox -- no model involved:

```console
$ gerbil reconstruct-patch ~/.gerbil/sessions/gerbil-260623-235800.jsonl
```

gerbil recreates the session's base commit (replaying ancestor patches first for
a ralph session), re-executes every state-mutating tool call it logged
(`bash`, `write_file`, `edit_file`; read-only and `lean_*` calls are skipped),
commits the result under the session's own commit message, and writes the
corresponding `.patch`. This is useful when the original patch is missing or was
corrupted -- for example if the agent ran `git` commands that confused gerbil's
bookkeeping.

If the target `.patch` already exists, gerbil asks before overwriting it (and
does so up front, before the slow replay). Pass `--force` to overwrite without
the prompt.

Replay is only as deterministic as the commands themselves: `bash` that depends
on time, the network, or randomness may not reproduce exactly.

