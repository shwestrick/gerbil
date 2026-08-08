# gerbil

:warning: work-in-progress, but fairly stable :warning:

A teensy tiny agent in a box, for Lean projects. Inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover), but with
container-based sandboxing (Docker or podman), a git-based workflow, and
built-in support for Ralph loops.

gerbil sessions are self-contained and sandboxed: each session runs in a
container and produces a git commit. The container only ever sees the current
branch -- the git repo uploaded into the sandbox is stripped to that branch's
history, with no other branches, tags, remotes, upstreams, or reflogs.

## Install

`gerbil` is a single self-contained launcher script. Put it on your `PATH`; it
fetches its own source from GitHub on first use. Requires `git`,
[`uv`](https://astral.sh/uv), and `docker` (or `podman`).

```console
curl -fsSL https://raw.githubusercontent.com/shwestrick/gerbil/main/bin/gerbil \
     -o ~/.local/bin/gerbil && chmod +x ~/.local/bin/gerbil
```

The sandbox image is built automatically on the first `gerbil run`. Later,
`gerbil update` updates gerbil and rebuilds the image.

More: [Installation](docs/install.md).

## Use

Run at your Lake project root, in a git repo with a clean working tree. Write
your task in a file and hand it to gerbil:

```console
$ cd /path/to/my/lake/project

$ export ANTHROPIC_API_KEY=...

$ gerbil run --model claude-opus-4-8 --prompt prompt.md
...
--- 139 turns, 17,742,881 tokens (in: 277 + 17,114,370 cache-read + 494,976 cache-write, out: 133,258), ~$14.9836 ---
session: /Users/shwestrick/.gerbil/sessions/gerbil-260722-171419.jsonl
patch:   /path/to/my/lake/project/.gerbil/gerbil-260722-171419.patch (git am)
```

Each session produces a `.jsonl` log of everything that happened and a `.patch`
holding the work. Nothing touches your repo until you apply it:

```console
$ gerbil commit
$ git push
```

Other subcommands: `gerbil resume` (continue a crashed session), `gerbil
summarize` (token and cost accounting), `gerbil cleanup` (drop
already-committed patches), `gerbil reconstruct-patch`. Every command takes
`--help`.

More: [workflow](docs/workflow.md).

## Documentation

- [Installation](docs/install.md) -- installing and updating, Docker vs. podman,
  `~/.gerbil`
- [Models and API keys](docs/models.md) -- providers, local models via ollama,
  Portkey gateways
- [Workflow](docs/workflow.md) -- prompt, patch, commit, etc.
- [Ralph loops](docs/ralph.md) -- `--ralph N`, and stopping when the work is
  actually done
- [Big-small mode](docs/big-small.md) -- a big model delegating individual
  `sorry`s to a cheaper one
- [Resuming and reconstructing](docs/resume.md) -- recovering a crashed session
  or a lost patch
- [Usage and cost reporting](docs/summarize.md) -- reading `gerbil summarize`
- [Sandboxing](docs/sandbox.md) -- containment, mathlib caching, submodules,
  custom images
- [Tools](docs/tools.md) -- the built-in tools and the lean-lsp MCP tools
- [Development](docs/development.md) -- layout, running from source, tests

## License

[MIT](LICENSE)
