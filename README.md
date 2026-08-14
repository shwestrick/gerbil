# gerbil

:warning: work-in-progress, but fairly stable :warning:

<img src="assets/gerbil.png" align="right" width="180" hspace="20" alt="gerbil sitting in its sandbox">

A sandboxed agent for Lean projects. gerbil sessions run in a
container (Docker or podman) and produce git patches.

```console
$ gerbil run --model claude-opus-4-8 \
             --fill-sorry MyProject.lemma1
...
$ gerbil commit
$ git push
```

![the gerbil live viewer: session stats and per-file diff on the left, the
scrolling turn-by-turn stream on the right](assets/screenshot.png)

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

More: [Installation](docs/install.md)

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

More: [Workflow](docs/workflow.md)

## Filling in `sorry`s

For the most common task -- "prove these `sorry`s" -- you don't have to write
a prompt at all. Point gerbil at the sorries, by position or by declaration
name, and it generates the prompt, a cross-session plan file, and a goal check
scoped to exactly those sorries, then loops until the check passes:

```console
$ gerbil run --fill-sorry MyProj/Basic.lean:42
$ gerbil run --fill-sorry MyNs.foo,MyProj/B.lean:107
```

A task spec (`--fill-sorry task.toml`) adds off-limits paths, an axiom policy,
approach notes, and more; `gerbil new-fill` opens your editor on a template of
every key and starts the run once you confirm.

More: [Filling sorries](docs/fill-sorry.md)

## Documentation

- [Installation](docs/install.md) -- installing and updating, Docker vs. podman,
  `~/.gerbil`
- [Models and API keys](docs/models.md) -- providers, local models via ollama,
  Portkey gateways
- [Workflow](docs/workflow.md) -- prompt, patch, commit, etc.
- [Ralph loops](docs/ralph.md) -- `--ralph N`, and stopping when the work is
  actually done
- [Filling sorries](docs/fill-sorry.md) -- `--fill-sorry` and `gerbil
  new-fill`: designate `sorry`s and let gerbil write the task
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
