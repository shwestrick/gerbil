# gerbil documentation

gerbil is a teensy tiny autonomous coding agent for Lean 4 / Lake projects. It
runs every session inside a container and hands you back a git patch.

Start with the [top-level README](../README.md) for installation and a first
run. These pages go deeper.

## Getting set up

- **[Installation](install.md)** -- installing and updating the launcher,
  Docker vs. podman, what lives in `~/.gerbil`.
- **[Models and API keys](models.md)** -- choosing `--model`, the supported
  providers, local models via ollama, routing through Portkey.

## Running sessions

- **[The workflow](workflow.md)** -- how a session goes from prompt to commit:
  preflight checks, the base-commit contract, patches, `gerbil commit`,
  `gerbil cleanup`.
- **[Ralph loops](ralph.md)** -- running the same prompt across many
  back-to-back sessions with `--ralph`.
- **[Filling sorries](fill-sorry.md)** -- `--fill-sorry`: point gerbil at
  specific `sorry`s and let it generate the prompt, plan file, and goal
  check itself.
- **[Big-small mode](big-small.md)** -- a big model driving the session and
  delegating individual `sorry`s to a smaller, cheaper one.
- **[Resuming and reconstructing](resume.md)** -- recovering a session that
  crashed, and rebuilding a lost patch by replaying tool calls.
- **[Usage and cost](summarize.md)** -- what `gerbil summarize` reports and how
  the numbers are computed.

## The sandbox

- **[Sandboxing](sandbox.md)** -- what the container can and cannot see, the
  mathlib cache, submodules, and using your own image.
- **[Tools](tools.md)** -- the built-in `bash`/`read_file`/`write_file`/
  `edit_file` tools and the lean-lsp MCP tools.

## Contributing

- **[Development](development.md)** -- repository layout, running from source,
  building the sandbox image, and the test scripts.
