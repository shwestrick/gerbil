# Installation

`gerbil` is a single self-contained launcher script. You put it on your `PATH`,
and it fetches its own source from GitHub on first use.

Requirements:

- [`uv`](https://astral.sh/uv) (the launcher runs gerbil's Python through it)
- `git`
- `docker` or [`podman`](https://podman.io)

```console
curl -fsSL https://raw.githubusercontent.com/shwestrick/gerbil/main/bin/gerbil \
     -o ~/.local/bin/gerbil && chmod +x ~/.local/bin/gerbil
```

The sandbox image is built automatically on the first `gerbil run`, tagged to
match the gerbil version. That first build takes a few minutes (it installs a
Debian base plus [elan](https://github.com/leanprover/elan)); afterwards it is
cached.

## Updating

```console
$ gerbil --version    # current version (commit hash)
$ gerbil update       # update to latest main
```

`gerbil update` refreshes the source, rebuilds the version-matched sandbox
image, prunes superseded images, and overwrites the launcher script in place --
so the copy on your `PATH` stays current too.

## Docker and/or podman

Docker must be usable **without sudo**, since gerbil talks to the daemon
through the Docker SDK. On Linux, either add yourself to the `docker` group
(`sudo usermod -aG docker $USER`, then re-login) or use
[rootless Docker](https://docs.docker.com/engine/security/rootless/).

Where Docker isn't available, set `GERBIL_SANDBOX` to run sessions under podman
instead:

```console
$ export GERBIL_SANDBOX=podman     # "docker" (the default) or "podman"
```

Everything else is unchanged: the same launcher builds the same image and runs
the same sessions, just through `podman`. No daemon or socket service is needed
-- gerbil drives podman's CLI directly. Because the image is per-runtime, the
first `gerbil run` after switching rebuilds it.

On a host where rootless podman has no `/etc/subuid` range, uid 1000 can be
neither chowned to at build time nor mapped at run time. The launcher detects
this and builds a variant image that runs as container-root instead (tag suffix
`-rootuser`). Nothing on the host is affected either way -- gerbil never bind
mounts; files enter the container as a tar and leave as `git format-patch`
text.

## The `~/.gerbil` directory

gerbil maintains a `$HOME/.gerbil` directory holding:

- `sessions/` -- an archive of recent session logs (`.jsonl`) and patches
- `versions/` -- checked-out source trees, one per gerbil version
- `repo/` -- a cache clone used for fetching updates
- `current` -- the active version's commit hash

This directory is safe to delete at any time; the next `gerbil` invocation
rebuilds what it needs. Deleting it does discard the archived session data.

Separately, gerbil maintains a per-project `.gerbil/` directory inside each
project it runs on: `patches/` (session patches), `sessions/` (committed
session logs), `plans/` (fill-sorry plan files), `tasks/` (`gerbil new-fill`
task specs), and an optional
[`config.toml`](sandbox.md#per-project-and-per-user-configuration). A
user-level `~/.gerbil/config.toml` can hold the same keys as personal
defaults; a project's values win.

## Environment variables

| Variable | Meaning |
| --- | --- |
| `GERBIL_SANDBOX` | `docker` (default) or `podman` |
| `GERBIL_SANDBOX_IMAGE` | Sandbox image to use; the launcher sets this to its version-matched build (see [image selection](sandbox.md#using-your-own-sandbox-image)) |
| `GERBIL_HOME` | Override `~/.gerbil` |
| `GERBIL_REPO_URL` | Override the GitHub repo the launcher fetches from |
| `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `PORTKEY_API_KEY`, `PORTKEY_BASE_URL`, `OLLAMA_HOST` | See [Models and API keys](models.md) |
