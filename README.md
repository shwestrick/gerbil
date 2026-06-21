# Gerbil

:warning: work-in-progress :warning

A teensy tiny agent for Lean projects, inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover), but with
Docker-based sandboxing and a built-in git-based workflow.

Gerbil sessions are self-contained and sandboxed: each session is run
in a container and produces a git commit.

### Setup (only have to do this once)
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

# this produces three timestamped files in the project directory:
# jsonl session data, and a git patch/commit
$ ls /path/to/lake/project/gerbil-*
gerbil-260621-190350.jsonl
gerbil-260621-190350.patch
gerbil-260621-190350.commit
```

You can then apply and commit if desired
```bash
$ cd /path/to/lake/project
$ git apply gerbil-260621-190350.patch
$ git commit -F gerbil-260621-190350.commit
```