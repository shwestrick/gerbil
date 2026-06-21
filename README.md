# Gerbil

:warning: work-in-progress :warning

A teensy tiny agent for Lean projects, inspired by
[lea-prover](https://github.com/chinmayhegde/lea-prover), but with
Docker-based sandboxing and a built-in git-based workflow.

Gerbil sessions are self-contained and sandboxed: each session is run
in a container, producing a git commit, for example:

```bash
# build the lean-sandbox Docker image (only have to do this once)
$ docker build -t lean-sandbox:latest .

# run a session
$ uv run gerbil --at /path/to/lake/project --prompt PROMPT.md

# this produces three timestamped files: jsonl session data, and a git patch/commit
$ ls gerbil-*
gerbil-260621-190350.jsonl
gerbil-260621-190350.patch
gerbil-260621-190350.commit

# can then apply and commit if desired
$ git apply gerbil-260621-190350.patch
$ git commit -F gerbil-260621-190350.commit
```

You can then apply and commit, if desired:
```bash
$ git apply gerbil-260621-190350.patch
$ git commit -F gerbil-260621-190350.commit
```