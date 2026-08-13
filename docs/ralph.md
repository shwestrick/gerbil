# Ralph loops

A "Ralph loop" runs the same prompt over and over, each run starting from the
previous run's output. It suits prompts of the form "find a remaining `sorry`
and close it" -- one session's leftovers become the next session's work.

```console
$ gerbil run --model gemini-3.1-pro-preview --prompt ralph-prompt.md --ralph 3
```

`--ralph N` runs at most `N` sessions back-to-back **in one sandbox**, so the
Lean build cache and the mathlib oleans are paid for once, not `N` times. Each
session is committed inside the container, so the next one builds on it.

Every session produces its own log and patch, numbered:

```console
$ ls .gerbil/patches
gerbil-260623-235900-01.patch
gerbil-260623-235900-02.patch
gerbil-260623-235900-03.patch

$ gerbil commit     # applies all three, in order
$ git push
```

A single `gerbil commit` applies the whole chain.

## Stopping early

`--ralph_done SCRIPT` stops the loop once the work is actually finished.
After each session, gerbil runs that script inside the container, from the
project directory, on that session's committed working tree. Exit `0` stops the
loop; any non-zero exit means keep going.

```bash
#!/bin/bash
# stop once no sorries remain
! grep -rq --include='*.lean' '\bsorry\b' MyProject/
```

```console
$ gerbil run --prompt ralph-prompt.md --ralph 20 --ralph_done ./no-sorries.sh
```

This is the usual way to use `--ralph`: set `N` to a budget you're willing to
spend and let the check end the run when the goal is met.

When a termination check is installed, the agent also gets a `check_goal`
tool that runs the same script and shows it the verdict and output -- so a
session that believes it is done can verify against the actual stop
condition (and see exactly which part fails) instead of guessing.

For the common "prove these specific `sorry`s" task, you don't have to write
the prompt or the check yourself: [`--fill-sorry`](fill-sorry.md) generates
both (and a cross-session plan file) and defaults to a ralph loop.

## Resumability

Each session records the chain's base commit and the ordered list of ancestor
patches needed to rebuild its starting point. That makes *any* session in the
chain independently resumable -- point [`gerbil resume`](resume.md) at the
crashed session's log and it replays the earlier patches, restores that
session's working tree, and finishes the remaining iterations. The
`--ralph_done` script is recorded too, so a resumed chain keeps the same
termination check.
