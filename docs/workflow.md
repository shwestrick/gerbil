# The workflow

gerbil is git-native: a session takes your repository as input and produces a
**patch** as output. Nothing on the host is modified until you explicitly apply
it.

```
     your repo (clean)                       your repo (+1 commit)
            |                                          ^
            | upload tracked files + sanitized .git    | gerbil commit  (git am)
            v                                          |
     +--------------+   agent loop    +----------+   .gerbil/*.patch
     |  container   | --------------> | squash + | ------+
     +--------------+                 |format-   |
                                      | patch    |
                                      +----------+
```

## Preflight

`gerbil run` refuses to start unless:

- you are in a git repository with at least one commit;
- there is a lakefile (`lakefile.toml` or `lakefile.lean`);
- the working tree is clean (no unstaged or uncommitted changes);
- any submodules are initialized, in place, and clean;
- a container runtime is usable (a reachable Docker daemon, or a working
  `podman`).

The clean-tree requirement exists because the current `HEAD` is the *base
commit*, and the base commit is the contract for everything that follows.

## What gets uploaded

Into the container go all git-tracked files plus a **sanitized `.git`**:
gerbil builds a fresh repository containing only the current branch's history.
No other branches, no tags, no remotes, no upstreams, no reflogs, no
unreachable objects. The agent cannot see, and cannot reach, anything outside
the branch you ran it on.

Submodules are uploaded fully populated so the project builds, but are
effectively read-only to the agent; see [Sandboxing](sandbox.md#submodules).

## The agent loop

The model streams a turn, gerbil executes any tool calls it makes
([tools](tools.md)), records everything to an append-only `.jsonl` session log,
and repeats. When the model stops calling tools, one extra turn asks it to
write a commit message.

`--max-turns N` caps the number of turns (the default is unlimited -- the
session runs until the model is done).

Transient API failures (rate limits, "service unavailable", dropped
connections) are retried automatically. A session that dies anyway can be
picked up again with [`gerbil resume`](resume.md).

Each turn opens with a rule giving the turn number, the wall-clock time, and
how full the context window is:

```
──── turn 12 · 14:32:07 ──────────────────  [context: 96,000 / 200,000 (48.0%)]
```

The rule is drawn with box-drawing characters when the terminal can encode
them and with dashes when it can't -- the two are the same width, so nothing
shifts between them. Set `GERBIL_ASCII=1` to force the plain-text form
everywhere; `NO_COLOR` (see [no-color.org](https://no-color.org/)) drops the
color separately. Sessions long enough to run low on context also announce
that here -- see [Running out of context](models.md#running-out-of-context).

## Finalizing

The session's work -- the agent's own intermediate commits *plus* whatever it
left uncommitted -- is squashed into a single commit, and gerbil writes

```
git format-patch <base>..HEAD
```

to the project's `.gerbil/` directory. **Anything not reachable from that range
is lost**, which is why the system prompt forbids the agent from running `git
reset`, `checkout`, `stash`, or `init`.

Outputs:

| File | Location |
| --- | --- |
| `gerbil-<ts>.jsonl` | live in `~/.gerbil/sessions/`, archived there permanently |
| `gerbil-<ts>.patch` | the project's `.gerbil/`, plus an archive copy |
| `gerbil-<ts>.wip.patch` | next to the log, only while the session is unfinished |

By default the session log is folded into the commit the patch carries, so the
transcript lands in the repository alongside the proof. `--omit-session-log`
keeps it out; the archive copy in `~/.gerbil/sessions/` is kept either way.

## Applying the work

```console
$ gerbil commit
$ git push
```

`gerbil commit` runs `git am` over the project's `.gerbil/*.patch` files in
order. It skips patches that are already committed (identified by stable
patch-id, so a rebase doesn't confuse it) and patches that no longer apply.
Leftover patch files are harmless.

`gerbil cleanup` deletes the `.gerbil/*.patch` files whose changes are already
committed, using that same patch-id test. Patches that have not been committed
yet are never touched, and the `~/.gerbil/sessions/` archive copies are kept.

## Reviewing before applying

The `.patch` is an ordinary `git format-patch` file. You do not have to use
`gerbil commit` at all:

```console
$ git apply --stat .gerbil/gerbil-260623-235800.patch   # what changed
$ git apply --check .gerbil/gerbil-260623-235800.patch  # would it apply?
$ git am .gerbil/gerbil-260623-235800.patch             # apply it yourself
```
