# Resuming and reconstructing

## Resuming a crashed session

If a session dies partway through -- a transient API error that outlasted the
retries, a lost connection, a Ctrl-C -- you can continue it from its log:

```console
$ gerbil resume ~/.gerbil/sessions/gerbil-260623-235800.jsonl
```

gerbil boots a fresh sandbox, recreates the git state the session started from,
replays the conversation up to the crash, and continues. The model and prompt
come from the log, so `gerbil resume` takes neither `--prompt` nor `--model`
(and has no `--ralph` -- a resumed ralph chain continues on its own).

Resume needs the same repository that produced the session, with the base
commit still in its history.

### The `.wip.patch` snapshot

The working tree is recovered from a `*.wip.patch` file kept live next to the
session log and refreshed after every turn. It is a `git format-patch` from the
session's base to the current state -- **including commits the agent made
itself**, not just uncommitted changes -- so nothing is lost.

It is an ordinary git patch. If you would rather not resume, you can just apply
it yourself:

```console
$ git am ~/.gerbil/sessions/gerbil-260623-235800.wip.patch
```

A clean finish deletes the `.wip.patch`; a crash leaves it in place.

### What the continuation produces

The continuation is written as its own session log and patch, named
`...-resume-<timestamp>`, so it is itself resumable if it too is interrupted.
The original crashed log is left untouched.

That new log opens with the parent log re-emitted **in full** (every event,
tagged `"replayed": true`) followed by a `resumed` boundary marker. The crashed
parent never commits, so this replay is the only copy of its history that
reaches the project's `.gerbil/` -- the one log that does get committed records
the whole session from the beginning.

The session-log setting is inherited as well: if the original run folded its
log into the commit (the default), so does the continuation. Pass
`--omit-session-log` to force the log out.

### Ralph chains

Point `gerbil resume` at the crashed session's log (e.g. `gerbil-<ts>-03.jsonl`).
gerbil rebuilds that session's starting point by replaying the earlier
sessions' patches on top of the chain's base commit, reapplies the crashed
session's `.wip.patch`, and then runs the remaining iterations.

Each ralph session's header records the chain's base commit and the ordered
list of ancestor patches, so the sibling `.patch` files -- found by the
`gerbil-<ts>-NN` naming convention -- are all that's required, including across
a resume-of-a-resume. If the chain used `--ralph_done`, its script is recorded
in the log and reused automatically (pass `--ralph_done` again to override).

### Big-small sessions

A crash inside a `zoom_in` sub-session resumes *inside* that sub-session: the
inner events are tagged in the log, so gerbil rebuilds both conversations and
picks up where the small model left off. See [Big-small mode](big-small.md).

## Reconstructing a patch by replaying tool calls

Where `gerbil resume` restores the working tree from the `.wip.patch` snapshot,
`gerbil reconstruct-patch` rebuilds a session's `.patch` by *actually replaying
the session's tool calls* in a fresh sandbox -- no model involved:

```console
$ gerbil reconstruct-patch ~/.gerbil/sessions/gerbil-260623-235800.jsonl
```

gerbil recreates the session's base commit (replaying ancestor patches first
for a ralph session), re-executes every state-mutating tool call it logged
(`bash`, `write_file`, `edit_file`; read-only and `lean_*` calls are skipped),
commits the result under the session's own commit message, and writes the
corresponding `.patch`.

This is the recovery path when the original patch is missing or was corrupted
-- for example, if the agent ran `git` commands that confused gerbil's
bookkeeping.

If the target `.patch` already exists, gerbil asks before overwriting it, and
does so up front, before the slow replay. `--force` overwrites without asking.

Replay is only as deterministic as the commands themselves: `bash` that depends
on time, the network, or randomness may not reproduce exactly.
