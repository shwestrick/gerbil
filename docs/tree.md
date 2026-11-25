# Browsing sessions: `gerbil tree`

`gerbil tree` opens a full-screen browser over your session archive
(`~/.gerbil/sessions/`). It links the sessions into trees and groups them by
project.

```console
$ gerbil tree                 # every project
$ gerbil tree --at .          # just this project
$ gerbil tree --plain         # print the forest instead (also used when piped)
```

```
/home/me/proofs
├─ “main theorem” gerbil-261123-101500 · lead · error · 41t
│  ├─ gerbil-261123-101622-brave-otter · x · complete · 18t
│  │  └─ gerbil-261123-102010-calm-wren · grinder · complete · 9t
│  ├─ gerbil-261123-101640-keen-lynx · y · complete · 12t
│  └─ ↻ gerbil-261123-101500-resume-261123-120000 · complete · 7t
└─ gerbil-261123-140000-01 · complete · 30t
   ├─ ⟳ gerbil-261123-140000-02 · complete · 25t
   └─ ⟳ gerbil-261123-140000-03 · interrupted · 4t
```

The links are all read from the logs themselves:

| Link | Shown as | Meaning |
| --- | --- | --- |
| agent | plain child | a sub-agent under the session that spawned it ([supervisor mode](agents.md)) |
| resume | `↻` | a `gerbil resume` continuation under the session it continued |
| ralph | `⟳` | sessions 2..N of a ralph chain, listed flat under the first |

A session whose parent log has been deleted is shown as a root, marked
`orphaned`. Each row shows the session's name (if you gave it one), its log
name, its role, its status (`complete`, `error`, `interrupted`, `running`,
or `unfinished` for a run that died without recording why) and its turn
count. The right pane shows details: project, model, tokens, agent id and
budget, the error if any, whether it can be resumed, the prompt, and the
last message.

## Unusual archives

Session logs are plain files that can be copied or hand-edited, so the
viewer never assumes they form a clean forest. Every session in the archive
is always listed, and any session it had to treat specially is marked `!`
with a `note` in the details pane:

- **Cycles** (A continues B, B is a sub-agent of A, and so on) are broken at
  the cycle's oldest session, which becomes a root. The note names the
  ignored link and the cycle.
- **An agent id claimed by more than one log** (usually a copied log) is
  ambiguous. Sub-agents pointing at it are left unlinked with an
  explanation, never attached to a guessed parent.
- **A link to itself** is ignored.
- **Malformed fields** (wrong types, junk lines) are treated as absent and
  noted. Files that aren't readable session logs are skipped, and the number
  skipped is reported.
- **Very deep chains** work. The viewer nests at most 64 levels; anything
  deeper is listed flat at that level, tagged with its real depth
  (`⋯ depth 812`). `--plain` and deletion use the true structure.

## Keys

| Key | |
| --- | --- |
| `r` | **resume** the selected session: the viewer exits and runs `gerbil resume` on it, in its project |
| `g` | **grab** it, if it is a live run (exits into `gerbil grab`) |
| `n` | **name** it; an empty name removes the name |
| `d` | **delete** it and everything under it (asks first) |
| `R` | refresh |
| `q` | quit |

## What can be resumed

Only crashed or interrupted sessions can be resumed. Complete sessions have
nothing left to do, and running ones should be grabbed instead.

- **A sub-agent's session can't be resumed on its own.** Its patch only
  applies on top of its parent's tree. `gerbil resume` refuses one with the
  same explanation.
- **Resuming a tree's root restores supervisor mode.** It gets its sub-agent
  tools back, and under an agent plan the plan is restored from the root's
  log. Its old sub-agents are gone, so it may need to spawn new ones. Budgets
  count from zero again for the continuation.

## Names and deletion

- **Names live in `~/.gerbil/sessions/labels.json`, not in the logs.** Logs
  are append-only and may already be folded into commits. A tree's name is
  its root's name.
- **Deleting a node deletes its whole subtree** (sessions below it in any of
  the three kinds of link). That means each session's `.jsonl` log, its
  leftover `.wip.patch`, and the archive copy of its `.patch`. Patches in
  your project's `.gerbil/patches/` and anything already committed are not
  touched.
- **A subtree containing a live run can't be deleted.** Stop the run first.
- **Deletion is permanent.** Deleting a session also stops its descendants
  from being reconstructed, since a sub-agent's starting point is rebuilt
  from its ancestors' logs.
