# Supervisor mode (sub-agents)

:warning: experimental :warning:

`--agents` lets the model delegate: it can start **sub-agents** -- complete
gerbil sessions, each in its own sandbox container -- watch them, stop them,
and merge their work back.

```console
$ gerbil run --agents --model claude-opus-4-8 --prompt prompt.md
```

## How it works

```
host: runner (supervisor) ──► container S
   ├─ runner (sub-agent A) ──► container A    (starts from S's tree)
   │     └─ runner (A.1)   ──► container A.1  (starts from A's tree)
   └─ runner (sub-agent B) ──► container B
```

Every agent is an ordinary background run (`gerbil ps` lists them all,
`gerbil grab NAME` attaches to any of them). Its registry entry records its
`parent`, so the runs form a tree.

- **A sub-agent starts from its parent's current tree.** That tree is the host
  base commit, plus the patches that rebuilt the parent's own base, plus a
  snapshot of the parent's work at spawn time, committed or not. The sub-agent
  never sees changes its parent makes afterwards, and never sees its siblings.
- **A sub-agent's result is a patch** in its run directory
  (`~/.gerbil/running/<name>/`), not in your project. Only the supervisor at
  the top writes to `.gerbil/patches/`.
- **The parent chooses what to merge.** `merge_agent` checkpoint-commits the
  parent's work, then runs `git am -3` on the sub-agent's patch inside the
  parent's sandbox. If there is a conflict, nothing is applied and the
  conflict is reported to the model. Merged work reaches your repo the normal
  way: inside the supervisor's one patch, applied with `gerbil commit`.

## Tools

| Tool | |
| --- | --- |
| `spawn_agent(prompt, model?, max_turns?, can_spawn?)` | start a sub-agent; returns its name immediately |
| `list_agents()` | your sub-agents (and theirs): state, turns, merged or not, task |
| `peek_agent(name, events?)` | its recent session events and the files its work touches |
| `wait_agents(names, timeout_seconds?)` | block until they finish; report outcomes |
| `merge_agent(name)` | apply a finished *direct* sub-agent's patch to your tree |
| `kill_agent(name)` | stop a sub-agent and its whole subtree |
| `review_agent(name, action, ...)` | with an agent plan: answer a sub-agent's budget review (see below) |
| `request_budget(reason, ...)` | with an agent plan: ask your own supervisor for more budget |

## Isolation

- **Each agent has its own container.** Agents share no filesystem and get
  no access to the container runtime.
- **The supervisor tools run on the host**, in the calling agent's runner
  process, and only act on the caller's own descendants. A sibling's or
  ancestor's name gets the same "no sub-agent named ..." error as a name that
  doesn't exist.
- **Limits only get tighter further down the tree.**
  - `--max-agent-depth N` (default 2) sets how many levels of sub-agents may
    exist. A sub-agent can spawn only if its parent passed `can_spawn` and
    depth remains; otherwise it isn't offered `spawn_agent` at all.
  - `--max-agent-children N` (default 4) caps how many sub-agents each agent
    may spawn over its lifetime.
  - `--max-agents N` (default 8) caps how many sub-agents, and so containers,
    may run at once across the whole tree.
  - A sub-agent's `max_turns` can never exceed its parent's.

## Project agent plans (`.gerbil/agents.toml`)

Without a plan, `--agents` is free-form: the model spawns whatever it wants,
up to the numeric limits above. A plan file fixes the shape of the hierarchy
and the budgets in advance. If `<project>/.gerbil/agents.toml` exists,
`--agents` uses it; `--agents-config FILE` points at a different file (and
implies `--agents`).

```toml
root = "lead"                    # the role the top-level session plays

[limits]
max_live_agents = 6              # sub-agents running at once, tree-wide
on_root_exhausted = "ask"        # "ask" (a human decides) or "finish"

[roles.lead]
description = "Plans the proof and integrates the pieces"
task = "Prove Main.theorem. Split the work across your sub-agents."
budget = { tokens = 20_000_000, time = "3h", cost = 60.0 }
spawn = { x = 1, y = 1, z = 1 }  # may spawn one each of x, y, z
adhoc = 1                        # plus one improvised helper...
adhoc_budget = { tokens = 1_000_000, time = "20m" }   # ...with this budget

[roles.x]
description = "Builds the list lemmas"
task = "Prove the lemmas in Foo/List.lean; touch nothing else."
model = "claude-sonnet-4-5"      # default: the run's --model
budget = { tokens = 6_000_000, time = "1h" }
spawn = { grinder = 2 }          # x may itself spawn two grinders

[roles.y]
budget = { tokens = 4_000_000, time = "1h" }

[roles.z]
budget = { tokens = 4_000_000, time = "1h" }

[roles.grinder]
task = "Close exactly the one sorry you are given."
budget = { tokens = 1_000_000, time = "15m" }
```

- **Roles.** `spawn_agent` takes a `role`. An agent may spawn only the
  roles in its own role's `spawn` table, up to the counts given, plus `adhoc`
  helpers if allowed. Ad-hoc helpers can't spawn anything. The role sets the
  model, and its `task` is placed before the instructions the parent writes.
  The root role's `task` is placed before `--prompt`, which becomes optional.
- **The role graph must be a DAG**, and a child role's budget may not exceed
  its parent's in any dimension. The plan is checked at preflight, with every
  problem reported at once. It is frozen for the run, so editing the file
  mid-run changes nothing.
- **Budgets** can set any of `tokens`, `time` (`"90s"`, `"45m"`, `"1h30m"`)
  and `cost` (estimated USD; only allowed for models with known pricing).
  **tokens and cost count the agent's whole subtree** (itself plus all of
  its descendants), so a role's budget bounds everything below it. **time**
  is the agent's own wall clock.

### Running out: the review step

An agent that reaches any limit stops at its next turn boundary and **waits
for a review**. Its container stays up and its conversation stays intact.

- **Its parent decides,** through `review_agent`:
  - `extend` adds more budget, but only from the parent's own remaining
    headroom.
  - `finish` has the agent wrap up and produce its patch.
  - `kill` stops it.
- **The parent finds out right away.** It gets a note at its next turn
  boundary with the child's usage, the files it has touched and its latest
  message, and `wait_agents` returns early.
- **Asking for more before running out:** any agent can call `request_budget`
  to ask its own supervisor. It keeps working, and the answer arrives as a
  note.
- **Bubbling up:** a parent that can't afford an extension asks its own
  parent with `request_budget`, and so on up the tree. Because children's
  spending counts against their parent's subtree budget, children that
  overspend also push the parent itself into review.
- **At the root, the reviewer is you:**

  ```console
  $ gerbil ps                    # a waiting agent shows as "running (review)"
  $ gerbil review                # list open reviews: usage, progress, requests
  $ gerbil review brave-otter --extend --tokens 5000000 --time 30m
  $ gerbil review brave-otter --finish      # or --deny / --kill
  ```

  With `on_root_exhausted = "finish"`, a root that runs out simply wraps up,
  so unattended runs never wait for you.

Budgets are checked between turns, so an agent can overshoot by one turn,
plus the final commit-message turn when it is told to finish.

## What the logs record

Every agent writes its own ordinary session log (`~/.gerbil/sessions/`).
The relations between agents are recorded **in the logs themselves**, so a
tree can still be traced and replayed after the run registry is gone. The
registry is deleted when you dismiss a run.

- **Each agent has a stable `agent_id`** (a UUID; run names get reused).
- **Each log's `session_start` has an `agent` block** placing it in the tree:
  - its own id, its parent's id and run, and the **parent's session log**;
  - the root's id, its depth, role and starting budget;
  - the host `base_sha` and its own **`base_patch`**: the parent's snapshot it
    started from, embedded in full;
  - `base_chain_sha256`: the hashes of its full starting chain;
  - the root's log also embeds the frozen plan text, and every log records
    the plan's hash.
- **The parent's log records structured events:**
  - `agent_spawn`: id, run, role, model, budget, the full prompt, and the
    snapshot's hash;
  - `agent_outcome`: how the child ended, its session log, the patch hash and
    files;
  - `agent_merge`: the **applied patch itself**, or the conflict;
  - `agent_kill` and `agent_review`.
- **Each agent's own log records** `budget_review` (out of budget, or a
  request) and `budget_decision` (what was decided, and by whom, including
  `"human"`).

Readers that don't know these event types ignore them. Two tools use them:

- **`gerbil reconstruct-patch` on a sub-agent's log** rebuilds its starting
  tree by following the `parent_session` links: host `base_sha`, then each
  ancestor's `base_patch`. Every link is checked against `base_chain_sha256`,
  so a missing or edited parent log is an error, not a silently different
  tree.
- **On a parent's log, the replay re-applies each logged merge** at the point
  where it happened, so merged sub-agent work is reproduced too.

## Lifecycle

- **Sub-agents never outlive their parent.** However a runner ends, it cancels
  its live sub-agents first, and each of those cancels its own. Cancelled
  runs show the status `cancelled`.
- **The parent is warned before finishing.** If a parent tries to finish while
  sub-agents are still running, or finished but unmerged, it is sent back
  once with a note about them.
- **The tree's base commit must stay in your history.** A sub-agent rebuilds
  its tree on the tree's base commit, so that commit must still be an ancestor
  of the host's `HEAD`. Committing on top is fine; switching to an unrelated
  branch is not.

## Not supported yet

- `--agents` together with `--ralph`, `--fill-sorry`, or `--small-model`.
- Resuming a whole agent tree. Resuming the root (`gerbil resume`, or `r` in
  [`gerbil tree`](tree.md)) restores its supervisor tools and plan, but its
  old sub-agents are not revived. A sub-agent's log can't be resumed on its
  own, because its patch only applies on top of its parent's tree.
- Messages between a running parent and child, and DAG-shaped dependencies
  (a sub-agent built on several siblings' results).
