"""Supervisor mode: agents that spawn, monitor, and merge sub-agents.

With `gerbil run --agents`, the model gets tools to start *sub-agents* -- each
one a complete, ordinary gerbil run with its own detached runner process, its
own sandbox container, and its own session log -- and to watch, stop, and
merge them. A sub-agent may itself be allowed to spawn, so the agents form a
tree.

How the pieces fit:

- **Identity is the run registry.** A sub-agent is a run in
  ~/.gerbil/running/<name>/ (runs.py) whose meta.json carries `parent`,
  `root`, `depth`, `caps`, and the `base_sha` + `base_chain` that rebuild its
  starting tree. The parent pointers ARE the tree; runs.is_descendant is the
  one visibility rule, checked by every tool here.

- **Isolation is enforced on the host, not requested in a prompt.** These
  tools execute in the parent's runner process on the host. No agent gets
  the container socket, and each agent's container holds only its own copy
  of the project, so an agent can reach another agent only through these
  tools -- and they refuse any target outside the caller's own subtree, with
  the very same error a nonexistent name gets (siblings and ancestors are
  not even confirmed to exist).

- **Work moves as patches, exactly like everything else in gerbil.** A child
  starts from its parent's *current* tree: the host base commit plus the
  parent's base_chain plus a fresh snapshot of the parent's work
  (LeanSandbox.wip_patch). It finishes with an ordinary squashed patch,
  written into ITS run dir (never the project's .gerbil/ -- only the root
  agent ever writes there). The parent decides whether to `git am -3` it
  into its own sandbox (merge_agent); merged work then reaches the host the
  usual way, inside the root's one patch and `gerbil commit`.

- **Caps only shrink downward.** Each agent's caps bound what it may spawn:
  depth_left (0 = no spawning; the tools are not even advertised to it),
  max_children (lifetime spawns per agent), max_agents (live agents in the
  whole tree, the container-count bound), and an optional per-agent
  max_turns. A child's caps are derived from its parent's, never larger.

- **Nobody outlives their parent.** runs.run_runner cancels a run's live
  children however it exits, and each cancelled child cascades further down.
  Before that happens, the parent model is nudged once (stop_note) about
  children still running or finished-but-unmerged.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from . import runs
from .tools import ToolResult
from .view import patch_file_stats

DEFAULT_MAX_DEPTH = 2
DEFAULT_MAX_CHILDREN = 4
DEFAULT_MAX_AGENTS = 8

# wait_agents polls the registry at this interval, and blocks at most this
# long per call (the model can always call it again).
WAIT_POLL_SECONDS = 3.0
WAIT_DEFAULT_SECONDS = 600
WAIT_MAX_SECONDS = 3600

LIVE_STATES = ("running", "paused")


@dataclass
class Caps:
    depth_left: int
    max_children: int
    max_agents: int
    max_turns: int | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> "Caps":
        d = d or {}
        return cls(
            depth_left=int(d.get("depth_left", 0)),
            max_children=int(d.get("max_children", 0)),
            max_agents=int(d.get("max_agents", 0)),
            max_turns=d.get("max_turns"),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def for_child(self, can_spawn: bool, max_turns: int | None) -> "Caps":
        """The caps a child of this agent gets: one level less depth (none if
        the parent said it may not spawn), the same per-agent child limit and
        tree-wide agent limit, and a turn cap no larger than this agent's."""
        turns = [t for t in (self.max_turns, max_turns) if t]
        return Caps(
            depth_left=max(0, self.depth_left - 1) if can_spawn else 0,
            max_children=self.max_children,
            max_agents=self.max_agents,
            max_turns=min(turns) if turns else None,
        )


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_NAME_PROP = {"type": "string", "description": "The sub-agent's name, as spawn_agent returned it."}

SPAWN_AGENT_TOOL = {
    "name": "spawn_agent",
    "description": (
        "Start a sub-agent: a separate, full agent session in its own isolated "
        "sandbox, starting from a snapshot of YOUR current working tree. It runs "
        "concurrently -- this returns its name immediately. It cannot ask you "
        "questions, so the prompt must be self-contained: the exact task, the "
        "relevant files and context, the rules it must follow, and what 'done' "
        "means. Its changes do not appear in your tree until you merge_agent it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {"type": "string", "description": "The sub-agent's complete task description."},
            "model": {"type": "string", "description": "Model for the sub-agent (default: your own model)."},
            "max_turns": {"type": "integer", "description": "Turn cap for the sub-agent (optional)."},
            "can_spawn": {"type": "boolean", "description": "Whether the sub-agent may spawn sub-agents of its own (default false; subject to the depth limit)."},
        },
        "required": ["prompt"],
    },
}

LIST_AGENTS_TOOL = {
    "name": "list_agents",
    "description": "List your sub-agents (and theirs, indented): state, model, turns, whether merged, and their task.",
    "input_schema": {"type": "object", "properties": {}},
}

PEEK_AGENT_TOOL = {
    "name": "peek_agent",
    "description": (
        "Look at what a sub-agent is doing: its state, the files its work touches "
        "so far, and the most recent events of its session (turns, tool calls, results)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": _NAME_PROP,
            "events": {"type": "integer", "description": "How many recent events to show (default 15, max 60)."},
        },
        "required": ["name"],
    },
}

WAIT_AGENTS_TOOL = {
    "name": "wait_agents",
    "description": (
        "Block until every named sub-agent has finished (or the timeout passes), "
        "then report each one's outcome: final state, its commit message, and the "
        "files its patch changes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "names": {"type": "array", "items": {"type": "string"}, "description": "Sub-agent names to wait for."},
            "timeout_seconds": {"type": "integer", "description": f"Give up waiting after this long (default {WAIT_DEFAULT_SECONDS}, max {WAIT_MAX_SECONDS})."},
        },
        "required": ["names"],
    },
}

MERGE_AGENT_TOOL = {
    "name": "merge_agent",
    "description": (
        "Apply a finished direct sub-agent's patch to YOUR working tree (git am -3, "
        "after checkpoint-committing your own current work). On a conflict nothing "
        "is applied and the conflict is reported. Review its outcome before merging."
    ),
    "input_schema": {
        "type": "object",
        "properties": {"name": _NAME_PROP},
        "required": ["name"],
    },
}

KILL_AGENT_TOOL = {
    "name": "kill_agent",
    "description": "Stop a running sub-agent (and all of its own sub-agents). Its unfinished work is discarded.",
    "input_schema": {
        "type": "object",
        "properties": {"name": _NAME_PROP},
        "required": ["name"],
    },
}

REVIEW_AGENT_TOOL = {
    "name": "review_agent",
    "description": (
        "Answer a direct sub-agent's budget review (it ran out of budget and is "
        "paused, or it asked for more). extend: grant more budget -- it comes out "
        "of YOUR remaining budget; finish: it wraps up now and produces its patch; "
        "deny: refuse a request (it keeps working within its budget); kill: stop "
        "it and discard its work. Look at its progress (peek_agent) before "
        "extending."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "name": _NAME_PROP,
            "action": {"type": "string", "enum": ["extend", "finish", "deny", "kill"]},
            "tokens": {"type": "integer", "description": "extend: extra tokens."},
            "minutes": {"type": "number", "description": "extend: extra wall-clock minutes."},
            "cost": {"type": "number", "description": "extend: extra estimated USD."},
            "note": {"type": "string", "description": "Guidance for the sub-agent, delivered with the decision."},
        },
        "required": ["name", "action"],
    },
}

SUPERVISOR_TOOLS = [
    SPAWN_AGENT_TOOL, LIST_AGENTS_TOOL, PEEK_AGENT_TOOL,
    WAIT_AGENTS_TOOL, MERGE_AGENT_TOOL, KILL_AGENT_TOOL,
]
SUPERVISOR_TOOL_NAMES = {t["name"] for t in SUPERVISOR_TOOLS}


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------

def tree_members(root: str) -> list[dict]:
    """Every registered run belonging to the tree rooted at `root` (the root
    itself included), in list_runs() shape."""
    return [r for r in runs.list_runs()
            if r.get("root") == root or r.get("name") == root]


def load_result(name: str) -> dict | None:
    """A finished sub-agent's result.json (see write_result), or None."""
    return runs._read_json(runs.run_dir(name) / "result.json")


def write_result(name: str, *, final_text: str, commit_message: str,
                 patch_path: Path | None, session_path: Path) -> None:
    """Record a sub-agent's outcome in its run dir, for its parent's
    wait_agents / merge_agent. Written by cli.cmd_run when a child's session
    finalizes; absent when the child died, errored, or was cancelled."""
    runs._write_json(runs.run_dir(name) / "result.json", {
        "final_text": final_text,
        "commit_message": commit_message,
        "patch": str(patch_path) if patch_path else None,
        "session": str(session_path),
        "finished_at": time.time(),
    })


def _log(session_getter, kind: str, /, **fields) -> None:
    """Record a structured event in the in-flight session log, if there is
    one. Never raises: bookkeeping must not take a session down."""
    try:
        session = session_getter() if session_getter else None
        if session is not None:
            session.record_event(kind, **fields)
    except Exception:
        pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def session_agent_info(run: str) -> dict | None:
    """The `agent` block of a session_start: this session's place in the
    agent tree, self-contained enough to reconstruct the tree -- and each
    agent's starting point -- from logs alone, after the run registry is gone.

    Links go by agent_id (a uuid minted at spawn; run names are recycled once
    dismissed) and by session log path (the archive in ~/.gerbil/sessions/ is
    permanent). The starting tree is the root's base_sha plus, down the
    ancestry, each agent's own `base_patch` -- so each log embeds only its own
    snapshot, and the parent_session link supplies the rest; `base_chain_sha256`
    lists the hashes of the full chain so a reconstruction can be verified."""
    meta = runs.load_meta(run)
    if not meta or not meta.get("agent_id"):
        return None
    parent = runs.load_meta(meta["parent"]) if meta.get("parent") else None
    chain = [_read_text(p) for p in meta.get("base_chain") or []]
    own_snapshot = ""
    if chain and parent is not None and len(chain) > len(parent.get("base_chain") or []):
        own_snapshot = chain[-1]
    info = {
        "agent_id": meta["agent_id"],
        "run": run,
        "root_id": meta.get("root_id") or meta["agent_id"],
        "depth": meta.get("depth", 0),
        "parent_id": (parent or {}).get("agent_id"),
        "parent_run": meta.get("parent"),
        "parent_session": (parent or {}).get("session"),
        "role": meta.get("role"),
        "budget": meta.get("budget"),
        "caps": meta.get("caps"),
        "base_sha": meta.get("base_sha"),
        "base_patch": own_snapshot,
        "base_chain_sha256": [_sha(t) for t in chain],
    }
    plan_text = _read_text(meta.get("agents_plan"))
    if plan_text:
        info["plan_sha256"] = _sha(plan_text)
        if parent is None:
            info["plan"] = plan_text  # the root carries the plan itself
    return info


def _session_start(path: Path) -> dict | None:
    """The (live, not replayed) session_start event of a session log."""
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return None
    for line in lines:
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "session_start" and not e.get("replayed"):
            return e
    return None


def chain_from_logs(agent: dict, source: Path) -> tuple[str, list[str]]:
    """Rebuild a sub-agent's starting point from session logs alone: the
    root's base_sha plus each ancestor's own base_patch, root-side first,
    found by following parent_session links up the tree. Every link is checked
    against the recorded base_chain_sha256, so a missing or altered parent log
    is an error rather than a silently different tree. Raises ValueError."""
    patches: list[str] = []
    node, where = agent, Path(source)
    seen: set[str] = set()
    while node is not None:
        if node.get("base_patch"):
            patches.insert(0, node["base_patch"])
        if not node.get("parent_id"):
            break
        parent_log = node.get("parent_session")
        if not parent_log or parent_log in seen:
            raise ValueError(f"{where.name}: no usable parent_session link")
        seen.add(parent_log)
        start = _session_start(Path(parent_log))
        if start is None or not start.get("agent"):
            raise ValueError(f"parent session log {parent_log} (needed by "
                             f"{where.name}) is missing or has no agent record")
        node, where = start["agent"], Path(parent_log)
    expected = agent.get("base_chain_sha256") or []
    got = [_sha(t) for t in patches]
    if got != expected:
        raise ValueError(
            "the base chain rebuilt from the parent logs does not match what "
            f"this session recorded ({len(got)} vs {len(expected)} patches, or "
            "different contents)")
    return agent.get("base_sha") or "", patches


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 3] + "..."


def _files_line(patch_text: str) -> str:
    stats = patch_file_stats(patch_text)
    if not stats:
        return "(no file changes)"
    return ", ".join(
        f"{path} (binary)" if c is None else f"{path} (+{c[0]}/-{c[1]})"
        for path, c in stats.items()
    )


def _read_text(path: str | Path | None) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Budgets and reviews (the project agent plan -- see agent_config.py)
#
# Accounting: every agent's own usage is what its runner publishes in
# stats.json (RunnerView). tokens and cost are SUBTREE quantities -- own plus
# every descendant's -- so a budget bounds everything below it; time is the
# agent's own wall clock since registration.
#
# The review protocol is two files in the reviewed agent's run dir, each with
# ONE writer, so no meta.json write ever races another process:
#   review.json    written by the agent itself: it is out of budget
#                  (kind "exhausted", blocking -- it waits) or asks for more
#                  while still working (kind "request", non-blocking).
#   decision.json  written by its reviewer: the parent's review_agent tool, or
#                  the human's `gerbil review` at the root.
# The agent consumes both and applies the decision to its own meta `budget`,
# so after spawn only the agent itself ever writes its limits.
# ---------------------------------------------------------------------------

REVIEW_POLL_SECONDS = 2.0


def run_usage(name: str) -> tuple[int, float | None]:
    """(tokens, estimated cost) one run has spent so far, from its stats.json.
    Cost is None when its model's pricing is unknown."""
    from .pricing import estimate_cost

    stats = (runs.load_stats_doc(name) or {}).get("stats") or {}
    tokens, cost = 0, 0.0
    for key, model in (("chain_outer", stats.get("model")),
                       ("chain_inner", stats.get("small_model"))):
        u = stats.get(key) or {}
        t = (u.get("input_tokens", 0) + u.get("cache_read_tokens", 0)
             + u.get("cache_write_tokens", 0) + u.get("output_tokens", 0))
        tokens += t
        if t:
            c = estimate_cost(model or "", u.get("input_tokens", 0),
                              u.get("output_tokens", 0),
                              u.get("cache_read_tokens", 0),
                              u.get("cache_write_tokens", 0))
            cost = None if (c is None or cost is None) else cost + c
    return tokens, cost


def subtree_usage(name: str) -> dict:
    """{tokens, cost, seconds} for `name` and all its descendants (cost None
    if any member's pricing is unknown); seconds is `name`'s own age."""
    tokens, cost = 0, 0.0
    stack, seen = [name], set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        t, c = run_usage(n)
        tokens += t
        cost = None if (c is None or cost is None) else cost + c
        stack.extend(r["name"] for r in runs.children(n))
    started = (runs.load_meta(name) or {}).get("started_at") or time.time()
    return {"tokens": tokens, "cost": cost, "seconds": time.time() - started}


def exceeded(budget: "Budget", usage: dict) -> list[str]:
    """The dimensions of `budget` that `usage` has reached."""
    out = []
    if budget.tokens is not None and usage["tokens"] >= budget.tokens:
        out.append("tokens")
    if budget.seconds is not None and usage["seconds"] >= budget.seconds:
        out.append("time")
    if budget.cost is not None and usage["cost"] is not None and usage["cost"] >= budget.cost:
        out.append("cost")
    return out


def describe_usage(budget: "Budget", usage: dict) -> str:
    from .agent_config import format_duration

    parts = [f"{usage['tokens']:,}"
             + (f"/{budget.tokens:,}" if budget.tokens is not None else "") + " tokens",
             format_duration(usage["seconds"])
             + (f"/{format_duration(budget.seconds)}" if budget.seconds is not None else "")]
    if usage["cost"] is not None:
        parts.append(f"${usage['cost']:.2f}"
                     + (f"/${budget.cost:.2f}" if budget.cost is not None else ""))
    return ", ".join(parts)


def load_review(name: str) -> dict | None:
    return runs._read_json(runs.run_dir(name) / "review.json")


def write_decision(name: str, review_id: str, action: str, *,
                   grant: dict | None = None, note: str = "", by: str = "") -> None:
    """The reviewer's half of the protocol (see above). action is one of
    extend / finish / deny; grant holds the added tokens/seconds/cost."""
    runs._write_json(runs.run_dir(name) / "decision.json", {
        "id": review_id, "action": action, "grant": grant or {},
        "note": note, "by": by, "at": time.time(),
    })


def headroom(name: str) -> dict:
    """How much more `name`'s subtree may spend before its own budget is
    exhausted: {tokens, cost, deadline} with None meaning unlimited."""
    meta = runs.load_meta(name) or {}
    from .agent_config import Budget as _B
    budget = _B.from_dict(meta.get("budget"))
    usage = subtree_usage(name)
    return {
        "tokens": None if budget.tokens is None else budget.tokens - usage["tokens"],
        "cost": (None if budget.cost is None else
                 budget.cost - (usage["cost"] or 0.0)),
        "deadline": (None if budget.seconds is None else
                     (meta.get("started_at") or time.time()) + budget.seconds),
    }


def _grant_args(tokens=None, minutes=None, cost=None) -> dict:
    grant = {}
    if tokens is not None:
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError("tokens must be a positive integer")
        grant["tokens"] = tokens
    if minutes is not None:
        if not isinstance(minutes, (int, float)) or isinstance(minutes, bool) or minutes <= 0:
            raise ValueError("minutes must be positive")
        grant["seconds"] = float(minutes) * 60
    if cost is not None:
        if not isinstance(cost, (int, float)) or isinstance(cost, bool) or cost <= 0:
            raise ValueError("cost must be positive")
        grant["cost"] = float(cost)
    return grant


def _describe_grant(grant: dict) -> str:
    from .agent_config import format_duration
    parts = []
    if grant.get("tokens"):
        parts.append(f"+{grant['tokens']:,} tokens")
    if grant.get("seconds"):
        parts.append(f"+{format_duration(grant['seconds'])}")
    if grant.get("cost"):
        parts.append(f"+${grant['cost']:.2f}")
    return ", ".join(parts) or "nothing"


REQUEST_BUDGET_TOOL = {
    "name": "request_budget",
    "description": (
        "Ask whoever supervises you (your parent agent, or the human at the top) "
        "for more budget, BEFORE you run out -- e.g. when you need to grant a "
        "sub-agent an extension larger than your own remaining headroom. You keep "
        "working; the answer arrives as a note. Give a concrete reason."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why the extra budget is worth it: progress so far and what it buys."},
            "tokens": {"type": "integer", "description": "Extra tokens requested."},
            "minutes": {"type": "number", "description": "Extra wall-clock minutes requested."},
            "cost": {"type": "number", "description": "Extra estimated USD requested."},
        },
        "required": ["reason"],
    },
}


class BudgetGuard:
    """Enforces one agent's budget from inside its own agent loop.

    agent.run_session calls before_turn() ahead of every model turn. Within
    budget it only delivers news (a decision on an earlier request_budget).
    Exhausted, it writes a blocking review and WAITS -- the process idles,
    its container stays up, the conversation is intact -- until its reviewer
    decides: extend (limits raised, the loop continues), finish (the loop ends
    and the session wraps up with its commit-message turn, so the work lands
    as a patch), or kill (the parent cancels the run outright; the SIGTERM
    unwinds this wait like anything else). A root agent with
    on_root_exhausted = "finish" skips the wait and wraps up directly.
    """

    def __init__(self, name: str, *, is_root: bool, root_policy: str = "ask",
                 parent: str | None = None,
                 session_getter: Callable[[], object] | None = None):
        # Returns the in-flight Session (or None), for the budget_* events
        # this agent's own log records; see _log.
        self.session_getter = session_getter
        self.name = name
        self.is_root = is_root
        self.root_policy = root_policy
        self.parent = parent

    # -- state ------------------------------------------------------------

    @property
    def budget(self):
        from .agent_config import Budget
        return Budget.from_dict((runs.load_meta(self.name) or {}).get("budget"))

    def usage(self) -> dict:
        return subtree_usage(self.name)

    def remaining_seconds(self) -> float | None:
        b = self.budget
        if b.seconds is None:
            return None
        return max(0.0, b.seconds - self.usage()["seconds"])

    def _reviewer(self) -> str:
        return (f"your parent agent {self.parent}" if self.parent
                else f"the human (`gerbil review {self.name}`)")

    def _progress(self) -> str:
        meta = runs.load_meta(self.name) or {}
        last = ""
        for line in reversed(_read_text(meta.get("session")).splitlines()):
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("event") == "turn" and e.get("role") == "assistant" and e.get("content"):
                last = e["content"]
                break
        wip = _read_text(meta.get("wip_patch"))
        return (f"files touched: {_files_line(wip) if wip else '(none yet)'}\n"
                f"latest message: {_clip(last, 800) or '(none)'}")

    # -- the protocol ---------------------------------------------------------

    def _apply(self, decision: dict) -> None:
        from .agent_config import Budget
        b = self.budget
        g = decision.get("grant") or {}
        if g.get("tokens"):
            b.tokens = (b.tokens or 0) + int(g["tokens"])
        if g.get("seconds"):
            b.seconds = (b.seconds or 0) + float(g["seconds"])
        if g.get("cost"):
            b.cost = (b.cost or 0) + float(g["cost"])
        runs.save_meta(self.name, budget=b.to_dict())

    def _consume(self) -> dict | None:
        """Take a decision answering our current review, if one has arrived."""
        rd = runs.run_dir(self.name)
        review = load_review(self.name)
        decision = runs._read_json(rd / "decision.json")
        if not review or not decision or decision.get("id") != review.get("id"):
            return None
        for f in ("decision.json", "review.json"):
            (rd / f).unlink(missing_ok=True)
        runs.save_meta(self.name, review_pending=None)
        # The decision is recorded on THIS side, which sees every one --
        # including a human's `gerbil review`, which has no log of its own.
        _log(self.session_getter, "budget_decision", review_id=review.get("id"),
             kind=review.get("kind"), action=decision.get("action"),
             grant=decision.get("grant") or {}, note=decision.get("note", ""),
             by=decision.get("by", ""))
        return decision

    def _write_review(self, kind: str, *, blocking: bool, reason: str,
                      requested: dict | None) -> str:
        review_id = f"{kind}-{time.time():.3f}"
        usage, budget = self.usage(), self.budget.to_dict()
        runs._write_json(runs.run_dir(self.name) / "review.json", {
            "id": review_id, "kind": kind, "blocking": blocking,
            "reason": reason, "requested": requested or {},
            "usage": usage, "budget": budget,
            "progress": self._progress(), "created_at": time.time(),
        })
        runs.save_meta(self.name, review_pending=kind)
        _log(self.session_getter, "budget_review", review_id=review_id, kind=kind,
             blocking=blocking, reason=reason, requested=requested or {},
             usage=usage, budget=budget)
        return review_id

    def request(self, reason: str, grant: dict) -> str:
        """A non-blocking request_budget. A blocking review already pending
        supersedes it (we are not working then anyway)."""
        current = load_review(self.name)
        if current and current.get("blocking"):
            return "a review is already pending."
        self._write_review("request", blocking=False, reason=reason, requested=grant)
        return (f"requested {_describe_grant(grant)} from {self._reviewer()}; "
                "keep working -- the answer arrives as a note.")

    def before_turn(self, view=None) -> tuple[str | None, str | None]:
        """(action, note): action None = carry on, "finish" = wrap up now; the
        note, if any, is delivered into the conversation by the caller."""
        notes = []
        # News on an earlier non-blocking request?
        current = load_review(self.name)
        if current and not current.get("blocking"):
            d = self._consume()
            if d:
                if d.get("action") == "extend":
                    self._apply(d)
                    notes.append(f"Your budget request was GRANTED "
                                 f"({_describe_grant(d.get('grant') or {})})."
                                 + (f" Note: {d['note']}" if d.get("note") else ""))
                else:
                    notes.append("Your budget request was DENIED."
                                 + (f" Note: {d['note']}" if d.get("note") else "")
                                 + " Plan to finish within your current budget.")

        while True:
            budget, usage = self.budget, self.usage()
            over = exceeded(budget, usage)
            if not over:
                return None, ("\n\n".join(notes) or None)
            status = describe_usage(budget, usage)
            if self.is_root and self.root_policy == "finish":
                if view is not None:
                    view.notice(f"[budget exhausted ({status}): wrapping up]")
                return "finish", None
            review = load_review(self.name)
            if not (review and review.get("blocking")):
                self._write_review(
                    "exhausted", blocking=True,
                    reason=f"budget exhausted ({', '.join(over)}): {status}",
                    requested=(review or {}).get("requested"))
            if view is not None:
                view.notice(f"[budget exhausted ({status}): waiting for "
                            f"{self._reviewer()} to extend, finish, or kill]")
            decision = None
            while decision is None:
                time.sleep(REVIEW_POLL_SECONDS)
                decision = self._consume()
            action = decision.get("action")
            if action != "extend":
                if view is not None:
                    view.notice(f"[review: {action} -- wrapping up]")
                return "finish", None
            self._apply(decision)
            if view is not None:
                view.notice(f"[review: extended ({_describe_grant(decision.get('grant') or {})})]")
            notes.append(
                "You ran out of budget, and your reviewer EXTENDED it ("
                f"{_describe_grant(decision.get('grant') or {})})."
                + (f" Note from the reviewer: {decision['note']}" if decision.get("note") else "")
                + " Use the extension economically and aim to finish within it.")
            # Loop: re-check, in case the grant does not cover the overrun.

    # -- Toolset protocol (request_budget) -----------------------------------

    def schemas(self) -> list[dict]:
        return [REQUEST_BUDGET_TOOL]

    def tool_names(self) -> set[str]:
        return {"request_budget"}

    def dispatch(self, name: str, args: dict) -> ToolResult:
        try:
            args = dict(args or {})
            reason = args.pop("reason", "")
            if not isinstance(reason, str) or not reason.strip():
                return ToolResult("error: give a reason.", is_error=True)
            grant = _grant_args(**args)
            if not grant:
                return ToolResult("error: request at least one of tokens, minutes, cost.",
                                  is_error=True)
            return ToolResult(self.request(reason, grant))
        except (TypeError, ValueError) as e:
            return ToolResult(f"error: {e}", is_error=True)


# ---------------------------------------------------------------------------
# The controller: one per agent that may spawn
# ---------------------------------------------------------------------------

class AgentController:
    """The supervisor tools for one agent (the run named `name`), bound to that
    agent's own sandbox. Plugged into tools.Toolset, which routes the names in
    tool_names() here; dispatch() never raises, like every tool handler."""

    def __init__(
        self,
        *,
        name: str,
        sandbox,
        base_commit: Callable[[], str],
        project_dir: Path,
        model: str,
        child_argv: list[str],
        caps: Caps,
        plan=None,
        role=None,
    ):
        # With a project agent plan (agent_config.AgentPlan), `role` is this
        # agent's Role (None for an ad-hoc agent, which never spawns), and
        # spawning is governed by the plan rather than by caps' depth/children.
        self.plan = plan
        self.role = role
        self._notified: set[str] = set()
        # Returns this agent's in-flight Session, for the agent_* events its
        # log records (set by cli.py; the controller predates the session).
        self.session_getter: Callable[[], object] | None = None
        self._outcomes_logged: set[str] = set()
        self.name = name
        self.sandbox = sandbox
        # The sandbox commit this agent's current session started from; the
        # snapshot handed to a new child is taken relative to it. A callable,
        # because the controller is built before the session's base is known.
        self._base_commit = base_commit
        self.project_dir = Path(project_dir)
        self.model = model
        # Flags every child inherits verbatim (--image, --skip-cache, --no-mcp).
        self.child_argv = list(child_argv)
        self.caps = caps
        self._last_nudge: tuple | None = None

    # -- registry views ---------------------------------------------------

    @property
    def meta(self) -> dict:
        return runs.load_meta(self.name) or {}

    @property
    def root(self) -> str:
        return self.meta.get("root") or self.name

    def can_spawn(self) -> bool:
        if self.plan is not None:
            return self.role is not None and self.role.can_spawn()
        return self.caps.depth_left > 0

    def _role_slots(self) -> dict[str | None, tuple[int, int]]:
        """Plan mode: {role (None = ad hoc): (spawned so far, allowed)}."""
        counts: dict[str | None, int] = {}
        for r in runs.children(self.name):
            counts[r.get("role")] = counts.get(r.get("role"), 0) + 1
        slots = {name: (counts.get(name, 0), n) for name, n in self.role.spawn.items()}
        if self.role.adhoc:
            slots[None] = (counts.get(None, 0), self.role.adhoc)
        return slots

    def _visible(self, target: str) -> bool:
        return isinstance(target, str) and runs.is_descendant(self.name, target)

    @staticmethod
    def _unknown(target) -> ToolResult:
        # Deliberately identical for "no such run" and "not yours": an agent
        # learns nothing about runs outside its own subtree.
        return ToolResult(f"error: no sub-agent named {target!r} among your sub-agents.",
                          is_error=True)

    # -- Toolset protocol ---------------------------------------------------

    def schemas(self) -> list[dict]:
        tools = list(SUPERVISOR_TOOLS)
        if not self.can_spawn():
            tools = [t for t in tools if t["name"] != "spawn_agent"]
        elif self.plan is not None:
            tools = [self._plan_spawn_schema() if t["name"] == "spawn_agent" else t
                     for t in tools]
        if self.plan is not None:
            tools.append(REVIEW_AGENT_TOOL)
        return tools

    def _plan_spawn_schema(self) -> dict:
        """spawn_agent as the plan shapes it: a `role` chosen from the roles
        this agent may spawn (omitted = an ad-hoc agent, when allowed), and no
        model/turn/spawn knobs -- the role fixes those."""
        lines = []
        for name, n in self.role.spawn.items():
            r = self.plan.roles[name]
            lines.append(f"  - {name} (up to {n}; budget {r.budget.describe()})"
                         + (f": {r.description}" if r.description else ""))
        if self.role.adhoc:
            lines.append(f"  - (no role) an ad-hoc helper (up to {self.role.adhoc}; "
                         f"budget {self.role.adhoc_budget.describe()}; cannot spawn)")
        props = {
            "prompt": {"type": "string", "description": "Instructions for this particular agent, appended to its role's standing task: exactly what to do, the relevant context, and what 'done' means."},
        }
        if self.role.spawn:
            props["role"] = {"type": "string", "enum": list(self.role.spawn),
                             "description": "The role to spawn"
                             + (" (omit for an ad-hoc helper)" if self.role.adhoc else "") + "."}
        return {
            "name": "spawn_agent",
            "description": SPAWN_AGENT_TOOL["description"]
            + " The project's agent plan allows you to spawn:\n" + "\n".join(lines),
            "input_schema": {"type": "object", "properties": props,
                             "required": ["prompt"]},
        }

    def tool_names(self) -> set[str]:
        return {t["name"] for t in self.schemas()}

    def dispatch(self, name: str, args: dict) -> ToolResult:
        handlers = {
            "spawn_agent": self.spawn,
            "list_agents": self.list_tree,
            "peek_agent": self.peek,
            "wait_agents": self.wait,
            "merge_agent": self.merge,
            "kill_agent": self.kill,
            "review_agent": self.review,
        }
        if name not in self.tool_names():
            return ToolResult(f"error: tool {name} is not available.", is_error=True)
        try:
            return handlers[name](**(args or {}))
        except TypeError as e:
            return ToolResult(f"error: bad arguments to {name}: {e}", is_error=True)
        except Exception as e:  # a tool must never take the session down
            return ToolResult(f"error: {name} failed: {type(e).__name__}: {e}",
                              is_error=True)

    # -- tools ----------------------------------------------------------------

    def spawn(self, prompt: str, model: str | None = None,
              max_turns: int | None = None, can_spawn: bool = False,
              role: str | None = None) -> ToolResult:
        if not self.can_spawn():
            return ToolResult("error: you are not allowed to spawn sub-agents.",
                              is_error=True)
        if not isinstance(prompt, str) or not prompt.strip():
            return ToolResult("error: prompt must be a non-empty task description.",
                              is_error=True)
        if max_turns is not None and (not isinstance(max_turns, int) or max_turns < 1):
            return ToolResult("error: max_turns must be a positive integer.",
                              is_error=True)

        child_role = None
        child_budget = None
        if self.plan is not None:
            # The plan, not the model, decides what may be spawned and with
            # what model and budget.
            slots = self._role_slots()
            if role is not None and role not in self.role.spawn:
                return ToolResult(
                    f"error: the agent plan does not let a {self.role.name} spawn "
                    f"role {role!r}; allowed: "
                    + (", ".join(self.role.spawn) or "(no roles)")
                    + (", or omit role for an ad-hoc helper" if self.role.adhoc else "")
                    + ".", is_error=True)
            if role is None and not self.role.adhoc:
                return ToolResult("error: choose a role ("
                                  + ", ".join(self.role.spawn) + ").", is_error=True)
            used, allowed = slots[role]
            if used >= allowed:
                return ToolResult(
                    f"error: you have already spawned {used} "
                    + (f"{role} agents" if role else "ad-hoc helpers")
                    + f"; the plan allows {allowed}.", is_error=True)
            if role is not None:
                child_role = self.plan.roles[role]
                child_budget = child_role.budget
                model = child_role.model or self.model
                if child_role.task:
                    prompt = child_role.task.rstrip() + "\n\n" + prompt
            else:
                child_budget = self.role.adhoc_budget
                model = model or self.model
            max_agents = self.plan.max_live_agents
        else:
            spawned = runs.children(self.name)
            if len(spawned) >= self.caps.max_children:
                return ToolResult(
                    f"error: you have already spawned {len(spawned)} sub-agents, "
                    f"the limit is {self.caps.max_children}.", is_error=True)
            max_agents = self.caps.max_agents
        live = [r for r in tree_members(self.root)
                if r["state"] in LIVE_STATES and r["name"] != self.root]
        if len(live) >= max_agents:
            return ToolResult(
                f"error: {len(live)} sub-agents are already running across the "
                f"whole agent tree (limit {max_agents}); wait for or "
                "kill one first.", is_error=True)

        me = self.meta
        model = model or self.model
        if self.plan is not None:
            child_caps = Caps(depth_left=0, max_children=0,
                              max_agents=self.plan.max_live_agents,
                              max_turns=self.caps.max_turns)
        else:
            child_caps = self.caps.for_child(bool(can_spawn), max_turns)
        child = runs.new_run_name()
        argv = ["run", "--plain", "--at", str(self.project_dir),
                "--prompt", "", "--model", model, "--_child-of", self.name,
                *self.child_argv]
        rd = runs.create_run(child, project_dir=self.project_dir, model=model,
                             theme=me.get("theme"), command=argv)
        prompt_path = rd / "prompt.md"
        prompt_path.write_text(prompt)
        argv[argv.index("--prompt") + 1] = str(prompt_path)
        if child_caps.max_turns:
            argv += ["--max-turns", str(child_caps.max_turns)]

        # The child's starting tree: whatever rebuilt ours, plus a snapshot of
        # everything we have done since (committed or not).
        chain = list(me.get("base_chain") or [])
        snapshot = self.sandbox.wip_patch(self._base_commit())
        if snapshot:
            base_patch = rd / "base.patch"
            base_patch.write_text(snapshot)
            chain.append(str(base_patch))

        agent_id = uuid.uuid4().hex
        runs.save_meta(
            child, command=argv, parent=self.name, root=self.root,
            agent_id=agent_id, root_id=me.get("root_id") or me.get("agent_id"),
            depth=int(me.get("depth") or 0) + 1, caps=child_caps.to_dict(),
            base_sha=me.get("base_sha"), base_chain=chain,
            task=_clip(prompt, 120),
            # Plan mode: the child's role, its budget (the child owns these
            # limits from here on -- see the review protocol), and the frozen
            # plan every agent in the tree reads.
            role=child_role.name if child_role else None,
            budget=child_budget.to_dict() if child_budget else None,
            agents_plan=me.get("agents_plan"),
        )
        runs.spawn_runner(child, argv, self.project_dir)
        _log(self.session_getter, "agent_spawn", agent_id=agent_id, run=child,
             role=child_role.name if child_role else None, model=model,
             budget=child_budget.to_dict() if child_budget else None,
             max_turns=child_caps.max_turns, prompt=prompt,
             base_chain=len(chain),
             base_patch_sha256=_sha(snapshot) if snapshot else None)
        return ToolResult(
            f"spawned sub-agent {child} ({model}"
            + (f", role {child_role.name}" if child_role else
               ", ad-hoc" if self.plan is not None else "")
            + (f", budget {child_budget.describe()}" if child_budget else "")
            + (f", max {child_caps.max_turns} turns" if child_caps.max_turns else "")
            + (", may spawn" if (child_role.can_spawn() if child_role
                                 else child_caps.depth_left) else "")
            + "). It starts from a snapshot of your current tree"
            + (" (including your uncommitted work)" if snapshot else "")
            + ". Use peek_agent / wait_agents to follow it and merge_agent to "
            "take its work."
        )

    def list_tree(self) -> ToolResult:
        rows: list[str] = []

        def walk(parent: str, indent: int) -> None:
            for r in runs.children(parent):
                stats = (r.get("stats") or {}).get("stats") or {}
                flags = []
                if r.get("merged"):
                    flags.append("merged")
                elif r["state"] == "complete":
                    res = load_result(r["name"])
                    flags.append("has patch" if res and res.get("patch") else "no changes")
                if r.get("review_pending") and r["state"] in LIVE_STATES:
                    flags.append("AWAITING REVIEW" if r["review_pending"] == "exhausted"
                                 else "requests budget")
                budget = ""
                if r.get("budget"):
                    from .agent_config import Budget
                    budget = (", " + describe_usage(Budget.from_dict(r["budget"]),
                                                    subtree_usage(r["name"])))
                rows.append(
                    "  " * indent
                    + f"- {r['name']}"
                    + (f" ({r['role']})" if r.get("role") else "")
                    + f": {r['state']}"
                    + (f" [{', '.join(flags)}]" if flags else "")
                    + f", {r.get('model')}, {stats.get('turns') or 0} turns"
                    + budget
                    + f" -- {r.get('task') or ''}"
                )
                walk(r["name"], indent + 1)

        walk(self.name, 0)
        if not rows:
            return ToolResult("You have no sub-agents.")
        return ToolResult("\n".join(rows))

    def peek(self, name: str, events: int = 15) -> ToolResult:
        if not self._visible(name):
            return self._unknown(name)
        meta = runs.load_meta(name) or {}
        state = runs.classify(meta)
        events = max(1, min(int(events or 15), 60))
        out = [f"{name}: {state}"]
        wip = _read_text(meta.get("wip_patch"))
        if wip:
            out.append("work so far: " + _files_line(wip))
        log = meta.get("session")
        lines = _read_text(log).splitlines()[-events:] if log else []
        if not lines:
            out.append("(no session events yet -- it may still be booting its sandbox)")
        for line in lines:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = e.get("event")
            if kind == "turn":
                if e.get("content"):
                    out.append(f"[{e.get('role')}] {_clip(e['content'], 400)}")
            elif kind == "tool_call":
                out.append(f"[call] {e.get('name')}({_clip(json.dumps(e.get('args')), 200)})")
            elif kind == "tool_result":
                out.append(f"[result] {e.get('name')}: {_clip(e.get('result', ''), 300)}")
            elif kind in ("warning", "error"):
                out.append(f"[{kind}] {_clip(e.get('message', ''), 300)}")
        return ToolResult("\n".join(out))

    def _outcome(self, name: str) -> str:
        meta = runs.load_meta(name) or {}
        state = runs.classify(meta)
        res = load_result(name)
        if state != "complete" or res is None:
            return f"{name}: {state}" + (
                "" if state in LIVE_STATES else " (no result; nothing to merge)")
        patch = _read_text(res.get("patch"))
        msg = res.get("commit_message") or res.get("final_text") or ""
        return (
            f"{name}: complete" + (" [merged]" if meta.get("merged") else "")
            + f"\n  files: {_files_line(patch) if patch else '(no changes)'}"
            + f"\n  report: {_clip(msg, 1500)}"
        )

    def wait(self, names: list[str], timeout_seconds: int = WAIT_DEFAULT_SECONDS) -> ToolResult:
        if isinstance(names, str):
            names = [names]
        if not names:
            return ToolResult("error: names must list at least one sub-agent.",
                              is_error=True)
        for n in names:
            if not self._visible(n):
                return self._unknown(n)
        timeout = max(1, min(int(timeout_seconds or WAIT_DEFAULT_SECONDS), WAIT_MAX_SECONDS))
        # Never sleep through our own time budget: the guard can only stop
        # us at a turn boundary, and this call is in the middle of a turn.
        remaining = self.budget_guard.remaining_seconds() if self.budget_guard else None
        if remaining is not None:
            timeout = max(1, min(timeout, int(remaining) + 1))
        deadline = time.monotonic() + timeout
        while True:
            pending = [n for n in names
                       if runs.classify(runs.load_meta(n) or {}) in LIVE_STATES]
            # A child waiting on OUR review cannot finish until we answer it.
            reviews = self._pending_reviews()
            if not pending or reviews or time.monotonic() >= deadline:
                break
            time.sleep(WAIT_POLL_SECONDS)
        if reviews:
            head = ("interrupted: sub-agents need your review first -- "
                    + "; ".join(self._review_line(n, rv) for n, rv in reviews.items()))
        elif not pending:
            head = "all finished."
        else:
            head = f"timed out after {timeout}s; still running: {', '.join(pending)}."
        for n in names:
            self._log_outcome(n)
        return ToolResult(head + "\n" + "\n".join(self._outcome(n) for n in names))

    def _log_outcome(self, name: str) -> None:
        """Once a child has ended, record how -- and where its own session log
        and patch are -- in OUR log, the first time we observe it."""
        meta = runs.load_meta(name) or {}
        state = runs.classify(meta)
        if state in LIVE_STATES or name in self._outcomes_logged:
            return
        self._outcomes_logged.add(name)
        res = load_result(name) or {}
        patch = _read_text(res.get("patch"))
        _log(self.session_getter, "agent_outcome", agent_id=meta.get("agent_id"),
             run=name, state=state, session=meta.get("session"),
             commit_message=res.get("commit_message"),
             patch_sha256=_sha(patch) if patch else None,
             files=list(patch_file_stats(patch)) if patch else [])

    # -- reviews (plan mode) ------------------------------------------------------

    budget_guard: "BudgetGuard | None" = None  # set by cli.py in plan mode

    def _pending_reviews(self) -> dict[str, dict]:
        """Open reviews from live direct children: {name: review.json}."""
        out = {}
        for r in runs.children(self.name):
            if r["state"] not in LIVE_STATES or not r.get("review_pending"):
                continue
            rv = load_review(r["name"])
            if rv:
                out[r["name"]] = rv
        return out

    @staticmethod
    def _review_line(name: str, rv: dict) -> str:
        if rv.get("kind") == "exhausted":
            return f"{name} is OUT OF BUDGET and paused ({rv.get('reason')})"
        return (f"{name} requests {_describe_grant(rv.get('requested') or {})}: "
                f"{_clip(rv.get('reason', ''), 300)}")

    def turn_notes(self) -> str | None:
        """News for the model at a turn boundary: reviews it has not been told
        about yet. Each review is announced once (by its id)."""
        fresh = {n: rv for n, rv in self._pending_reviews().items()
                 if rv.get("id") not in self._notified}
        if not fresh:
            return None
        self._notified.update(rv.get("id") for rv in fresh.values())
        lines = []
        for n, rv in fresh.items():
            lines.append("- " + self._review_line(n, rv)
                         + f"\n  {rv.get('progress', '').replace(chr(10), chr(10) + '  ')}")
        return ("Sub-agent review needed (call review_agent: extend, finish, or "
                "kill -- an extension comes out of your own remaining budget):\n"
                + "\n".join(lines))

    def review(self, name: str, action: str, tokens: int | None = None,
               minutes: float | None = None, cost: float | None = None,
               note: str = "") -> ToolResult:
        """Answer a direct child's review. extend grants budget out of OUR
        headroom (asking our own parent first is request_budget's job); finish
        lets it wrap up into a patch; kill cancels it; deny answers a
        non-blocking request without granting anything."""
        if not self._visible(name):
            return self._unknown(name)
        meta = runs.load_meta(name) or {}
        if meta.get("parent") != self.name:
            return ToolResult(f"error: {name} is not your direct sub-agent; its "
                              "own parent reviews it.", is_error=True)
        if action == "kill":
            return self.kill(name)
        rv = load_review(name)
        if not rv or runs.classify(meta) not in LIVE_STATES:
            return ToolResult(f"error: {name} has no open review.", is_error=True)
        if action in ("finish", "deny"):
            if action == "deny" and rv.get("blocking"):
                action = "finish"  # an exhausted agent cannot just keep going
            write_decision(name, rv["id"], action, note=note or "", by=self.name)
            _log(self.session_getter, "agent_review", agent_id=meta.get("agent_id"),
                 run=name, review_id=rv["id"], action=action, grant={}, note=note or "")
            return ToolResult(f"{name}: {action}"
                              + (" -- it will wrap up and produce its patch."
                                 if action == "finish" else "."))
        if action != "extend":
            return ToolResult("error: action must be extend, finish, deny, or kill.",
                              is_error=True)
        try:
            grant = _grant_args(tokens, minutes, cost)
        except ValueError as e:
            return ToolResult(f"error: {e}", is_error=True)
        if not grant:
            grant = dict(rv.get("requested") or {})
            if not grant:
                return ToolResult("error: say how much to extend (tokens, minutes, "
                                  "cost).", is_error=True)
        # Extensions come out of our own headroom.
        room = headroom(self.name)
        short = []
        if grant.get("tokens") and room["tokens"] is not None and grant["tokens"] > room["tokens"]:
            short.append(f"tokens (you have {max(0, room['tokens']):,} left)")
        if grant.get("cost") and room["cost"] is not None and grant["cost"] > room["cost"]:
            short.append(f"cost (you have ${max(0.0, room['cost']):.2f} left)")
        if grant.get("seconds") and room["deadline"] is not None:
            from .agent_config import Budget as _B
            cb = _B.from_dict(meta.get("budget"))
            child_deadline = ((meta.get("started_at") or time.time())
                              + (cb.seconds or 0) + grant["seconds"])
            if child_deadline > room["deadline"]:
                short.append("time (it would outlast your own deadline)")
        if short:
            return ToolResult(
                f"error: you cannot afford that extension: {', '.join(short)}. "
                "Grant less, let it finish, kill it, or call request_budget to "
                "ask your own supervisor for more first.", is_error=True)
        write_decision(name, rv["id"], "extend", grant=grant, note=note or "",
                       by=self.name)
        _log(self.session_getter, "agent_review", agent_id=meta.get("agent_id"),
             run=name, review_id=rv["id"], action="extend", grant=grant,
             note=note or "")
        return ToolResult(f"{name}: extended by {_describe_grant(grant)}.")

    def merge(self, name: str) -> ToolResult:
        meta = runs.load_meta(name) or {}
        if not self._visible(name):
            return self._unknown(name)
        if meta.get("parent") != self.name:
            return ToolResult(
                f"error: {name} is not your direct sub-agent; its own parent "
                "merges it.", is_error=True)
        if meta.get("merged"):
            return ToolResult(f"error: {name} is already merged.", is_error=True)
        state = runs.classify(meta)
        res = load_result(name)
        if state != "complete" or res is None:
            return ToolResult(f"error: {name} is {state}; only a complete "
                              "sub-agent can be merged.", is_error=True)
        patch = _read_text(res.get("patch"))
        self._log_outcome(name)
        if not patch.strip():
            runs.save_meta(name, merged=True)
            return ToolResult(f"{name} made no changes; nothing to merge.")
        # git am needs a clean index; commit our own work first. Intermediate
        # commits are fine -- the session is squashed onto its base at the end.
        self.sandbox.commit(f"gerbil: checkpoint before merging sub-agent {name}")
        ok, output = self.sandbox.try_git_am_3way(patch)
        # The merged patch itself goes into OUR log: the sub-agent's edits
        # happened in another sandbox, so this is the only record from which
        # reconstruct-patch can replay them as a step of this session.
        _log(self.session_getter, "agent_merge", agent_id=meta.get("agent_id"),
             run=name, session=meta.get("session"), ok=ok,
             patch_sha256=_sha(patch), patch=patch if ok else "",
             error=None if ok else _clip(output, 2000))
        if not ok:
            return ToolResult(
                f"merge of {name} FAILED (nothing applied; your tree is as it "
                f"was):\n{_clip(output, 3000)}\nYou can resolve this by doing "
                "the work yourself, or spawning a sub-agent from your current "
                "tree to redo it.", is_error=True)
        runs.save_meta(name, merged=True)
        return ToolResult(f"merged {name}: {_files_line(patch)}")

    def kill(self, name: str) -> ToolResult:
        if not self._visible(name):
            return self._unknown(name)
        if runs.cancel_run(name):
            _log(self.session_getter, "agent_kill",
                 agent_id=(runs.load_meta(name) or {}).get("agent_id"), run=name)
            return ToolResult(f"cancelled {name} (and its sub-agents).")
        state = runs.classify(runs.load_meta(name) or {})
        return ToolResult(f"{name} is not running ({state}); nothing to stop.")

    # -- loop hooks -------------------------------------------------------------

    def stop_note(self) -> str | None:
        """Called by the agent loop when the model stops calling tools. If
        direct children are still running or finished-but-unmerged, return a
        note sending the model back (they would be cancelled or dropped once
        this agent ends). Each distinct situation is raised once, so a model
        that deliberately stops anyway is not ping-ponged."""
        reviews = self._pending_reviews()
        if reviews:
            key = ("reviews", tuple(sorted(rv.get("id") for rv in reviews.values())))
            if key != self._last_nudge:
                self._last_nudge = key
                return ("Before you finish: sub-agents are waiting on your "
                        "review -- "
                        + "; ".join(self._review_line(n, rv) for n, rv in reviews.items())
                        + ". Answer with review_agent.")
        live, unmerged = [], []
        for r in runs.children(self.name):
            if r["state"] in LIVE_STATES:
                live.append(r["name"])
            elif r["state"] == "complete" and not r.get("merged"):
                res = load_result(r["name"])
                if res and res.get("patch"):
                    unmerged.append(r["name"])
        key = (tuple(live), tuple(unmerged))
        if not (live or unmerged) or key == self._last_nudge:
            return None
        self._last_nudge = key
        parts = []
        if live:
            parts.append(f"still running: {', '.join(live)} (they will be "
                         "cancelled when you finish)")
        if unmerged:
            parts.append(f"finished but not merged: {', '.join(unmerged)} "
                         "(their work is discarded unless you merge_agent it)")
        return ("Before you finish: you have sub-agents " + "; ".join(parts)
                + ". Use wait_agents / merge_agent / kill_agent as appropriate. "
                "If you have deliberately decided to finish anyway, just stop "
                "again.")
