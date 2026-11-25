"""The project agent plan: `.gerbil/agents.toml`.

A project can declare, in one file, the shape supervisor mode (`--agents`,
see agents.py) is allowed to take: named ROLES, which role the top-level
session plays, which roles each role may spawn and how many of each, and a
BUDGET for every role. Without the file, `--agents` is free-form (the model
spawns whatever it likes within the numeric --max-agent-* caps); with it, a
spawn must name a role the spawning role is allowed, and every agent runs
under its role's budget.

    root = "lead"                       # the role the top-level session plays

    [limits]
    max_live_agents = 6                 # sub-agents running at once, tree-wide
    on_root_exhausted = "ask"           # or "finish": see below

    [roles.lead]
    description = "Plans the proof and integrates the pieces"
    task = "..."                        # optional; the role's standing task
    model = "claude-opus-4-8"           # optional; default: the run's --model
    budget = { tokens = 20_000_000, time = "3h", cost = 60.0 }
    spawn = { prover = 2, reviewer = 1 }   # child role -> how many, max
    adhoc = 1                           # improvised (role-less) sub-agents
    adhoc_budget = { tokens = 1_000_000, time = "20m" }

    [roles.prover]
    task = "Fill exactly the sorry your instructions name; touch nothing else."
    model = "claude-sonnet-4-5"
    budget = { tokens = 4_000_000, time = "1h" }
    spawn = { grinder = 2 }

Budgets (see agents.BudgetGuard for enforcement):

- Any of `tokens` (input + cache + output), `time` (wall clock since the
  agent started: an int of seconds or "90s" / "45m" / "2h" / "1h30m"), and
  `cost` (estimated USD; a model with unknown pricing cannot be cost-capped,
  which is a preflight error rather than a silently ignored limit). Whichever
  set limit is reached first exhausts the budget.
- tokens and cost are SUBTREE budgets: they count the agent's own usage plus
  every descendant's. So a parent's budget really bounds everything below
  it, and children overspending pushes the parent itself into review -- the
  "bubbling up" falls out of the accounting.
- An exhausted agent stops at its next turn boundary and asks for a review;
  its parent decides (extend / finish / kill), and an extension can only be
  granted out of the parent's own remaining headroom -- otherwise the parent
  must escalate to ITS parent (request_budget). At the root the decision is
  the human's (`gerbil review`), unless `on_root_exhausted = "finish"`
  makes the root simply wrap up.

The role graph must be acyclic (a DAG), so the hierarchy's depth is bounded
by the file itself. Validation is strict and all problems are reported at
once: a typo in a plan meant to bound spending must not be silently ignored.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_NAME = "agents.toml"

_DURATION = re.compile(r"^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$")


@dataclass
class Budget:
    tokens: int | None = None
    seconds: float | None = None
    cost: float | None = None

    def is_set(self) -> bool:
        return any(v is not None for v in (self.tokens, self.seconds, self.cost))

    def to_dict(self) -> dict:
        return {"tokens": self.tokens, "seconds": self.seconds, "cost": self.cost}

    @classmethod
    def from_dict(cls, d: dict | None) -> "Budget":
        d = d or {}
        return cls(tokens=d.get("tokens"), seconds=d.get("seconds"),
                   cost=d.get("cost"))

    def describe(self) -> str:
        parts = []
        if self.tokens is not None:
            parts.append(f"{self.tokens:,} tokens")
        if self.seconds is not None:
            parts.append(format_duration(self.seconds))
        if self.cost is not None:
            parts.append(f"${self.cost:.2f}")
        return ", ".join(parts) or "unlimited"


@dataclass
class Role:
    name: str
    description: str = ""
    task: str = ""
    model: str | None = None
    budget: Budget = field(default_factory=Budget)
    spawn: dict[str, int] = field(default_factory=dict)
    adhoc: int = 0
    adhoc_budget: Budget = field(default_factory=Budget)

    def can_spawn(self) -> bool:
        return bool(self.spawn) or self.adhoc > 0


@dataclass
class AgentPlan:
    path: Path
    root: str
    roles: dict[str, Role]
    max_live_agents: int
    on_root_exhausted: str  # "ask" | "finish"

    def role(self, name: str | None) -> Role | None:
        return self.roles.get(name) if name else None

    def describe(self) -> str:
        """The hierarchy as an indented tree, for the preflight banner."""
        lines: list[str] = []

        def walk(name: str, count: int, indent: int) -> None:
            r = self.roles[name]
            mult = f"{count} x " if count > 1 else ""
            lines.append("  " * indent + f"- {mult}{name} [{r.budget.describe()}]"
                         + (f" -- {r.description}" if r.description else ""))
            for child, n in r.spawn.items():
                walk(child, n, indent + 1)
            if r.adhoc:
                lines.append("  " * (indent + 1)
                             + f"- up to {r.adhoc} ad-hoc [{r.adhoc_budget.describe()}]")

        walk(self.root, 1, 0)
        return "\n".join(lines)


def parse_duration(value) -> float:
    """Seconds from an int/float or "90s" / "45m" / "2h" / "1h30m"."""
    if isinstance(value, bool):
        raise ValueError(f"not a duration: {value!r}")
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("duration must be positive")
        return float(value)
    if isinstance(value, str):
        m = _DURATION.match(value.strip())
        if m and any(m.groups()):
            h, mi, s = (int(g) if g else 0 for g in m.groups())
            total = h * 3600 + mi * 60 + s
            if total > 0:
                return float(total)
    raise ValueError(f"not a duration: {value!r} (use seconds or e.g. \"45m\", \"1h30m\")")


def format_duration(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return "".join(f"{v}{u}" for v, u in ((h, "h"), (m, "m"), (s, "s")) if v) or "0s"


def _parse_budget(raw, where: str, problems: list[str]) -> Budget:
    if raw is None:
        return Budget()
    if not isinstance(raw, dict):
        problems.append(f"{where}: must be a table like {{ tokens = 1000000, time = \"30m\" }}")
        return Budget()
    b = Budget()
    for key in raw:
        if key not in ("tokens", "time", "cost"):
            problems.append(f"{where}: unknown key {key!r} (expected tokens, time, cost)")
    if "tokens" in raw:
        t = raw["tokens"]
        if isinstance(t, int) and not isinstance(t, bool) and t > 0:
            b.tokens = t
        else:
            problems.append(f"{where}.tokens: must be a positive integer")
    if "time" in raw:
        try:
            b.seconds = parse_duration(raw["time"])
        except ValueError as e:
            problems.append(f"{where}.time: {e}")
    if "cost" in raw:
        c = raw["cost"]
        if isinstance(c, (int, float)) and not isinstance(c, bool) and c > 0:
            b.cost = float(c)
        else:
            problems.append(f"{where}.cost: must be a positive number (USD)")
    return b


def load(path: Path, *, default_model: str) -> AgentPlan:
    """Parse and validate an agent plan. Raises ValueError listing every
    problem found."""
    try:
        doc = tomllib.loads(Path(path).read_text())
    except (OSError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"could not read {path}: {e}") from e

    problems: list[str] = []
    for key in doc:
        if key not in ("root", "limits", "roles"):
            problems.append(f"unknown top-level key {key!r}")

    raw_roles = doc.get("roles")
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise ValueError(f"{path}: needs at least one [roles.<name>] table")

    roles: dict[str, Role] = {}
    for name, raw in raw_roles.items():
        where = f"roles.{name}"
        if not isinstance(raw, dict):
            problems.append(f"{where}: must be a table")
            continue
        known = {"description", "task", "model", "budget", "spawn", "adhoc",
                 "adhoc_budget"}
        for key in raw:
            if key not in known:
                problems.append(f"{where}: unknown key {key!r}")
        role = Role(name=name)
        for key in ("description", "task", "model"):
            if key in raw:
                if isinstance(raw[key], str):
                    setattr(role, key, raw[key])
                else:
                    problems.append(f"{where}.{key}: must be a string")
        role.budget = _parse_budget(raw.get("budget"), f"{where}.budget", problems)
        spawn = raw.get("spawn", {})
        if not isinstance(spawn, dict):
            problems.append(f"{where}.spawn: must be a table like {{ prover = 2 }}")
            spawn = {}
        for child, n in spawn.items():
            if not isinstance(n, int) or isinstance(n, bool) or n < 1:
                problems.append(f"{where}.spawn.{child}: count must be a positive integer")
            else:
                role.spawn[child] = n
        adhoc = raw.get("adhoc", 0)
        if not isinstance(adhoc, int) or isinstance(adhoc, bool) or adhoc < 0:
            problems.append(f"{where}.adhoc: must be a non-negative integer")
        else:
            role.adhoc = adhoc
        role.adhoc_budget = _parse_budget(raw.get("adhoc_budget"),
                                          f"{where}.adhoc_budget", problems)
        if role.adhoc and not role.adhoc_budget.is_set():
            problems.append(f"{where}: adhoc sub-agents need an adhoc_budget "
                            "(an improvised agent must still be bounded)")
        roles[name] = role

    for role in roles.values():
        for child in role.spawn:
            if child not in roles:
                problems.append(f"roles.{role.name}.spawn: unknown role {child!r}")

    root = doc.get("root")
    if not isinstance(root, str) or root not in roles:
        problems.append(f"root: must name one of the roles ({', '.join(roles)})")

    # The role graph must be a DAG: depth is then bounded by the file.
    state: dict[str, int] = {}

    def visit(name: str, trail: list[str]) -> None:
        if state.get(name) == 1:
            problems.append("role cycle: " + " -> ".join(trail + [name]))
            return
        if state.get(name) == 2 or name not in roles:
            return
        state[name] = 1
        for child in roles[name].spawn:
            visit(child, trail + [name])
        state[name] = 2

    for name in roles:
        visit(name, [])

    # A child role's budget must fit inside its parent's, dimension by
    # dimension: a subtree budget the subtree could never stay within is a
    # plan bug, not something to discover mid-run.
    for role in roles.values():
        for child in list(role.spawn) + ([None] if role.adhoc else []):
            if child and child not in roles:
                continue  # already reported as an unknown role
            cb = roles[child].budget if child else role.adhoc_budget
            label = child or "adhoc"
            for dim in ("tokens", "seconds", "cost"):
                pv, cv = getattr(role.budget, dim), getattr(cb, dim)
                if pv is not None and cv is not None and cv > pv:
                    problems.append(
                        f"roles.{role.name}: {label}'s {dim} budget ({cv:g}) "
                        f"exceeds its parent's ({pv:g})")

    # Cost caps need pricing for the model that will actually run.
    from .pricing import estimate_cost
    for role in roles.values():
        model = role.model or default_model
        capped = role.budget.cost is not None or (role.adhoc and role.adhoc_budget.cost is not None)
        if capped and estimate_cost(model, 1, 1) is None:
            problems.append(f"roles.{role.name}: has a cost budget, but pricing for "
                            f"model {model!r} is unknown (use tokens instead)")

    limits = doc.get("limits", {})
    if not isinstance(limits, dict):
        problems.append("limits: must be a table")
        limits = {}
    for key in limits:
        if key not in ("max_live_agents", "on_root_exhausted"):
            problems.append(f"limits: unknown key {key!r}")
    max_live = limits.get("max_live_agents", 8)
    if not isinstance(max_live, int) or isinstance(max_live, bool) or max_live < 1:
        problems.append("limits.max_live_agents: must be a positive integer")
        max_live = 8
    on_root = limits.get("on_root_exhausted", "ask")
    if on_root not in ("ask", "finish"):
        problems.append('limits.on_root_exhausted: must be "ask" or "finish"')
        on_root = "ask"

    if problems:
        raise ValueError(f"{path}:\n" + "\n".join(f"  - {p}" for p in problems))
    return AgentPlan(path=Path(path), root=root, roles=roles,
                     max_live_agents=max_live, on_root_exhausted=on_root)


def find(project_dir: Path) -> Path | None:
    """The project's agent plan, if it has one."""
    p = Path(project_dir) / ".gerbil" / CONFIG_NAME
    return p if p.is_file() else None
