"""The session forest behind `gerbil tree`: every session log in the archive
(~/.gerbil/sessions/), linked into trees, plus the management operations the
viewer offers (names, subtree deletion, resumability).

Pure and testable: this module reads logs and the run registry and touches
files only in deliberate operations (set_label, delete_subtree). The textual
shell lives in tree_tui.py.

Three kinds of link make the trees, all read from the logs themselves:

- agent   a supervised sub-agent under the session that spawned it
          (session_start.agent.parent_id -> that session's agent_id);
- resume  a `gerbil resume` continuation under the session it continued
          (session_start.resumed_from);
- ralph   ralph-chain sessions 2..N under the chain's first session
          (session_start.ralph.ancestors[0]) -- flat, not nested N deep.

A session whose linked parent log is gone stays a root, flagged as orphaned.

Names: session logs are append-only and may already be folded into commits,
so user-given names live in a sidecar, labels.json in the archive dir, keyed
by session stem. A tree's name is its root's label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import runs

LABELS_FILE = "labels.json"


def archive_dir() -> Path:
    return Path.home() / ".gerbil" / "sessions"


@dataclass
class Node:
    stem: str
    path: Path
    project_dir: str = ""
    model: str = ""
    started: str = ""
    status: str = "unfinished"   # complete | error | interrupted | running | unfinished
    error: str = ""
    turns: int = 0
    tokens: int = 0
    prompt: str = ""
    message: str = ""            # the last assistant message (commit message, usually)
    agent: dict | None = None
    ralph: dict | None = None
    resumed_from: str | None = None
    live_run: str | None = None  # registry run name, when the session is live
    label: str = ""
    parent: str | None = None    # parent stem
    link: str = ""               # agent | resume | ralph
    orphan: bool = False
    # Anything odd about this node, shown in the details pane and marked
    # with "!" in its row: malformed fields, an ignored link (cycle,
    # ambiguous parent), ... The viewer never hides a session over it.
    notes: list[str] = field(default_factory=list)
    children: list["Node"] = field(default_factory=list)

    @property
    def is_sub_agent(self) -> bool:
        return bool(self.agent and self.agent.get("parent_id"))

    @property
    def role(self) -> str | None:
        return (self.agent or {}).get("role")


def _events(path: Path):
    """The log's events, skipping anything that is not a JSON object --
    a log is user-reachable data, so every line is suspect."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return
    for line in text.splitlines():
        try:
            e = json.loads(line)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(e, dict):
            yield e


def _str(v) -> str:
    return v if isinstance(v, str) else ""


def _int(v) -> int:
    return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def _clean_agent(v, notes: list[str]) -> dict | None:
    """A session_start `agent` block, or None. Ids must be strings; anything
    else is dropped with a note rather than trusted or crashed on."""
    if v is None:
        return None
    if not isinstance(v, dict):
        notes.append("malformed agent record ignored")
        return None
    out = dict(v)
    for key in ("agent_id", "parent_id", "role"):
        if out.get(key) is not None and not isinstance(out[key], str):
            notes.append(f"malformed agent {key} ignored")
            out[key] = None
    if out.get("budget") is not None and not isinstance(out["budget"], dict):
        out["budget"] = None
    return out


def _clean_ralph(v, notes: list[str]) -> dict | None:
    if v is None:
        return None
    if not isinstance(v, dict):
        notes.append("malformed ralph record ignored")
        return None
    out = dict(v)
    anc = out.get("ancestors")
    if anc is not None and not (isinstance(anc, list) and all(isinstance(a, str) for a in anc)):
        notes.append("malformed ralph ancestors ignored")
        out["ancestors"] = []
    return out


def scan_log(path: Path, live: dict[str, str] | None = None) -> Node | None:
    """One session log -> a Node (unlinked). None if it has no session_start.
    Replayed events (a resume's copy of its parent) are skipped: they belong
    to the parent session's node. Malformed fields degrade to "absent", with
    a note -- one bad log must never take the whole viewer down."""
    node = Node(stem=path.stem, path=path)
    seen_start = False
    last = None
    for e in _events(path):
        if e.get("replayed"):
            continue
        kind = e.get("event")
        if kind == "session_start" and not seen_start:
            seen_start = True
            node.project_dir = _str(e.get("project_dir"))
            node.model = _str(e.get("model"))
            node.started = _str(e.get("timestamp"))
            node.agent = _clean_agent(e.get("agent"), node.notes)
            node.ralph = _clean_ralph(e.get("ralph"), node.notes)
            rf = e.get("resumed_from")
            if rf is not None and not isinstance(rf, str):
                node.notes.append("malformed resumed_from ignored")
                rf = None
            node.resumed_from = rf or None
        elif kind == "turn":
            u = e.get("usage") if isinstance(e.get("usage"), dict) else {}
            node.tokens += (_int(u.get("input_tokens")) + _int(u.get("cache_read_tokens"))
                            + _int(u.get("cache_write_tokens")) + _int(u.get("output_tokens")))
            content = e.get("content")
            content = content if isinstance(content, str) else ""
            if e.get("role") == "assistant":
                node.turns += 1
                if content:
                    node.message = content
            elif e.get("role") == "user" and not node.prompt:
                node.prompt = content
        if kind in ("session_end", "error"):
            last = e
    if not seen_start:
        return None
    # The prompt of a resumed session lives in the replayed region.
    if not node.prompt:
        for e in _events(path):
            if e.get("event") == "turn" and e.get("role") == "user":
                node.prompt = _str(e.get("content"))
                break
    if last is not None and last.get("event") == "session_end":
        node.status = "complete"
    elif last is not None:
        node.status = ("interrupted" if last.get("error_type") == "KeyboardInterrupt"
                       else "error")
        node.error = f"{_str(last.get('error_type'))}: {_str(last.get('message'))}"
    if live and node.stem in live and node.status not in ("complete", "error", "interrupted"):
        node.status = "running"
        node.live_run = live[node.stem]
    elif live and node.stem in live:
        node.live_run = live[node.stem]
    return node


def live_sessions() -> dict[str, str]:
    """{session stem: run name} for the registry's live runs."""
    out = {}
    for r in runs.list_runs():
        if r.get("state") in ("running", "paused") and r.get("session"):
            out[Path(r["session"]).stem] = r["name"]
    return out


def load_labels(base: Path | None = None) -> dict[str, str]:
    p = (base or archive_dir()) / LABELS_FILE
    try:
        doc = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {k: v for k, v in doc.items() if isinstance(v, str)} if isinstance(doc, dict) else {}


def set_label(stem: str, label: str, base: Path | None = None) -> None:
    """Name (or, with an empty label, un-name) a session. Atomic replace."""
    base = base or archive_dir()
    labels = load_labels(base)
    label = " ".join(label.split())
    if label:
        labels[stem] = label
    else:
        labels.pop(stem, None)
    runs._write_json(base / LABELS_FILE, labels)


def _order(n: Node) -> tuple:
    """Oldest first; the stem breaks ties (and orders missing timestamps)."""
    return (n.started, n.stem)


def build_forest(base: Path | None = None, *, project: str | None = None,
                 live: dict[str, str] | None = None,
                 skipped: list[Path] | None = None) -> list[Node]:
    """Every session in the archive, linked into trees; returns the roots,
    oldest first (children likewise). `project` filters to one project dir
    (a tree is kept whole if its root belongs to it). Files that are not
    session logs are appended to `skipped`, if given.

    Logs are data, possibly hand-edited or copied, so linking never trusts
    them to form a forest:
    - a duplicated agent_id (two logs claiming one identity -- a copied log)
      makes that id ambiguous: its would-be children are left unlinked, with
      a note, rather than attached to whichever copy happened to sort last;
    - links that form a cycle are cut, deterministically at the cycle's
      oldest session, which becomes a root with a note -- so every session
      in the archive is always reachable, viewable, and deletable;
    - everything is traversed iteratively, so no chain depth can overflow
      the stack.
    """
    base = base or archive_dir()
    if live is None:
        live = live_sessions()
    labels = load_labels(base)
    nodes: dict[str, Node] = {}
    for p in sorted(base.glob("*.jsonl")):
        n = scan_log(p, live)
        if n is None:
            if skipped is not None:
                skipped.append(p)
            continue
        n.label = labels.get(n.stem, "")
        nodes[n.stem] = n

    by_agent_id: dict[str, list[Node]] = {}
    for n in nodes.values():
        aid = (n.agent or {}).get("agent_id")
        if aid:
            by_agent_id.setdefault(aid, []).append(n)
    for aid, owners in by_agent_id.items():
        if len(owners) > 1:
            names = ", ".join(o.stem for o in owners)
            for o in owners:
                o.notes.append(f"agent id {aid[:12]} is also claimed by another "
                               f"log ({names}) -- a copy?")

    for n in nodes.values():
        parent, link, wanted, why = None, "", False, "its parent session log is missing"
        if n.is_sub_agent:
            wanted, link = True, "agent"
            owners = by_agent_id.get(n.agent["parent_id"]) or []
            if len(owners) == 1:
                parent = owners[0]
            elif owners:
                why = ("its parent agent id is claimed by several logs ("
                       + ", ".join(o.stem for o in owners) + "); not guessing")
        elif n.resumed_from:
            wanted, link = True, "resume"
            parent = nodes.get(Path(n.resumed_from).stem)
        elif n.ralph and n.ralph.get("ancestors"):
            wanted, link = True, "ralph"
            parent = nodes.get(Path(n.ralph["ancestors"][0]).stem)
        if parent is n:
            n.notes.append(f"{link} link to itself ignored")
        elif parent is not None:
            n.parent, n.link = parent.stem, link
        elif wanted:
            n.orphan = True
            n.notes.append(f"unlinked: {why}")

    # Cycles: a parent chain that never reaches a root. Walk each node's
    # chain; a revisit within the same walk is a cycle, cut at its oldest
    # member. Iterative, and each node is settled once.
    settled: set[str] = set()
    for start in sorted(nodes.values(), key=_order):
        path, on_path = [], set()
        cur = start
        while cur is not None and cur.stem not in settled:
            if cur.stem in on_path:
                cycle = path[path.index(cur):]
                cut = min(cycle, key=_order)
                cut.notes.append(
                    f"{cut.link} link to {cut.parent} ignored: it would form a "
                    "cycle (" + " -> ".join(c.stem for c in cycle + [cycle[0]]) + ")")
                cut.parent, cut.link = None, ""
                break
            path.append(cur)
            on_path.add(cur.stem)
            cur = nodes.get(cur.parent) if cur.parent else None
        settled.update(on_path)

    for n in nodes.values():
        if n.parent:
            nodes[n.parent].children.append(n)
    for n in nodes.values():
        n.children.sort(key=_order)
    roots = sorted((n for n in nodes.values() if n.parent is None), key=_order)
    if project is not None:
        want = str(Path(project).resolve())
        roots = [r for r in roots if r.project_dir and
                 str(Path(r.project_dir).resolve()) == want]
    return roots


def walk(node: Node):
    """Pre-order, iterative (chains can be arbitrarily deep)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        stack.extend(reversed(n.children))


def find(roots: list[Node], stem: str) -> Node | None:
    for r in roots:
        for n in walk(r):
            if n.stem == stem:
                return n
    return None


def subtree_files(node: Node) -> list[Path]:
    """Every archive file a subtree owns: each session's log, its live
    .wip.patch, and the archive copy of its patch. (The project's own
    .gerbil/patches/ copies are never touched: they belong to the repo.)"""
    out = []
    for n in walk(node):
        for suffix in (".jsonl", ".wip.patch", ".patch"):
            p = n.path.with_name(n.stem + suffix)
            if p.exists():
                out.append(p)
    return out


def delete_subtree(node: Node, base: Path | None = None) -> list[Path]:
    """Delete a session and everything below it. Refuses (ValueError) if any
    of them is live -- stop it first. Returns the files removed."""
    live = [n.stem for n in walk(node) if n.status == "running" or n.live_run]
    if live:
        raise ValueError("still running: " + ", ".join(live)
                         + " -- stop it first (gerbil grab, or kill its parent)")
    files = subtree_files(node)
    for p in files:
        p.unlink(missing_ok=True)
    base = base or archive_dir()
    labels = load_labels(base)
    stems = {n.stem for n in walk(node)}
    if any(s in labels for s in stems):
        runs._write_json(base / LABELS_FILE,
                         {k: v for k, v in labels.items() if k not in stems})
    return files


def resumability(node: Node) -> tuple[bool, str]:
    """(can resume?, why not / what it will do)."""
    if node.status == "running":
        return False, f"it is running (run {node.live_run}) -- press g to grab it"
    if node.is_sub_agent:
        return False, ("a sub-agent's patch only applies on top of its parent's "
                       "tree, so it can't be resumed on its own -- resume its "
                       "tree's root instead")
    if node.status == "complete":
        return False, "it completed -- there is nothing to resume"
    if any(c.link == "resume" for c in node.children):
        return True, ("note: it was already resumed once (see below it); this "
                      "starts another continuation from the same point")
    return True, "continues the session in a fresh sandbox (gerbil resume)"


def describe(node: Node) -> str:
    """The details pane, as plain text."""
    a = node.agent or {}
    lines = [
        f"session   {node.stem}" + (f"   \"{node.label}\"" if node.label else ""),
        f"status    {node.status}" + (f" ({node.live_run})" if node.live_run else ""),
        f"project   {node.project_dir}",
        f"model     {node.model}",
        f"started   {node.started}",
        f"turns     {node.turns}    tokens {node.tokens:,}",
    ]
    if node.error:
        lines.append(f"error     {node.error[:300]}")
    if a:
        lines.append(f"agent     {(a.get('agent_id') or '?')[:12]}  depth {a.get('depth', 0)}"
                     + (f"  role {a['role']}" if a.get("role") else ""))
        if a.get("budget"):
            from .agent_config import Budget
            try:
                lines.append(f"budget    {Budget.from_dict(a['budget']).describe()}")
            except (TypeError, ValueError):
                lines.append("budget    (malformed)")
    if node.link:
        lines.append(f"linked    {node.link} child of {node.parent}")
    for note in node.notes:
        lines.append(f"note      {note}")
    if node.ralph:
        lines.append(f"ralph     session {node.ralph.get('iteration')}/{node.ralph.get('total')}")
    ok, why = resumability(node)
    lines.append(f"resume    {'yes' if ok else 'no'}: {why}")
    lines.append("")
    lines.append("prompt:")
    lines.extend("  " + l for l in node.prompt.strip().splitlines()[:25])
    if node.message:
        lines.append("")
        lines.append("last message:")
        lines.extend("  " + l for l in node.message.strip().splitlines()[:15])
    return "\n".join(lines)


def node_line(node: Node) -> str:
    """One row of the tree."""
    mark = {"agent": "", "resume": "↻ ", "ralph": "⟳ "}.get(node.link, "")
    bits = [mark + (f"“{node.label}” " if node.label else "") + node.stem]
    if node.role:
        bits.append(node.role)
    bits.append(node.status)
    bits.append(f"{node.turns}t")
    if node.orphan:
        bits.append("orphaned")
    if node.notes:
        bits.append("!")
    return " · ".join(bits)


def render_plain(roots: list[Node]) -> str:
    """The forest as indented text (for `gerbil tree --plain`), grouped by
    project."""
    out: list[str] = []
    by_project: dict[str, list[Node]] = {}
    for r in roots:
        by_project.setdefault(r.project_dir or "?", []).append(r)
    for project, rs in by_project.items():
        out.append(project)

        stack = [(r, "", i == len(rs) - 1) for i, r in reversed(list(enumerate(rs)))]
        while stack:  # iterative: no depth limit
            n, prefix, last = stack.pop()
            out.append(prefix + ("└─ " if last else "├─ ") + node_line(n))
            child_prefix = prefix + ("   " if last else "│  ")
            k = len(n.children)
            stack.extend((c, child_prefix, i == k - 1)
                         for i, c in reversed(list(enumerate(n.children))))
        out.append("")
    return "\n".join(out).rstrip() + "\n" if out else "no sessions.\n"
