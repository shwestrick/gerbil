"""Tests for `gerbil tree`: the session forest (sessions_tree.py), its
management operations, the resume refusal for sub-agent logs, and the textual
viewer driven headlessly through textual's pilot.

No Docker, no network, no real terminal.
Run: uv run python tests/test_tree.py
"""

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from gerbil import sessions_tree as st
from gerbil.session import Session


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


def make_log(base, stem, *, project="/proj/a", agent=None, ralph=None,
             resumed_from=None, end="complete", turns=2, prompt="do the thing",
             replay_from=None):
    s = Session(path=base / f"{stem}.jsonl", model="claude-opus-4-8",
                project_dir=Path(project), prompt_file=Path("p.md"),
                base_commit="abc", agent=agent, ralph=ralph,
                resumed_from=resumed_from)
    if replay_from is not None:
        for line in (base / f"{replay_from}.jsonl").read_text().splitlines():
            s.record_replayed(json.loads(line))
        s.record_resumed(f"{replay_from}.jsonl", 0)
    else:
        s.record_turn("user", prompt)
    for i in range(turns):
        s.record_turn("assistant", f"message {i}", 10, 5)
    if end == "complete":
        s.close()
    elif end == "error":
        s.record_error(RuntimeError("boom"))
    elif end == "interrupted":
        s.record_error(KeyboardInterrupt())
    (base / f"{stem}.patch").write_text("patch")
    return base / f"{stem}.jsonl"


def agent_block(aid, parent=None, role=None):
    return {"agent_id": aid, "parent_id": parent, "root_id": "R", "depth": 1 if parent else 0,
            "role": role, "budget": {"tokens": 1000, "seconds": None, "cost": None}}


def fixture():
    base = Path(tempfile.mkdtemp())
    make_log(base, "gerbil-01-root", agent=agent_block("R"), end="error")
    make_log(base, "gerbil-02-kid-a", agent=agent_block("A", "R", "x"))
    make_log(base, "gerbil-03-kid-b", agent=agent_block("B", "R", "y"), end="none")
    make_log(base, "gerbil-04-grandkid", agent=agent_block("G", "A", "grinder"))
    make_log(base, "gerbil-05-root-resume-1", agent=agent_block("R2"),
             resumed_from="gerbil-01-root.jsonl", replay_from="gerbil-01-root")
    make_log(base, "gerbil-06-ralph-01", project="/proj/b",
             ralph={"iteration": 1, "total": 3, "chain_base": "abc", "ancestors": []})
    make_log(base, "gerbil-06-ralph-02", project="/proj/b",
             ralph={"iteration": 2, "total": 3, "chain_base": "abc",
                    "ancestors": ["gerbil-06-ralph-01.patch"]})
    make_log(base, "gerbil-06-ralph-03", project="/proj/b", end="interrupted",
             ralph={"iteration": 3, "total": 3, "chain_base": "abc",
                    "ancestors": ["gerbil-06-ralph-01.patch", "gerbil-06-ralph-02.patch"]})
    make_log(base, "gerbil-07-orphan", agent=agent_block("O", "GONE"))
    (base / "gerbil-01-root.wip.patch").write_text("wip")
    (base / "junk.jsonl").write_text("not json\n")
    return base


def test_forest():
    base = fixture()
    roots = st.build_forest(base, live={"gerbil-03-kid-b": "brave-otter"})
    stems = [r.stem for r in roots]
    check("roots: tree root, ralph chain head, orphan",
          stems == ["gerbil-01-root", "gerbil-06-ralph-01", "gerbil-07-orphan"], str(stems))
    root = roots[0]
    kids = [(c.stem, c.link) for c in root.children]
    check("agent children + resume continuation under the root",
          kids == [("gerbil-02-kid-a", "agent"), ("gerbil-03-kid-b", "agent"),
                   ("gerbil-05-root-resume-1", "resume")], str(kids))
    a = st.find(roots, "gerbil-02-kid-a")
    check("grandchild nests under its parent",
          [c.stem for c in a.children] == ["gerbil-04-grandkid"])
    chain = roots[1]
    check("ralph chain flat under its first session",
          [(c.stem, c.link) for c in chain.children]
          == [("gerbil-06-ralph-02", "ralph"), ("gerbil-06-ralph-03", "ralph")])
    check("orphan flagged", roots[2].orphan)

    check("status: error", root.status == "error" and "boom" in root.error)
    check("status: complete", a.status == "complete")
    check("status: live run", st.find(roots, "gerbil-03-kid-b").status == "running"
          and st.find(roots, "gerbil-03-kid-b").live_run == "brave-otter")
    check("status: interrupted", st.find(roots, "gerbil-06-ralph-03").status == "interrupted")
    check("turns and tokens counted", a.turns == 2 and a.tokens == 30)
    resumed = st.find(roots, "gerbil-05-root-resume-1")
    check("resume node counts only its own (non-replayed) turns",
          resumed.turns == 2 and resumed.prompt == "do the thing", f"{resumed.turns} {resumed.prompt!r}")

    check("project filter keeps whole trees",
          [r.stem for r in st.build_forest(base, project="/proj/b", live={})]
          == ["gerbil-06-ralph-01"])

    text = st.render_plain(roots)
    check("plain render shows nesting and links",
          "/proj/a" in text and "├─ gerbil-02-kid-a · x · complete" in text
          and "│  └─ gerbil-04-grandkid" in text
          and "↻ gerbil-05-root-resume-1" in text and "⟳ gerbil-06-ralph-02" in text,
          text)


def test_resumability():
    base = fixture()
    roots = st.build_forest(base, live={"gerbil-03-kid-b": "brave-otter"})
    get = lambda s: st.resumability(st.find(roots, s))
    check("crashed root resumable (notes prior continuation)",
          get("gerbil-01-root")[0] and "already resumed" in get("gerbil-01-root")[1])
    check("sub-agent not resumable", not get("gerbil-02-kid-a")[0]
          and "parent's tree" in get("gerbil-02-kid-a")[1])
    check("running not resumable", not get("gerbil-03-kid-b")[0])
    check("complete not resumable", not get("gerbil-06-ralph-01")[0])
    check("interrupted ralph session resumable", get("gerbil-06-ralph-03")[0])


def test_labels_and_delete():
    base = fixture()
    st.set_label("gerbil-01-root", "  proof   of   main ", base)
    st.set_label("gerbil-04-grandkid", "grinder one", base)
    roots = st.build_forest(base, live={})
    check("label normalized and shown", roots[0].label == "proof of main"
          and "“proof of main” gerbil-01-root" in st.node_line(roots[0]))
    st.set_label("gerbil-04-grandkid", "", base)
    check("empty label clears", "gerbil-04-grandkid" not in st.load_labels(base))
    st.set_label("gerbil-04-grandkid", "grinder one", base)

    live_roots = st.build_forest(base, live={"gerbil-03-kid-b": "brave-otter"})
    try:
        st.delete_subtree(live_roots[0], base)
        check("delete refuses a subtree with a live run", False)
    except ValueError as e:
        check("delete refuses a subtree with a live run", "brave-otter" in str(e)
              or "gerbil-03-kid-b" in str(e), str(e))
    check("nothing deleted on refusal", (base / "gerbil-01-root.jsonl").exists())

    roots = st.build_forest(base, live={})
    a = st.find(roots, "gerbil-02-kid-a")
    removed = st.delete_subtree(a, base)
    check("subtree delete removes the node and its descendants' files",
          sorted(p.name for p in removed) == sorted([
              "gerbil-02-kid-a.jsonl", "gerbil-02-kid-a.patch",
              "gerbil-04-grandkid.jsonl", "gerbil-04-grandkid.patch"]), str(removed))
    check("labels of deleted sessions dropped",
          st.load_labels(base) == {"gerbil-01-root": "proof of main"})
    roots = st.build_forest(base, live={})
    check("siblings untouched",
          [c.stem for c in roots[0].children]
          == ["gerbil-03-kid-b", "gerbil-05-root-resume-1"])
    removed = st.delete_subtree(roots[0], base)
    check("deleting a root removes the whole tree incl. wip patch",
          "gerbil-01-root.wip.patch" in [p.name for p in removed]
          and not list(base.glob("gerbil-0[1-5]*")))
    check("other trees untouched", (base / "gerbil-06-ralph-01.jsonl").exists())


def test_resume_refuses_sub_agent():
    base = fixture()
    out = subprocess.run(
        [sys.executable, "-m", "gerbil", "resume", str(base / "gerbil-02-kid-a.jsonl"),
         "--plain"], capture_output=True, text=True)
    check("gerbil resume refuses a sub-agent log up front",
          out.returncode != 0 and "is a sub-agent session" in out.stderr, out.stderr)


def test_tui():
    from gerbil.tree_tui import TreeApp

    async def run():
        base = fixture()
        app = TreeApp(base=base, live={})
        async with app.run_test(size=(140, 40)) as pilot:
            tree = app.query_one("#forest")
            await pilot.pause()

            def goto(stem):
                for node in _all(tree.root):
                    if node.data == stem:
                        tree.move_cursor(node)
                        return
                raise AssertionError(stem)

            goto("gerbil-02-kid-a")
            await pilot.pause()
            details = str(app.query_one("#details").render())
            check("details pane shows the selection",
                  "gerbil-02-kid-a" in details and "role x" in details, details[:300])

            await pilot.press("r")
            await pilot.pause()
            check("resume refused for a sub-agent, with a reason",
                  "can't resume" in str(app.query_one("#status").render()))

            await pilot.press("n")
            await pilot.pause()
            await pilot.press(*"[my] kid")
            await pilot.press("enter")
            await pilot.pause()
            check("rename via dialog writes the label (markup-safe)",
                  st.load_labels(base).get("gerbil-02-kid-a") == "[my] kid",
                  str(st.load_labels(base)))

            goto("gerbil-02-kid-a")
            await pilot.press("d")
            await pilot.pause()
            await pilot.press("n")  # cancel
            await pilot.pause()
            check("delete cancelled leaves files", (base / "gerbil-02-kid-a.jsonl").exists())
            goto("gerbil-02-kid-a")
            await pilot.press("d")
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            check("delete confirmed removes the subtree",
                  not (base / "gerbil-02-kid-a.jsonl").exists()
                  and not (base / "gerbil-04-grandkid.jsonl").exists())

            goto("gerbil-06-ralph-03")
            await pilot.press("r")
            await pilot.pause()
        check("resume exits the viewer with the log to resume",
              app.return_value == ("resume", str(base / "gerbil-06-ralph-03.jsonl")),
              str(app.return_value))

    asyncio.run(run())


def raw_log(base, stem, started, **start):
    """A minimal, possibly malformed, hand-written log."""
    ev = {"event": "session_start", "timestamp": started, "model": "m",
          "project_dir": "/p", **start}
    lines = [json.dumps(ev),
             json.dumps({"event": "turn", "role": "user", "content": "hi", "usage": {}}),
             json.dumps({"event": "session_end"})]
    (base / f"{stem}.jsonl").write_text("\n".join(lines) + "\n")


def shown(roots):
    return sorted(n.stem for r in roots for n in st.walk(r))


def test_contrived_archives():
    # Cycles of every link kind: all sessions stay visible, the cycle is cut
    # at its OLDEST member, and the cut is explained.
    base = Path(tempfile.mkdtemp())
    raw_log(base, "A", "1", agent={"agent_id": "a", "parent_id": "b"})
    raw_log(base, "B", "2", agent={"agent_id": "b", "parent_id": "a"})
    raw_log(base, "C", "3", agent={"agent_id": "c", "parent_id": "a"})
    roots = st.build_forest(base, live={})
    check("agent cycle: nothing vanishes", shown(roots) == ["A", "B", "C"], str(shown(roots)))
    check("agent cycle cut at the oldest session",
          [r.stem for r in roots] == ["A"]
          and any("would form a cycle (A -> B -> A)" in n for n in roots[0].notes),
          str(roots[0].notes))
    check("cut node flagged in its row", st.node_line(roots[0]).endswith("!"))
    removed = st.delete_subtree(roots[0], base)
    check("a former cycle deletes cleanly", len(removed) == 3 and not list(base.iterdir()))

    base = Path(tempfile.mkdtemp())
    raw_log(base, "A", "1", ralph={"ancestors": ["C.patch"]})
    raw_log(base, "B", "2", ralph={"ancestors": ["A.patch"]})
    raw_log(base, "C", "3", ralph={"ancestors": ["B.patch"]})
    raw_log(base, "D", "4", resumed_from="D.jsonl")
    roots = st.build_forest(base, live={})
    check("3-cycle + self-link: all shown, two roots",
          shown(roots) == ["A", "B", "C", "D"] and [r.stem for r in roots] == ["A", "D"])
    check("self-link noted", any("link to itself" in n for n in roots[1].notes))

    base = Path(tempfile.mkdtemp())
    raw_log(base, "A", "1", agent={"agent_id": "a"}, resumed_from="B.jsonl")
    raw_log(base, "B", "2", agent={"agent_id": "b", "parent_id": "a"})
    roots = st.build_forest(base, live={})
    check("mixed-kind cycle broken", shown(roots) == ["A", "B"] and len(roots) == 1)

    # Duplicate agent ids: never guess the parent.
    base = Path(tempfile.mkdtemp())
    raw_log(base, "P1", "1", agent={"agent_id": "p"})
    raw_log(base, "P2", "2", agent={"agent_id": "p"})
    raw_log(base, "K", "3", agent={"agent_id": "k", "parent_id": "p"})
    roots = st.build_forest(base, live={})
    k = st.find(roots, "K")
    check("ambiguous parent id: child left unlinked, explained",
          k.parent is None and k.orphan
          and any("claimed by several logs (P1, P2)" in n for n in k.notes), str(k.notes))
    check("duplicate owners flagged",
          all(any("also claimed" in n for n in st.find(roots, s).notes) for s in ("P1", "P2")))

    # Malformed fields and junk lines degrade, never crash.
    base = Path(tempfile.mkdtemp())
    raw_log(base, "A", "1", agent="not-a-dict")
    raw_log(base, "B", None, ralph={"ancestors": "oops"})
    raw_log(base, "C", "3", resumed_from=42, agent={"parent_id": 7, "budget": "x"})
    (base / "D.jsonl").write_text('[1, 2]\n"str"\n{"event": "session_start", "timestamp": "4",'
                                  ' "usage": 5}\n{"event": "turn", "role": "assistant",'
                                  ' "content": ["x"], "usage": {"input_tokens": "9"}}\n')
    (base / "E.jsonl").write_bytes(b"\xff\xfe garbage")
    skipped = []
    roots = st.build_forest(base, live={}, skipped=skipped)
    check("malformed logs all shown", shown(roots) == ["A", "B", "C", "D"], str(shown(roots)))
    check("non-session file reported as skipped", [p.name for p in skipped] == ["E.jsonl"])
    notes = {n.stem: n.notes for r in roots for n in st.walk(r)}
    check("malformed fields noted",
          "malformed agent record ignored" in notes["A"]
          and "malformed ralph ancestors ignored" in notes["B"]
          and "malformed resumed_from ignored" in notes["C"]
          and "malformed agent parent_id ignored" in notes["C"], str(notes))
    check("details render for every malformed node",
          all(st.describe(n) for r in roots for n in st.walk(r)))
    check("plain render survives", "C" in st.render_plain(roots))


def test_deep_chain():
    base = Path(tempfile.mkdtemp())
    depth = 2500  # well past Python's default recursion limit of 1000
    for i in range(depth):
        raw_log(base, f"S{i:05d}", f"{i:05d}",
                **({"resumed_from": f"S{i-1:05d}.jsonl"} if i else {}))
    roots = st.build_forest(base, live={})
    check("deep chain: one root, all reachable",
          len(roots) == 1 and len(list(st.walk(roots[0]))) == depth)
    check("deep chain: plain render", len(st.render_plain(roots).splitlines()) == depth + 1)
    check("deep chain: subtree files", len(st.subtree_files(roots[0])) == depth)

    from gerbil.tree_tui import TreeApp

    from gerbil.tree_tui import MAX_VIEW_DEPTH

    async def run():
        app = TreeApp(base=base, live={})
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await pilot.press("end")
            await pilot.pause()
            tree = app.query_one("#forest")
            items, deepest = [], 0
            stack = [(tree.root, 0)]
            while stack:
                n, d = stack.pop()
                if n.data:
                    items.append(n)
                deepest = max(deepest, d)
                stack.extend((c, d + 1) for c in n.children)
            last = tree.cursor_node
            return (str(app.query_one("#details").render()), len(items), deepest,
                    str(last.label) if last else "")

    details, count, deepest, last_label = asyncio.run(run())
    check("deep chain: every session listed in the viewer", count == depth, str(count))
    check("deep chain: display nesting capped",
          deepest <= MAX_VIEW_DEPTH + 2, str(deepest))  # + project group + hidden root
    check("deep chain: flattened rows carry their true depth",
          f"depth {depth - 1}" in last_label and f"S{depth - 1:05d}" in last_label, last_label)
    check("deep chain: the viewer scrolls to the end and shows details",
          f"S{depth - 1:05d}" in details, details[:200])


def _all(node):
    yield node
    for c in node.children:
        yield from _all(c)


if __name__ == "__main__":
    test_forest()
    test_resumability()
    test_labels_and_delete()
    test_resume_refuses_sub_agent()
    test_tui()
    test_contrived_archives()
    test_deep_chain()
    print("\nall tree tests passed")
