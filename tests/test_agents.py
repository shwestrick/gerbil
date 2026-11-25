"""Tests for supervisor mode (agents.py + its registry/loop/tool plumbing).

Covers: cap derivation, the tree visibility rule (runs.is_descendant), the
supervisor tools against a fabricated registry (spawn bookkeeping, the
indistinguishable "not yours" refusal, merge rules, limits), the stop-note
nudge, cascade cancellation in runs.run_runner, and run_session's stop hook.

No Docker, no network: spawn_runner is stubbed and the sandbox is a fake.
Run: uv run python tests/test_agents.py
"""

import contextlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from gerbil import agents, runs
from gerbil.agents import AgentController, Caps
from gerbil.providers import Usage
from gerbil.session import Session
from gerbil.tools import Toolset


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


@contextlib.contextmanager
def tmp_registry():
    real = runs.RUNNING_DIR
    runs.RUNNING_DIR = Path(tempfile.mkdtemp()) / "running"
    try:
        yield runs.RUNNING_DIR
    finally:
        runs.RUNNING_DIR = real


def register(name, parent=None, root=None, status="running", pid=None, **meta):
    """A registry entry. Live runs need a live pid for classify(); the test
    process's own pid serves (the tests never signal those)."""
    import os

    runs.create_run(name, project_dir=Path("/repo"), model="m", theme=None,
                    command=[])
    runs.save_meta(name, parent=parent, root=root or parent or name,
                   status=status, pid=pid if pid is not None else os.getpid(),
                   **meta)


class FakeSandbox:
    submodule_paths: list[str] = []

    def __init__(self, snapshot="", am_ok=True):
        self.snapshot = snapshot
        self.am_ok = am_ok
        self.commits: list[str] = []
        self.applied: list[str] = []

    def wip_patch(self, base):
        return self.snapshot

    def head(self):
        return "c0ffee"

    def commit(self, message):
        self.commits.append(message)
        return True

    def try_git_am_3way(self, text):
        if self.am_ok:
            self.applied.append(text)
            return True, "Applying: x"
        return False, "CONFLICT (content): Merge conflict in Foo.lean"

    def read_file(self, path):
        return ""

    def diff_since(self, base):
        return ""

    def get_diff(self):
        return ""


PATCH = """From 1 Mon Sep 17 00:00:00 2001
Subject: [PATCH] prove foo

---
diff --git a/Foo.lean b/Foo.lean
--- a/Foo.lean
+++ b/Foo.lean
@@ -1 +1 @@
-theorem foo : True := by sorry
+theorem foo : True := trivial
""" + "-- \n2.40\n"


def controller(name="root", sandbox=None, caps=None):
    return AgentController(
        name=name, sandbox=sandbox or FakeSandbox(), base_commit=lambda: "base",
        project_dir=Path("/repo"), model="claude-opus-4-8",
        child_argv=["--skip-cache"],
        caps=caps or Caps(depth_left=2, max_children=3, max_agents=4),
    )


def test_caps():
    c = Caps(depth_left=2, max_children=4, max_agents=8, max_turns=50)
    ch = c.for_child(can_spawn=True, max_turns=None)
    check("child loses one level", ch.depth_left == 1)
    check("child inherits turn cap", ch.max_turns == 50)
    check("child turn cap never exceeds parent's",
          c.for_child(True, 500).max_turns == 50 and c.for_child(True, 10).max_turns == 10)
    check("can_spawn=False -> depth 0", c.for_child(False, None).depth_left == 0)
    check("depth never negative",
          Caps(0, 1, 1).for_child(True, None).depth_left == 0)
    check("dict round-trip", Caps.from_dict(c.to_dict()) == c)


def test_visibility():
    with tmp_registry():
        register("root")
        register("a", parent="root", root="root")
        register("b", parent="root", root="root")
        register("a1", parent="a", root="root")
        register("other")  # a different tree
        check("child is descendant", runs.is_descendant("root", "a"))
        check("grandchild is descendant", runs.is_descendant("root", "a1"))
        check("self is not descendant", not runs.is_descendant("a", "a"))
        check("sibling is not", not runs.is_descendant("a", "b"))
        check("ancestor is not", not runs.is_descendant("a1", "root"))
        check("cousin is not", not runs.is_descendant("b", "a1"))
        check("other tree is not", not runs.is_descendant("root", "other"))
        check("missing is not", not runs.is_descendant("root", "nope"))
        # A corrupted cycle must terminate.
        runs.save_meta("a", parent="a1")
        check("cycle terminates", not runs.is_descendant("root", "a1"))

        runs.save_meta("a", parent="root")
        ctl = controller("a")
        sib = ctl.dispatch("peek_agent", {"name": "b"})
        missing = ctl.dispatch("peek_agent", {"name": "zz"})
        check("sibling refused", sib.is_error)
        check("refusal indistinguishable from missing",
              sib.content.replace("'b'", "X") == missing.content.replace("'zz'", "X"),
              f"{sib.content!r} vs {missing.content!r}")
        check("kill of ancestor refused",
              ctl.dispatch("kill_agent", {"name": "root"}).is_error)
        check("own child visible", not ctl.dispatch("peek_agent", {"name": "a1"}).is_error)


def test_spawn():
    spawned = []
    real = runs.spawn_runner
    runs.spawn_runner = lambda name, argv, cwd: spawned.append((name, argv)) or 4242
    try:
        with tmp_registry():
            register("root", base_sha="abc123", base_chain=["/p/one.patch"], depth=0)
            ctl = controller(sandbox=FakeSandbox(snapshot=PATCH))
            check("spawn advertised", "spawn_agent" in ctl.tool_names())
            r = ctl.dispatch("spawn_agent", {"prompt": "prove foo", "max_turns": 20,
                                             "can_spawn": True})
            check("spawn ok", not r.is_error, r.content)
            name, argv = spawned[0]
            meta = runs.load_meta(name)
            check("parent/root/depth recorded",
                  (meta["parent"], meta["root"], meta["depth"]) == ("root", "root", 1),
                  str(meta))
            check("base_sha inherited", meta["base_sha"] == "abc123")
            chain = meta["base_chain"]
            check("chain = parent's chain + snapshot",
                  chain[0] == "/p/one.patch" and len(chain) == 2
                  and Path(chain[1]).read_text() == PATCH, str(chain))
            check("caps derived", meta["caps"]["depth_left"] == 1
                  and meta["caps"]["max_turns"] == 20)
            check("argv marks child + passes flags",
                  "--_child-of" in argv and argv[argv.index("--_child-of") + 1] == "root"
                  and "--skip-cache" in argv and "--plain" in argv
                  and argv[argv.index("--max-turns") + 1] == "20", str(argv))
            prompt = Path(argv[argv.index("--prompt") + 1])
            check("prompt written to run dir", prompt.read_text() == "prove foo")

            # No snapshot -> chain is just the parent's.
            ctl.sandbox.snapshot = ""
            ctl.dispatch("spawn_agent", {"prompt": "x"})
            check("empty snapshot adds nothing to the chain",
                  runs.load_meta(spawned[1][0])["base_chain"] == ["/p/one.patch"])

            check("empty prompt refused",
                  ctl.dispatch("spawn_agent", {"prompt": "  "}).is_error)
            ctl.dispatch("spawn_agent", {"prompt": "third"})
            r = ctl.dispatch("spawn_agent", {"prompt": "fourth"})
            check("max_children enforced", r.is_error and "limit is 3" in r.content,
                  r.content)

            # depth 0 -> the tool is not even offered, and refused if called.
            leaf = controller("leaf", caps=Caps(0, 3, 4))
            check("depth 0 hides spawn_agent", "spawn_agent" not in leaf.tool_names())
            check("depth 0 refuses spawn",
                  leaf.dispatch("spawn_agent", {"prompt": "x"}).is_error)
    finally:
        runs.spawn_runner = real


def test_max_agents():
    real = runs.spawn_runner
    runs.spawn_runner = lambda *a: 1
    try:
        with tmp_registry():
            register("root")
            for n in ("a", "b"):
                register(n, parent="root", root="root")
            ctl = controller(caps=Caps(depth_left=1, max_children=10, max_agents=2))
            r = ctl.dispatch("spawn_agent", {"prompt": "x"})
            check("tree-wide live limit enforced", r.is_error and "limit 2" in r.content,
                  r.content)
            runs.save_meta("a", status="complete")
            check("finished agents free a slot",
                  not ctl.dispatch("spawn_agent", {"prompt": "x"}).is_error)
    finally:
        runs.spawn_runner = real


def finish(name, patch_text):
    rd = runs.run_dir(name)
    patch = rd / "out.patch"
    patch.write_text(patch_text)
    agents.write_result(name, final_text="done", commit_message="Prove foo",
                        patch_path=patch if patch_text else None,
                        session_path=rd / "s.jsonl")
    runs.save_meta(name, status="complete")


def test_merge_and_wait():
    with tmp_registry():
        register("root")
        register("a", parent="root", root="root")
        register("a1", parent="a", root="root")
        register("b", parent="root", root="root")
        sb = FakeSandbox()
        ctl = controller(sandbox=sb)

        check("merge of running child refused",
              ctl.dispatch("merge_agent", {"name": "a"}).is_error)
        finish("a", PATCH)
        finish("a1", PATCH)
        check("merge of grandchild refused (its parent merges it)",
              ctl.dispatch("merge_agent", {"name": "a1"}).is_error)

        w = ctl.dispatch("wait_agents", {"names": ["a"], "timeout_seconds": 1})
        check("wait reports finished child's files + report",
              "all finished" in w.content and "Foo.lean (+1/-1)" in w.content
              and "Prove foo" in w.content, w.content)

        r = ctl.dispatch("merge_agent", {"name": "a"})
        check("merge applies", not r.is_error and sb.applied == [PATCH], r.content)
        check("checkpoint committed first", len(sb.commits) == 1)
        check("merged flag set", runs.load_meta("a")["merged"] is True)
        check("double merge refused",
              ctl.dispatch("merge_agent", {"name": "a"}).is_error)

        finish("b", PATCH)
        sb.am_ok = False
        r = ctl.dispatch("merge_agent", {"name": "b"})
        check("conflict reported, not merged",
              r.is_error and "CONFLICT" in r.content
              and not runs.load_meta("b").get("merged"), r.content)

        listing = ctl.dispatch("list_agents", {}).content
        check("list shows tree with indentation",
              "- a: complete [merged]" in listing and "  - a1:" in listing, listing)

        agents.WAIT_POLL_SECONDS, real_poll = 0.05, agents.WAIT_POLL_SECONDS
        register("c", parent="root", root="root")
        t0 = time.monotonic()
        w = ctl.dispatch("wait_agents", {"names": ["c"], "timeout_seconds": 1})
        agents.WAIT_POLL_SECONDS = real_poll
        check("wait times out on a live child",
              "timed out" in w.content and time.monotonic() - t0 >= 1, w.content)


def test_stop_note():
    with tmp_registry():
        register("root")
        register("a", parent="root", root="root")
        ctl = controller()
        note = ctl.stop_note()
        check("live child -> note", note is not None and "still running: a" in note)
        check("same situation -> not repeated", ctl.stop_note() is None)
        finish("a", PATCH)
        note = ctl.stop_note()
        check("unmerged result -> new note", note is not None and "not merged: a" in note)
        runs.save_meta("a", merged=True)
        check("all settled -> no note", ctl.stop_note() is None)


def test_cascade():
    with tmp_registry():
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        try:
            register("parent")
            register("kid", parent="parent", root="parent", pid=proc.pid)
            runs.run_runner("parent", lambda: None)
            proc.wait(timeout=10)
            check("child process terminated on parent exit", proc.returncode is not None)
            check("child marked cancel-requested",
                  runs.load_meta("kid").get("cancel_requested") is True)
            check("parent status complete",
                  runs.load_meta("parent")["status"] == "complete")
        finally:
            if proc.poll() is None:
                proc.kill()

        # A runner that was asked to cancel records "cancelled".
        register("victim")
        runs.save_meta("victim", cancel_requested=True)
        try:
            runs.run_runner("victim", lambda: sys.exit(143))
        except SystemExit:
            pass
        check("cancelled status recorded",
              runs.load_meta("victim")["status"] == "cancelled")
        check("cancelled is terminal",
              runs.classify(runs.load_meta("victim")) == "cancelled")


def test_run_session_stop_hook():
    import gerbil.agent as agent

    class Ctl:
        caps = Caps(1, 1, 1)
        calls = 0

        def can_spawn(self):
            return True

        def schemas(self):
            return []

        def tool_names(self):
            return set()

        def stop_note(self):
            Ctl.calls += 1
            return "children outstanding" if Ctl.calls == 1 else None

    systems = []
    state = {"n": 0}

    def stub(model, system, messages, tools, provider, read_file=None,
             session=None, view=None):
        systems.append(system)
        state["n"] += 1
        return ([{"type": "text", "text": "done"}], [], "done",
                Usage(input_tokens=10, output_tokens=5))

    tmp = Path(tempfile.mkdtemp())
    session = Session(path=tmp / "s.jsonl", model="claude-opus-4-8",
                      project_dir=Path("/repo"), prompt_file=Path("/p.md"))
    ts = Toolset(FakeSandbox(), agents=Ctl())
    real_turn, real_window = agent._run_turn_with_retry, agent.get_context_window
    agent._run_turn_with_retry = stub
    agent.get_context_window = lambda *a, **k: 100_000
    try:
        agent.run_session(FakeSandbox(), session, "task", "claude-opus-4-8", ts,
                          provider="anthropic", max_turns=10)
    finally:
        agent._run_turn_with_retry, agent.get_context_window = real_turn, real_window
    events = [json.loads(l) for l in session.path.read_text().splitlines()]
    users = [e["content"] for e in events if e.get("event") == "turn" and e["role"] == "user"]
    check("stop hook sent the model back exactly once",
          state["n"] == 2 and users == ["task", "children outstanding"],
          f"{state['n']} turns, users={users}")
    check("supervisor note in system prompt", "spawn_agent" in systems[0])


def test_toolset_routing():
    class Ctl:
        def schemas(self):
            return [{"name": "list_agents", "description": "", "input_schema": {}}]

        def tool_names(self):
            return {"list_agents"}

        def dispatch(self, name, args):
            from gerbil.tools import ToolResult
            return ToolResult("routed")

    ts = Toolset(FakeSandbox(), agents=Ctl())
    check("agent tools advertised",
          "list_agents" in [t["name"] for t in ts.schemas()])
    check("agent tools routed", ts.dispatch("list_agents", {}).content == "routed")
    check("no agents -> no agent tools",
          "list_agents" not in [t["name"] for t in Toolset(FakeSandbox()).schemas()])


def read_log(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def test_logged_relations():
    """Agent relations land in the session logs themselves: ids and events
    on the parent side, the agent block in each session_start, and a base
    chain rebuildable from logs alone (tamper-checked)."""
    spawned = []
    real = runs.spawn_runner
    runs.spawn_runner = lambda name, argv, cwd: spawned.append(name) or 1
    tmp = Path(tempfile.mkdtemp())
    try:
        with tmp_registry():
            register("root", agent_id="R" * 32, root_id="R" * 32, base_sha="abc",
                     base_chain=[], session=str(tmp / "root.jsonl"))
            root_session = Session(path=tmp / "root.jsonl", model="m",
                                   project_dir=Path("/repo"), prompt_file=Path("/p"),
                                   agent=agents.session_agent_info("root"))
            sb = FakeSandbox(snapshot=PATCH)
            ctl = controller(sandbox=sb)
            ctl.session_getter = lambda: root_session
            ctl.dispatch("spawn_agent", {"prompt": "do it", "can_spawn": True})
            kid = spawned[-1]
            kmeta = runs.load_meta(kid)
            check("child has its own agent id, root id inherited",
                  len(kmeta["agent_id"]) == 32 and kmeta["agent_id"] != "R" * 32
                  and kmeta["root_id"] == "R" * 32, str(kmeta))
            spawn_ev = [e for e in read_log(tmp / "root.jsonl") if e["event"] == "agent_spawn"]
            check("agent_spawn logged with id, prompt, snapshot hash",
                  len(spawn_ev) == 1 and spawn_ev[0]["agent_id"] == kmeta["agent_id"]
                  and spawn_ev[0]["prompt"] == "do it"
                  and spawn_ev[0]["base_patch_sha256"] == agents._sha(PATCH))

            # The child's own session_start agent block.
            runs.save_meta(kid, session=str(tmp / "kid.jsonl"))
            info = agents.session_agent_info(kid)
            check("child agent block links parent by id and session",
                  info["parent_id"] == "R" * 32 and info["parent_run"] == "root"
                  and info["parent_session"] == str(tmp / "kid.jsonl").replace("kid", "root")
                  and info["base_patch"] == PATCH and info["base_sha"] == "abc", str(info))
            Session(path=tmp / "kid.jsonl", model="m", project_dir=Path("/repo"),
                    prompt_file=Path("/p"), agent=info)

            # A grandchild, spawned from the child with ITS snapshot.
            PATCH2 = PATCH.replace("prove foo", "prove bar")
            kctl = controller(kid, sandbox=FakeSandbox(snapshot=PATCH2))
            kctl.dispatch("spawn_agent", {"prompt": "deeper"})
            gk = spawned[-1]
            runs.save_meta(gk, session=str(tmp / "gk.jsonl"))
            ginfo = agents.session_agent_info(gk)
            Session(path=tmp / "gk.jsonl", model="m", project_dir=Path("/repo"),
                    prompt_file=Path("/p"), agent=ginfo)
            check("grandchild embeds only its own snapshot",
                  ginfo["base_patch"] == PATCH2 and len(ginfo["base_chain_sha256"]) == 2)

            # Delete the registry: the logs alone must rebuild the chain.
            runs.remove_run(kid)
            runs.remove_run(gk)
            start = agents._session_start(tmp / "gk.jsonl")
            base, chain = agents.chain_from_logs(start["agent"], tmp / "gk.jsonl")
            check("chain rebuilt from logs after registry deletion",
                  base == "abc" and chain == [PATCH, PATCH2])

            # Tampering with a parent log is detected.
            lines = (tmp / "kid.jsonl").read_text().replace("prove foo", "prove baz")
            (tmp / "kid.jsonl").write_text(lines)
            try:
                agents.chain_from_logs(start["agent"], tmp / "gk.jsonl")
                check("tampered parent log rejected", False)
            except ValueError:
                check("tampered parent log rejected", True)

            # Merge + outcome events on the parent side.
            register("m1", parent="root", root="root", agent_id="M" * 32,
                     session=str(tmp / "m1.jsonl"))
            finish("m1", PATCH)
            ctl.dispatch("merge_agent", {"name": "m1"})
            ctl.dispatch("wait_agents", {"names": ["m1"], "timeout_seconds": 1})
            evs = read_log(tmp / "root.jsonl")
            merge = [e for e in evs if e["event"] == "agent_merge"]
            outcome = [e for e in evs if e["event"] == "agent_outcome"]
            check("agent_merge embeds the applied patch",
                  len(merge) == 1 and merge[0]["ok"] and merge[0]["patch"] == PATCH
                  and merge[0]["agent_id"] == "M" * 32
                  and merge[0]["session"] == str(tmp / "m1.jsonl"))
            check("agent_outcome logged exactly once",
                  len(outcome) == 1 and outcome[0]["state"] == "complete"
                  and outcome[0]["files"] == ["Foo.lean"], str(outcome))
    finally:
        runs.spawn_runner = real


if __name__ == "__main__":
    test_caps()
    test_visibility()
    test_spawn()
    test_max_agents()
    test_merge_and_wait()
    test_stop_note()
    test_cascade()
    test_run_session_stop_hook()
    test_toolset_routing()
    test_logged_relations()
    print("\nall agents tests passed")
