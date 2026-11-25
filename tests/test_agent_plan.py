"""Tests for project agent plans (agent_config.py) and budget enforcement
(agents.BudgetGuard + the review protocol + plan-mode spawning).

No Docker, no network: the registry lives in a scratch dir, usage is fed
through fabricated stats.json files, and spawn_runner is stubbed.
Run: uv run python tests/test_agent_plan.py
"""

import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from gerbil import agent_config, agents, runs
from gerbil.agent_config import Budget
from gerbil.agents import AgentController, BudgetGuard, Caps
from gerbil.providers import Usage
from gerbil.session import Session
from gerbil.tools import Toolset


def check(label, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


@contextlib.contextmanager
def tmp_registry():
    real = runs.RUNNING_DIR
    runs.RUNNING_DIR = Path(tempfile.mkdtemp()) / "running"
    try:
        yield
    finally:
        runs.RUNNING_DIR = real


PLAN = """
root = "lead"

[limits]
max_live_agents = 5

[roles.lead]
description = "Plans and integrates"
task = "You lead the proof of the main theorem."
budget = { tokens = 1_000_000, time = "2h", cost = 50.0 }
spawn = { x = 1, y = 1, z = 1 }
adhoc = 1
adhoc_budget = { tokens = 50_000, time = "10m" }

[roles.x]
task = "Do x."
model = "claude-sonnet-4-5"
budget = { tokens = 400_000, time = "1h" }
spawn = { grinder = 2 }

[roles.y]
budget = { tokens = 200_000 }

[roles.z]
budget = { time = "30m" }

[roles.grinder]
budget = { tokens = 100_000, time = "15m" }
"""


def write(text):
    p = Path(tempfile.mkdtemp()) / "agents.toml"
    p.write_text(text)
    return p


def load(text):
    return agent_config.load(write(text), default_model="claude-opus-4-8")


def problems(text):
    try:
        load(text)
    except ValueError as e:
        return str(e)
    return ""


def test_parse():
    plan = load(PLAN)
    check("root role", plan.root == "lead")
    check("spawn table", plan.roles["lead"].spawn == {"x": 1, "y": 1, "z": 1})
    check("budget parsed",
          plan.roles["lead"].budget == Budget(tokens=1_000_000, seconds=7200.0, cost=50.0))
    check("limits", plan.max_live_agents == 5 and plan.on_root_exhausted == "ask")
    tree = plan.describe()
    check("describe shows nesting",
          "- lead [" in tree and "\n  - x [" in tree and "\n    - 2 x grinder [" in tree
          and "ad-hoc" in tree, tree)

    for v, s in ((90, 90.0), ("90s", 90.0), ("45m", 2700.0), ("2h", 7200.0),
                 ("1h30m", 5400.0)):
        check(f"duration {v!r}", agent_config.parse_duration(v) == s)
    for bad in ("", "abc", "5d", 0, -1, True):
        try:
            agent_config.parse_duration(bad)
            check(f"duration {bad!r} rejected", False)
        except ValueError:
            pass
    check("bad durations rejected", True)

    p = problems(PLAN.replace("spawn = { grinder = 2 }", "spawn = { nobody = 1 }"))
    check("unknown child role", "unknown role 'nobody'" in p, p)
    p = problems(PLAN + '\n[roles.grinder2]\nspawn = { grinder3 = 1 }\n'
                 '[roles.grinder3]\nspawn = { grinder2 = 1 }\n')
    check("cycle rejected", "role cycle" in p, p)
    p = problems(PLAN.replace('budget = { tokens = 100_000, time = "15m" }',
                              'budget = { tokens = 999_000, time = "15m" }'))
    check("child budget may not exceed parent's", "exceeds its parent's" in p, p)
    p = problems(PLAN.replace('adhoc_budget = { tokens = 50_000, time = "10m" }', ""))
    check("adhoc needs a budget", "adhoc_budget" in p, p)
    p = problems(PLAN.replace('time = "1h"', 'time = "soon"'))
    check("bad duration reported", "roles.x.budget.time" in p, p)
    p = problems(PLAN.replace("[limits]", "[limits]\nbogus = 1").replace(
        'description = "Plans', 'descripton = "x"\ndescription = "Plans'))
    check("all problems reported at once",
          "limits: unknown key 'bogus'" in p and "unknown key 'descripton'" in p, p)
    p = problems(PLAN.replace('model = "claude-sonnet-4-5"\nbudget = { tokens = 400_000, time = "1h" }',
                              'model = "mystery-model"\nbudget = { cost = 5.0 }'))
    check("cost cap needs known pricing", "pricing" in p, p)
    p = problems(PLAN.replace('root = "lead"', 'root = "boss"'))
    check("root must be a role", "root:" in p, p)


def set_usage(name, tokens, model="claude-opus-4-8"):
    rd = runs.run_dir(name)
    runs.write_stats_doc(rd, {"stats": {
        "model": model, "small_model": None,
        "chain_outer": {"input_tokens": tokens, "output_tokens": 0,
                        "cache_read_tokens": 0, "cache_write_tokens": 0},
        "chain_inner": {}}})


def register(name, parent=None, **meta):
    runs.create_run(name, project_dir=Path("/repo"), model="m", theme=None, command=[])
    runs.save_meta(name, parent=parent, root=meta.pop("root", parent or name),
                   status="running", pid=os.getpid(), **meta)


def test_usage():
    with tmp_registry():
        register("lead")
        register("a", parent="lead", root="lead")
        register("a1", parent="a", root="lead")
        set_usage("lead", 1000)
        set_usage("a", 200)
        set_usage("a1", 30)
        u = agents.subtree_usage("lead")
        check("subtree tokens sum descendants", u["tokens"] == 1230, str(u))
        check("subtree cost priced", u["cost"] is not None and u["cost"] > 0)
        check("child subtree excludes parent", agents.subtree_usage("a")["tokens"] == 230)
        set_usage("a1", 30, model="mystery-model")
        check("unknown pricing -> cost None", agents.subtree_usage("lead")["cost"] is None)
        b = Budget(tokens=1000)
        check("exceeded tokens", agents.exceeded(b, agents.subtree_usage("lead")) == ["tokens"])
        check("not exceeded", agents.exceeded(Budget(tokens=10**9), agents.subtree_usage("lead")) == [])


class FakeSandbox:
    submodule_paths: list[str] = []

    def wip_patch(self, base):
        return ""

    def head(self):
        return "c0ffee"

    def read_file(self, p):
        return ""

    def diff_since(self, base):
        return "diff --git a/F b/F\n+x\n"

    def get_diff(self):
        return self.diff_since("")


def plan_controller(name, plan, role):
    return AgentController(
        name=name, sandbox=FakeSandbox(), base_commit=lambda: "b",
        project_dir=Path("/repo"), model="claude-opus-4-8", child_argv=[],
        caps=Caps(0, 0, plan.max_live_agents), plan=plan, role=plan.roles[role])


def test_plan_spawn():
    plan = load(PLAN)
    spawned = []
    real = runs.spawn_runner
    runs.spawn_runner = lambda name, argv, cwd: spawned.append((name, argv)) or 1
    try:
        with tmp_registry():
            register("lead", role="lead", agents_plan="/frozen/agents.toml",
                     budget=plan.roles["lead"].budget.to_dict(), base_sha="abc",
                     base_chain=[])
            ctl = plan_controller("lead", plan, "lead")
            schema = next(t for t in ctl.schemas() if t["name"] == "spawn_agent")
            check("spawn schema offers the plan's roles",
                  schema["input_schema"]["properties"]["role"]["enum"] == ["x", "y", "z"])
            check("spawn schema has no free model knob",
                  "model" not in schema["input_schema"]["properties"])
            check("review_agent offered", "review_agent" in ctl.tool_names())

            r = ctl.dispatch("spawn_agent", {"role": "x", "prompt": "the x part"})
            check("role spawn ok", not r.is_error, r.content)
            name, argv = spawned[-1]
            meta = runs.load_meta(name)
            check("role/budget/plan recorded",
                  meta["role"] == "x" and meta["budget"]["tokens"] == 400_000
                  and meta["agents_plan"] == "/frozen/agents.toml", str(meta))
            check("role model used", argv[argv.index("--model") + 1] == "claude-sonnet-4-5")
            prompt = Path(argv[argv.index("--prompt") + 1]).read_text()
            check("role task leads the prompt", prompt == "Do x.\n\nthe x part", prompt)

            r = ctl.dispatch("spawn_agent", {"role": "x", "prompt": "again"})
            check("per-role count enforced", r.is_error and "allows 1" in r.content, r.content)
            r = ctl.dispatch("spawn_agent", {"role": "grinder", "prompt": "no"})
            check("role not in my spawn table refused",
                  r.is_error and "does not let a lead spawn" in r.content, r.content)
            r = ctl.dispatch("spawn_agent", {"prompt": "help me"})
            check("ad-hoc spawn ok", not r.is_error, r.content)
            meta = runs.load_meta(spawned[-1][0])
            check("ad-hoc gets adhoc_budget, no role",
                  meta["role"] is None and meta["budget"]["tokens"] == 50_000)
            check("second ad-hoc refused",
                  ctl.dispatch("spawn_agent", {"prompt": "more"}).is_error)

            y = plan_controller("y-agent", plan, "y")
            check("leaf role cannot spawn", not y.can_spawn()
                  and "spawn_agent" not in y.tool_names())
    finally:
        runs.spawn_runner = real


def decide_later(name, action, delay=0.3, **kw):
    def go():
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            rv = agents.load_review(name)
            if rv:
                time.sleep(delay)
                agents.write_decision(name, rv["id"], action, **kw)
                return
            time.sleep(0.05)
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def test_guard():
    agents.REVIEW_POLL_SECONDS = 0.05
    with tmp_registry():
        register("lead", budget={"tokens": 1000, "seconds": None, "cost": None})
        register("kid", parent="lead", root="lead",
                 budget={"tokens": 100, "seconds": None, "cost": None})
        logged = []

        class Rec:
            def record_event(self, kind, /, **f):
                logged.append((kind, f))

        g = BudgetGuard("kid", is_root=False, parent="lead",
                        session_getter=lambda: Rec())
        check("within budget -> carry on", g.before_turn() == (None, None))

        set_usage("kid", 150)
        decide_later("kid", "extend", grant={"tokens": 100}, note="be quick")
        action, note = g.before_turn()
        check("exhausted child blocks, then extension resumes it",
              action is None and "EXTENDED" in note and "be quick" in note, f"{action} {note}")
        check("extension applied to own meta", runs.load_meta("kid")["budget"]["tokens"] == 200)
        check("review files consumed",
              agents.load_review("kid") is None
              and not (runs.run_dir("kid") / "decision.json").exists())
        check("review_pending cleared", runs.load_meta("kid").get("review_pending") is None)
        kinds = [k for k, _ in logged]
        check("budget review + decision logged in the agent's own log",
              kinds == ["budget_review", "budget_decision"]
              and logged[1][1]["action"] == "extend"
              and logged[1][1]["grant"] == {"tokens": 100}, str(logged))

        set_usage("kid", 250)
        answers = [("extend", {"tokens": 10}), ("finish", None)]
        reviews_seen = []

        def reviewer():
            # Answer each NEW review in turn, as a real reviewer would.
            deadline = time.monotonic() + 10
            while answers and time.monotonic() < deadline:
                rv = agents.load_review("kid")
                if rv and rv["id"] not in reviews_seen:
                    reviews_seen.append(rv["id"])
                    action, grant = answers.pop(0)
                    agents.write_decision("kid", rv["id"], action, grant=grant)
                time.sleep(0.05)

        t = threading.Thread(target=reviewer, daemon=True)
        t.start()
        action, _ = g.before_turn()
        t.join(timeout=10)
        check("too-small grant re-reviews; finish ends it",
              action == "finish" and len(reviews_seen) == 2, str(reviews_seen))

        root = BudgetGuard("lead", is_root=True, root_policy="finish")
        set_usage("lead", 5000)
        check("root with on_root_exhausted=finish wraps up",
              root.before_turn()[0] == "finish")


def test_review_protocol():
    plan = load(PLAN)
    with tmp_registry():
        register("lead", role="lead", budget={"tokens": 1000, "seconds": None, "cost": None})
        register("kid", parent="lead", root="lead", role="x",
                 budget={"tokens": 100, "seconds": None, "cost": None})
        register("other", parent="lead", root="lead", role="y")
        set_usage("lead", 500)
        set_usage("kid", 50)
        ctl = plan_controller("lead", plan, "lead")
        kid = BudgetGuard("kid", is_root=False, parent="lead")

        check("no reviews -> no note", ctl.turn_notes() is None)
        r = kid.dispatch("request_budget", {"reason": "almost there", "tokens": 300})
        check("request filed", "requested +300 tokens" in r.content, r.content)
        note = ctl.turn_notes()
        check("parent told once", note and "kid requests +300 tokens" in note
              and ctl.turn_notes() is None, str(note))

        # Headroom: lead has 1000 - (500 + 50) = 450 left.
        r = ctl.dispatch("review_agent", {"name": "kid", "action": "extend", "tokens": 600})
        check("extension beyond own headroom refused",
              r.is_error and "cannot afford" in r.content and "request_budget" in r.content,
              r.content)
        r = ctl.dispatch("review_agent", {"name": "kid", "action": "extend"})
        check("extend defaults to the requested amount", not r.is_error and "+300" in r.content,
              r.content)
        action, note = kid.before_turn()
        check("granted request delivered as a note",
              action is None and "GRANTED" in note, str(note))
        check("grant applied", runs.load_meta("kid")["budget"]["tokens"] == 400)

        kid.dispatch("request_budget", {"reason": "more", "tokens": 10})
        ctl.dispatch("review_agent", {"name": "kid", "action": "deny", "note": "no"})
        action, note = kid.before_turn()
        check("deny delivered", "DENIED" in note, str(note))

        check("review of non-child refused",
              ctl.dispatch("review_agent", {"name": "nope", "action": "finish"}).is_error)
        check("no open review -> error",
              ctl.dispatch("review_agent", {"name": "other", "action": "finish"}).is_error)

        # wait_agents returns early when a child needs review.
        agents.WAIT_POLL_SECONDS = 0.05
        kid.request("again", {"tokens": 5})
        t0 = time.monotonic()
        r = ctl.dispatch("wait_agents", {"names": ["other"], "timeout_seconds": 30})
        check("wait interrupted by a pending review",
              time.monotonic() - t0 < 5 and "need your review" in r.content, r.content)
        check("stop note mentions pending review",
              "waiting on your review" in (ctl.stop_note() or ""))


def test_run_session_budget_finish():
    import gerbil.agent as agent

    class Guard:
        calls = 0

        class budget:  # noqa: N801 -- mimics BudgetGuard.budget
            @staticmethod
            def is_set():
                return True

            @staticmethod
            def describe():
                return "100 tokens"

        def _reviewer(self):
            return "your parent agent p"

        def before_turn(self, view=None):
            Guard.calls += 1
            return ("finish", None) if Guard.calls == 2 else (None, None)

        def schemas(self):
            return []

        def tool_names(self):
            return set()

    state = {"tools_seen": []}

    def stub(model, system, messages, tools, provider, read_file=None,
             session=None, view=None):
        state["tools_seen"].append(len(tools))
        state.setdefault("system", system)
        if not tools:
            state["last_user"] = messages[-1]["content"]
            return ([{"type": "text", "text": "Partial work"}], [], "Partial work",
                    Usage(input_tokens=1, output_tokens=1))
        call = {"name": "bash", "args": {"command": "ls"}, "id": "t1", "raw_part": None}
        return ([{"type": "tool_call", "name": "bash", "args": {"command": "ls"},
                  "id": "t1", "raw_part": None}], [call], "",
                Usage(input_tokens=1, output_tokens=1))

    class TS(Toolset):
        def dispatch(self, name, args):
            from gerbil.tools import ToolResult
            return ToolResult("ok")

    ts = TS(FakeSandbox())
    ts.budget = Guard()
    tmp = Path(tempfile.mkdtemp())
    session = Session(path=tmp / "s.jsonl", model="claude-opus-4-8",
                      project_dir=Path("/repo"), prompt_file=Path("/p.md"))
    real_turn, real_window = agent._run_turn_with_retry, agent.get_context_window
    agent._run_turn_with_retry = stub
    agent.get_context_window = lambda *a, **k: 100_000
    try:
        result = agent.run_session(FakeSandbox(), session, "task", "claude-opus-4-8", ts,
                                   provider="anthropic", max_turns=10)
    finally:
        agent._run_turn_with_retry, agent.get_context_window = real_turn, real_window
    check("budget finish ends the loop after one working turn, then commits",
          state["tools_seen"][0] > 0 and state["tools_seen"][1:] == [0],
          str(state["tools_seen"]))
    last = state["last_user"]
    text = json.dumps(last)
    check("model told why it stopped, with the commit request",
          "budget is exhausted" in text and "commit message" in text, text[:500])
    check("commit message captured", result.commit_message == "Partial work")
    check("budget note in system prompt", "working under a budget: 100 tokens" in state["system"])


if __name__ == "__main__":
    test_parse()
    test_usage()
    test_plan_spawn()
    test_guard()
    test_review_protocol()
    test_run_session_budget_finish()
    print("\nall agent-plan tests passed")
