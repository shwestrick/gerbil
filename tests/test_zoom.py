"""Tests for big-small mode's inner loop (agent._run_zoom) and tool schemas.

Pure/fast: drives _run_zoom with a stubbed _run_turn_with_retry (no network,
no Docker) against a scratch Session, and checks that Toolset advertises the
zoom tools to the right side only.

Run with: uv run python tests/test_zoom.py
"""

import json
import tempfile
from pathlib import Path

import gerbil.agent as agent
from gerbil.agent import _run_zoom
from gerbil.providers import Usage
from gerbil.session import Session
from gerbil.tools import Toolset


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


class FakeSandbox:
    """The bits of LeanSandbox that _run_zoom touches."""

    def read_file(self, path: str) -> str:
        return "theorem foo : True := by sorry\n"

    def run(self, command: str):
        raise AssertionError("no bash expected in these tests")


def make_session(tmp: Path) -> Session:
    return Session(
        path=tmp / "zoom-test.jsonl",
        model="claude-opus-4-8",
        project_dir=Path("/repo"),
        prompt_file=Path("/p.md"),
        small_model="claude-haiku-4-5-20251001",
        inner_max_turns=25,
    )


def read_events(session: Session) -> list[dict]:
    return [json.loads(line) for line in session.path.read_text().splitlines()]


def scripted(turns):
    """A _run_turn_with_retry stub yielding pre-scripted turns in order.

    Each entry is (text, tool_calls); usage is a fixed 10 in / 5 out per turn.
    """
    state = {"n": 0, "calls": []}

    def stub(model, system, messages, tools, provider, read_file=None, session=None):
        state["calls"].append({
            "model": model, "n_messages": len(messages),
            "tools": [t["name"] for t in tools],
        })
        text, tool_calls = turns[state["n"]]
        state["n"] += 1
        parts = ([{"type": "text", "text": text}] if text else []) + [
            {"type": "tool_call", "name": tc["name"], "args": tc["args"],
             "id": tc["id"], "raw_part": None}
            for tc in tool_calls
        ]
        return parts, tool_calls, text, Usage(input_tokens=10, output_tokens=5)

    return stub, state


def tc(name, args, cid):
    return {"name": name, "args": args, "id": cid, "raw_part": None}


def test_zoom_out_returns_summary() -> None:
    """The small model works a turn, then zooms out: the summary comes back,
    usage is accumulated, and every inner event carries the zoom tag."""
    tmp = Path(tempfile.mkdtemp())
    session = make_session(tmp)
    stub, state = scripted([
        ("looking", [tc("read_file", {"path": "Foo.lean"}, "t1")]),
        ("", [tc("zoom_out", {"summary": "closed the sorry with rfl"}, "t2")]),
    ])
    agent._run_turn_with_retry = stub
    agent.get_context_window = lambda *a, **k: None

    summary, usage = _run_zoom(
        FakeSandbox(), session, Toolset(FakeSandbox()), None,
        "claude-haiku-4-5-20251001",
        {"prompt": "Use only simp lemmas.", "file": "Foo.lean", "line": 3},
        inner_max_turns=25, wip_patch_path=None,
    )
    check("summary returned", summary == "closed the sorry with rfl", summary)
    check("usage accumulated", usage.input_tokens == 20 and usage.output_tokens == 10,
          repr(usage))
    check("small model got zoom_out tool",
          "zoom_out" in state["calls"][0]["tools"], str(state["calls"][0]["tools"]))
    check("small model did not get zoom_in",
          "zoom_in" not in state["calls"][0]["tools"])

    events = read_events(session)
    convo = [e for e in events if e["event"] in ("turn", "tool_call", "tool_result")]
    check("all inner events zoom-tagged", all(e.get("zoom") is True for e in convo),
          json.dumps(convo, indent=2))
    check("initial prompt is the big model's, with the task line appended",
          convo[0]["role"] == "user"
          and convo[0]["content"].startswith("Use only simp lemmas.")
          and "YOUR TASK" in convo[0]["content"]
          and "Foo.lean:3" in convo[0]["content"]
          and "25 turns" in convo[0]["content"],
          convo[0].get("content", ""))
    check("zoom_out call recorded",
          any(e["event"] == "tool_call" and e["name"] == "zoom_out" for e in convo))
    check("zoom_out result recorded (stream stays well-formed)",
          any(e["event"] == "tool_result" and e["name"] == "zoom_out" for e in convo))


def test_turn_cap_synthesizes_summary() -> None:
    """A small model that never calls zoom_out is cut off at the cap with a
    synthesized summary (and the no-tool-calls nudge is recorded, zoom-tagged)."""
    tmp = Path(tempfile.mkdtemp())
    session = make_session(tmp)
    stub, state = scripted([
        ("thinking hard", []),
        ("still thinking", []),
    ])
    agent._run_turn_with_retry = stub
    agent.get_context_window = lambda *a, **k: None

    summary, usage = _run_zoom(
        FakeSandbox(), session, Toolset(FakeSandbox()), None,
        "claude-haiku-4-5-20251001",
        {"prompt": "Use only simp lemmas.", "file": "Foo.lean", "line": 3},
        inner_max_turns=2, wip_patch_path=None,
    )
    check("cap: aborted summary", "zoom_in aborted" in summary, summary)
    check("cap: mentions the cap", "2 turns" in summary, summary)
    check("cap: carries last text", "still thinking" in summary, summary)
    check("cap: exactly 2 turns ran", state["n"] == 2, str(state["n"]))

    events = read_events(session)
    nudges = [e for e in events
              if e["event"] == "turn" and e["role"] == "user"
              and "zoom_out" in e.get("content", "")
              and e is not events[1]]  # skip the initial prompt
    check("cap: nudges recorded and zoom-tagged",
          len(nudges) >= 1 and all(e.get("zoom") for e in nudges),
          json.dumps(nudges))


def test_siblings_after_zoom_out_dropped() -> None:
    """Tool calls issued alongside (after) zoom_out in the same turn are
    dropped, not dispatched or recorded."""
    tmp = Path(tempfile.mkdtemp())
    session = make_session(tmp)
    stub, _state = scripted([
        ("", [tc("zoom_out", {"summary": "done"}, "t1"),
              tc("bash", {"command": "rm -rf /"}, "t2")]),
    ])
    agent._run_turn_with_retry = stub
    agent.get_context_window = lambda *a, **k: None

    summary, _usage = _run_zoom(
        FakeSandbox(), session, Toolset(FakeSandbox()), None,
        "claude-haiku-4-5-20251001",
        {"prompt": "Use only simp lemmas.", "file": "Foo.lean", "line": 3},
        inner_max_turns=5, wip_patch_path=None,
    )
    check("sibling: summary returned", summary == "done", summary)
    events = read_events(session)
    check("sibling: bash never recorded",
          not any(e.get("name") == "bash" for e in events),
          json.dumps(events))


def test_schema_advertisement() -> None:
    """zoom_in only for the outer (big) model, zoom_out only for the inner
    (small) model, neither by default."""
    ts = Toolset(FakeSandbox())
    default = {t["name"] for t in ts.schemas()}
    outer = {t["name"] for t in ts.schemas(zoom="outer")}
    inner = {t["name"] for t in ts.schemas(zoom="inner")}
    check("default has no zoom tools",
          "zoom_in" not in default and "zoom_out" not in default)
    check("outer has zoom_in only",
          "zoom_in" in outer and "zoom_out" not in outer)
    check("inner has zoom_out only",
          "zoom_out" in inner and "zoom_in" not in inner)
    check("outer otherwise matches default", outer - {"zoom_in"} == default)
    check("inner otherwise matches default", inner - {"zoom_out"} == default)


def main() -> None:
    test_zoom_out_returns_summary()
    test_turn_cap_synthesizes_summary()
    test_siblings_after_zoom_out_dropped()
    test_schema_advertisement()
    print("\nAll zoom tests passed.")


if __name__ == "__main__":
    main()
