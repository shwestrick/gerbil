"""Tests for big-small mode's resume path -- parsing logs with zoom-tagged
(inner sub-session) events, and continuing a mid-zoom crash.

Pure/fast: parsing plus a stubbed run_session continuation. No network, no
Docker. Run with: uv run python tests/test_zoom_resume.py
"""

import json
import tempfile
from pathlib import Path

from gerbil.resume import parse_session


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


def write_log(events: list[dict]) -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "gerbil-test.jsonl"
    with p.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return p


START = {
    "event": "session_start", "model": "claude-opus-4-8",
    "base_commit": "abc123def456", "project_dir": "/repo", "prompt_file": "/p.md",
    "gerbil_version": "v1", "include_session": True,
    "small_model": "claude-haiku-4-5-20251001", "inner_max_turns": 7,
}
USER = {"event": "turn", "role": "user", "content": "Prove foo."}

# The outer model reads a file, then zooms in.
OUTER_PREFIX = [
    START, USER,
    {"event": "turn", "role": "assistant", "content": "Delegating."},
    {"event": "tool_call", "name": "read_file", "args": {"path": "Foo.lean"}},
    {"event": "tool_result", "name": "read_file", "result": "theorem ..."},
    {"event": "tool_call", "name": "zoom_in",
     "args": {"prompt": "Use only simp lemmas.", "file": "Foo.lean", "line": 3}},
]

# A complete inner sub-session (all zoom-tagged), ending in zoom_out.
INNER_FULL = [
    {"event": "turn", "zoom": True, "role": "user", "content": "Fix Foo.lean:3."},
    {"event": "turn", "zoom": True, "role": "assistant", "content": "trying"},
    {"event": "tool_call", "zoom": True, "name": "edit_file",
     "args": {"path": "Foo.lean", "old_string": "sorry", "new_string": "rfl"}},
    {"event": "tool_result", "zoom": True, "name": "edit_file",
     "result": "edited Foo.lean"},
    {"event": "turn", "zoom": True, "role": "assistant", "content": "done"},
    {"event": "tool_call", "zoom": True, "name": "zoom_out",
     "args": {"summary": "replaced the sorry with rfl"}},
    {"event": "tool_result", "zoom": True, "name": "zoom_out",
     "result": "[zoom_out received; sub-session ended]"},
]


def test_completed_zoom() -> None:
    """A finished zoom: the outer conversation is rebuilt from untagged events
    alone (the inner ones are skipped), the zoom_in call/result pair up, and
    nothing is pending."""
    p = write_log(OUTER_PREFIX + INNER_FULL + [
        {"event": "tool_result", "name": "zoom_in",
         "result": "replaced the sorry with rfl"},
        {"event": "turn", "role": "assistant", "content": "All done."},
        {"event": "session_end", "total_usage": {}},
    ])
    ps = parse_session(p)
    check("small_model recovered", ps.small_model == "claude-haiku-4-5-20251001")
    check("inner_max_turns recovered", ps.inner_max_turns == 7)
    check("no pending zoom", ps.pending_zoom is None)
    check("complete", ps.complete is True)

    # Outer messages: user, assistant(read_file+zoom_in calls), user(results),
    # assistant. No trace of the inner conversation.
    texts = json.dumps(ps.messages)
    check("inner events not in outer messages", "rfl" not in json.dumps(
        [m for m in ps.messages if m["role"] == "assistant"]))
    check("outer rebuild skips inner turns", len(ps.messages) == 4,
          f"{len(ps.messages)}: {texts[:400]}")
    asst = ps.messages[1]["content"]
    zoom_call = next(c for c in asst if c["type"] == "tool_call" and c["name"] == "zoom_in")
    results = ps.messages[2]["content"]
    zoom_res = next(r for r in results if r["tool_name"] == "zoom_in")
    check("zoom_in call/result ids pair up",
          zoom_call["id"] == zoom_res["tool_use_id"],
          f"{zoom_call['id']} vs {zoom_res['tool_use_id']}")
    check("replay carries inner events too",
          sum(1 for e in ps.events if e.get("zoom")) == len(INNER_FULL),
          str(len(ps.events)))


def test_crash_mid_inner() -> None:
    """A crash while the small model is working: pending_zoom carries the call
    id, the sorry position, the earlier same-turn result, and the rebuilt inner
    conversation."""
    inner_partial = INNER_FULL[:4]  # user, assistant, edit_file call+result
    p = write_log(OUTER_PREFIX + inner_partial)
    ps = parse_session(p)
    pz = ps.pending_zoom
    check("mid-inner: pending_zoom set", pz is not None)
    check("mid-inner: not complete", ps.complete is False)
    check("mid-inner: args carry the sorry position",
          pz["args"] == {"prompt": "Use only simp lemmas.",
                         "file": "Foo.lean", "line": 3}, str(pz["args"]))
    check("mid-inner: no summary yet", pz["summary"] is None)

    # Outer messages end on the assistant turn whose zoom_in is unanswered.
    check("mid-inner: outer ends on assistant",
          ps.messages[-1]["role"] == "assistant")
    asst = ps.messages[-1]["content"]
    zoom_call = next(c for c in asst if c["type"] == "tool_call" and c["name"] == "zoom_in")
    check("mid-inner: call_id matches the pending zoom_in part",
          pz["call_id"] == zoom_call["id"], f"{pz['call_id']} vs {zoom_call['id']}")

    # The read_file result from the same turn is held, not flushed.
    held = pz["held_results"]
    check("mid-inner: held_results carries the same-turn result",
          len(held) == 1 and held[0]["tool_name"] == "read_file", json.dumps(held))
    read_call = next(c for c in asst if c["type"] == "tool_call" and c["name"] == "read_file")
    check("mid-inner: held result id pairs with its call",
          held[0]["tool_use_id"] == read_call["id"])

    # Inner conversation rebuilt: user prompt, assistant, user (edit result).
    inner = pz["messages"]
    check("mid-inner: inner conversation rebuilt", len(inner) == 3,
          json.dumps(inner)[:400])
    check("mid-inner: inner starts on its task prompt",
          inner[0]["role"] == "user" and "Foo.lean:3" in inner[0]["content"])
    check("mid-inner: inner ends on the tool result",
          inner[-1]["role"] == "user"
          and inner[-1]["content"][0]["tool_name"] == "edit_file")


def test_crash_after_inner_zoom_out() -> None:
    """A crash in the window between the small model's zoom_out and the outer
    zoom_in result being recorded: the summary is recovered, nothing re-runs."""
    p = write_log(OUTER_PREFIX + INNER_FULL)  # no outer zoom_in result
    ps = parse_session(p)
    pz = ps.pending_zoom
    check("post-zoom_out: pending_zoom set", pz is not None)
    check("post-zoom_out: summary recovered",
          pz["summary"] == "replaced the sorry with rfl", str(pz))


def test_crash_right_after_zoom_in_call() -> None:
    """A crash before the sub-session recorded anything: pending_zoom with an
    empty inner conversation, meaning 'start the zoom fresh'."""
    p = write_log(OUTER_PREFIX)  # log ends on the zoom_in tool_call itself
    ps = parse_session(p)
    pz = ps.pending_zoom
    check("fresh-zoom: pending_zoom set", pz is not None)
    check("fresh-zoom: inner conversation empty", pz["messages"] == [])
    check("fresh-zoom: no summary", pz["summary"] is None)
    check("fresh-zoom: not complete", ps.complete is False)


def test_plain_log_unaffected() -> None:
    """A log without any zoom events parses exactly as before."""
    p = write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "working"},
        {"event": "tool_call", "name": "bash", "args": {"command": "ls"}},
        {"event": "tool_result", "name": "bash", "result": "Foo.lean"},
    ])
    ps = parse_session(p)
    check("plain: no pending zoom", ps.pending_zoom is None)
    check("plain: ends on user", ps.messages[-1]["role"] == "user")


def test_mid_zoom_continuation() -> None:
    """End-to-end: run_session picks up a mid-zoom crash -- finishes the inner
    conversation first (seeded), then continues the outer one with the summary
    as the zoom_in result."""
    import gerbil.agent as agent
    from gerbil.providers import Usage
    from gerbil.session import Session
    from gerbil.tools import Toolset

    inner_partial = INNER_FULL[:4]
    ps = parse_session(write_log(OUTER_PREFIX + inner_partial))

    class FakeSandbox:
        def read_file(self, path):
            return ""
        def get_diff(self):
            return ""
        def diff_since(self, base):
            return ""

    calls = []

    def stub(model, system, messages, tools, provider, read_file=None, session=None):
        calls.append({"model": model, "last_role": messages[-1]["role"],
                      "n_messages": len(messages)})
        if model == "claude-haiku-4-5-20251001":
            tcs = [{"name": "zoom_out", "args": {"summary": "finished it"},
                    "id": "z1", "raw_part": None}]
            parts = [{"type": "tool_call", **{k: tcs[0][k] for k in ("name", "args", "id", "raw_part")}}]
            return parts, tcs, "", Usage(input_tokens=1, output_tokens=1)
        return ([{"type": "text", "text": "great, done"}], [], "great, done",
                Usage(input_tokens=1, output_tokens=1))

    agent._run_turn_with_retry = stub
    agent.get_context_window = lambda *a, **k: None

    tmp = Path(tempfile.mkdtemp())
    session = Session(
        path=tmp / "cont.jsonl", model="claude-opus-4-8",
        project_dir=Path("/repo"), prompt_file=Path("/p.md"),
        small_model="claude-haiku-4-5-20251001", inner_max_turns=7,
    )
    result = agent.run_session(
        FakeSandbox(), session, ps.prompt, "claude-opus-4-8",
        Toolset(FakeSandbox()), messages=ps.messages,
        small_model="claude-haiku-4-5-20251001", inner_max_turns=7,
        pending_zoom=ps.pending_zoom,
    )

    check("continuation: inner ran first, then outer",
          [c["model"] for c in calls]
          == ["claude-haiku-4-5-20251001", "claude-opus-4-8"], str(calls))
    # The inner stub was seeded with the rebuilt 3-message inner conversation.
    check("continuation: inner conversation was seeded",
          calls[0]["n_messages"] == 3, str(calls[0]))
    check("continuation: outer resumed past the zoom", result.final_text == "great, done")

    events = [json.loads(l) for l in session.path.read_text().splitlines()]
    zoom_in_results = [e for e in events
                       if e["event"] == "tool_result" and e["name"] == "zoom_in"]
    check("continuation: zoom_in result recorded, untagged",
          len(zoom_in_results) == 1 and not zoom_in_results[0].get("zoom"),
          json.dumps(zoom_in_results))
    check("continuation: zoom_in result is the summary",
          zoom_in_results[0]["result"] == "finished it")

    # The outer model's next turn saw held_results + zoom_in result together
    # (run_session mutates ps.messages in place; find that user message).
    user_after = next(
        m for m in ps.messages
        if m["role"] == "user" and isinstance(m["content"], list)
        and any(r.get("tool_name") == "zoom_in" for r in m["content"])
    )
    check("continuation: results share one user message",
          len(user_after["content"]) == 2
          and {r["tool_name"] for r in user_after["content"]}
          == {"read_file", "zoom_in"},
          json.dumps(user_after)[:400])


def test_multi_zoom_seed_isolation() -> None:
    """A log can hold many sub-sessions; the pending_zoom seed must contain
    ONLY the interrupted (last) one -- its assistant count is what _run_zoom
    charges against the turn cap on resume, so earlier zooms leaking in would
    shrink (or exhaust) the resumed budget."""

    def zoom(n_turns, complete):
        ev = [{"event": "tool_call", "name": "zoom_in",
               "args": {"prompt": "p", "file": "F.lean", "line": 1}},
              {"event": "turn", "zoom": True, "role": "user", "content": "task"}]
        for i in range(n_turns):
            ev += [
                {"event": "turn", "zoom": True, "role": "assistant", "content": f"t{i}"},
                {"event": "tool_call", "zoom": True, "name": "bash",
                 "args": {"command": "x"}},
                {"event": "tool_result", "zoom": True, "name": "bash", "result": "ok"},
            ]
        if complete:
            ev += [
                {"event": "tool_call", "zoom": True, "name": "zoom_out",
                 "args": {"summary": "done"}},
                {"event": "tool_result", "zoom": True, "name": "zoom_out",
                 "result": "[ended]"},
                {"event": "tool_result", "name": "zoom_in", "result": "done"},
            ]
        return ev

    def n_assistant(msgs):
        return sum(1 for m in msgs if m.get("role") == "assistant")

    # Two completed zooms, then a third interrupted after 3 inner turns.
    events = ([START, USER, {"event": "turn", "role": "assistant", "content": "z1"}]
              + zoom(2, complete=True)
              + [{"event": "turn", "role": "assistant", "content": "z2"}]
              + zoom(5, complete=True)
              + [{"event": "turn", "role": "assistant", "content": "z3"}]
              + zoom(3, complete=False))
    ps = parse_session(write_log(events))
    check("multi-zoom: seed holds only the interrupted sub-session",
          n_assistant(ps.pending_zoom["messages"]) == 3,
          str(n_assistant(ps.pending_zoom["messages"])))

    # Continuation log (a resume of a resume): its own session_start, the same
    # events replayed, then the interrupted zoom continued live for 4 more
    # turns before crashing again. The seed must span both runs of that one
    # sub-session.
    cont_start = dict(START, resumed_from="gerbil-parent.jsonl")
    replayed = [dict(e, replayed=True) for e in events]
    live = []
    for i in range(4):
        live += [
            {"event": "turn", "zoom": True, "role": "assistant", "content": f"live{i}"},
            {"event": "tool_call", "zoom": True, "name": "bash",
             "args": {"command": "y"}},
            {"event": "tool_result", "zoom": True, "name": "bash", "result": "ok"},
        ]
    ps2 = parse_session(write_log([cont_start] + replayed + live))
    check("multi-zoom: continuation seed spans both runs of the last zoom",
          n_assistant(ps2.pending_zoom["messages"]) == 7,
          str(n_assistant(ps2.pending_zoom["messages"])))


def test_summarize_accounting() -> None:
    """`gerbil summarize` splits a big-small session's usage into the two
    models' buckets and prices each at its own rates; replayed zoom events are
    not double-counted."""
    from gerbil.cli import _cost, _scan_session

    usage_big = {"input_tokens": 100, "output_tokens": 10}
    usage_small = {"input_tokens": 1000, "output_tokens": 100}
    p = write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "zooming",
         "usage": usage_big},
        {"event": "tool_call", "name": "zoom_in",
         "args": {"prompt": "Use only simp lemmas.", "file": "Foo.lean", "line": 3}},
        {"event": "turn", "zoom": True, "role": "user", "content": "fix it"},
        {"event": "turn", "zoom": True, "role": "assistant", "content": "ok",
         "usage": usage_small},
        {"event": "tool_call", "zoom": True, "name": "zoom_out",
         "args": {"summary": "done"}},
        {"event": "tool_result", "zoom": True, "name": "zoom_out",
         "result": "[zoom_out received; sub-session ended]"},
        {"event": "tool_result", "name": "zoom_in", "result": "done"},
        # A replayed inner turn (from a resume) must not count again.
        {"event": "turn", "zoom": True, "role": "assistant", "content": "old",
         "usage": usage_small, "replayed": True},
        {"event": "session_end", "total_usage": {}},
    ])
    s = _scan_session(p)
    check("summarize: outer bucket",
          s["input_tokens"] == 100 and s["output_tokens"] == 10, str(s))
    check("summarize: zoom bucket",
          s["zoom_input_tokens"] == 1000 and s["zoom_output_tokens"] == 100,
          str(s))
    # _scan_session counts assistant turns only (matching the live counters):
    # one outer assistant turn, one zoom assistant turn -- the USER prompt and
    # the zoom task prompt don't count.
    check("summarize: turn split", s["turns"] == 1 and s["zoom_turns"] == 1,
          str(s))
    check("summarize: small model recovered",
          s["small_model"] == "claude-haiku-4-5-20251001")
    check("summarize: zoom tools visible",
          s["tool_calls"]["zoom_in"] == 1 and s["tool_calls"]["zoom_out"] == 1)

    # opus-4-8 at (5, 25) + haiku-4-5 at (1, 5) per Mtok.
    expected = (100 * 5.0 + 10 * 25.0) / 1e6 + (1000 * 1.0 + 100 * 5.0) / 1e6
    c = _cost(s)
    check("summarize: two-model cost", c is not None and abs(c - expected) < 1e-12,
          f"{c} vs {expected}")


def test_summarize_resume_accounting() -> None:
    """A continuation log's replayed events are the only record of a crashed
    parent's spend (the parent never committed, so its log never reached the
    project). Fold them in when the parent log is absent; keep skipping them
    when it is present, so the two rows stay disjoint."""
    from gerbil.cli import _scan_sessions

    usage = {"input_tokens": 10, "output_tokens": 1}
    parent_events = [
        START, USER,
        {"event": "turn", "role": "assistant", "content": "a1", "usage": usage},
        {"event": "turn", "zoom": True, "role": "user", "content": "task"},
        {"event": "turn", "zoom": True, "role": "assistant", "content": "z1",
         "usage": usage},
    ]
    cont_start = dict(START)
    cont_start["resumed_from"] = "gerbil-parent.jsonl"
    # The continuation carries the parent's FULL log -- session_start and the
    # crash's error event included -- then the `resumed` marker, then live work.
    cont_events = (
        [cont_start]
        + [dict(e, replayed=True) for e in parent_events]
        + [{"event": "error", "error_type": "RuntimeError", "message": "boom",
            "replayed": True},
           {"event": "resumed", "resumed_from": "gerbil-parent.jsonl",
            "replayed_events": len(parent_events) + 1},
           {"event": "turn", "role": "assistant", "content": "live",
            "usage": usage},
           {"event": "session_end", "total_usage": {}}]
    )

    d = Path(tempfile.mkdtemp())
    parent = d / "gerbil-parent.jsonl"
    cont = d / "gerbil-parent-resume-x.jsonl"
    for p, evs in ((parent, parent_events), (cont, cont_events)):
        with p.open("w") as f:
            for e in evs:
                f.write(json.dumps(e) + "\n")

    # Parent absent: the continuation's row absorbs the replayed history.
    [folded] = _scan_sessions([cont])
    check("resume accounting: parent absent folds replay in",
          folded["turns"] == 2 and folded["zoom_turns"] == 1, str(folded))
    check("resume accounting: replayed tokens counted",
          folded["input_tokens"] == 20 and folded["zoom_input_tokens"] == 10,
          str(folded))
    # Folding replayed spend in must not let the parent's replayed
    # session_start/error speak for this row: status comes from the live
    # session_end, not the parent's crash.
    check("resume accounting: replayed error does not mark the row errored",
          folded["status"] == "completed", folded["status"])

    # Parent present: rows stay disjoint (no double count).
    both = _scan_sessions([parent, cont])
    check("resume accounting: parent present keeps rows disjoint",
          both[0]["turns"] == 1 and both[0]["zoom_turns"] == 1
          and both[1]["turns"] == 1 and both[1]["zoom_turns"] == 0,
          str(both))


def main() -> None:
    test_completed_zoom()
    test_crash_mid_inner()
    test_crash_after_inner_zoom_out()
    test_crash_right_after_zoom_in_call()
    test_plain_log_unaffected()
    test_mid_zoom_continuation()
    test_multi_zoom_seed_isolation()
    test_summarize_accounting()
    test_summarize_resume_accounting()
    print("\nAll zoom resume tests passed.")


if __name__ == "__main__":
    main()
