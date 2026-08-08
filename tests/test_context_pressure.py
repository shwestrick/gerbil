#!/usr/bin/env python3
"""Context-exhaustion guards (no network, no Docker, no API key).

A session whose conversation fills the model's context window used to die on a
provider error, losing everything it hadn't committed. gerbil now winds the
session down in three steps -- advise at 75%, order at 85%, take over at 95% --
and the last one still spends the remaining room on a commit message so the
work lands explained.

What is checked here: the thresholds fire once each and in order, the note
reaches the conversation in a shape every provider actually transmits (the part
easiest to get subtly wrong -- a user message carrying tool results has strict
rules), and the forced ending still produces a commit message.

Run: uv run python tests/test_context_pressure.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gerbil import prompts  # noqa: E402
from gerbil.agent import (  # noqa: E402
    _append_user_text,
    _commit_diff,
    _context_level,
)
from gerbil.prompts import (  # noqa: E402
    CONTEXT_TERMINAL,
    CONTEXT_URGENT,
    CONTEXT_WIND_DOWN,
    context_pressure_note,
)
from gerbil.providers import Usage  # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
failures = []


def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


def usage_at(pct, limit=100_000):
    """A Usage whose context_tokens are `pct` of `limit`, spread across the
    fields that occupy the window -- cached prompt tokens included, which is the
    part a naive `input_tokens` check would miss."""
    used = int(limit * pct)
    return Usage(
        input_tokens=used // 4,
        cache_read_tokens=used // 4,
        cache_write_tokens=used // 4,
        output_tokens=used - 3 * (used // 4),
    )


def test_thresholds():
    print("\n-- thresholds --")
    check("ordered 75 < 85 < 95",
          CONTEXT_WIND_DOWN < CONTEXT_URGENT < CONTEXT_TERMINAL)
    check("terminal leaves room to finish", CONTEXT_TERMINAL < 1.0)

    check("cached tokens count toward the window",
          usage_at(0.80, 100_000).context_tokens >= 79_000,
          str(usage_at(0.80).context_tokens))

    for pct, expected in [
        (0.10, None), (0.74, None),
        (0.75, CONTEXT_WIND_DOWN), (0.80, CONTEXT_WIND_DOWN),
        (0.85, CONTEXT_URGENT), (0.94, CONTEXT_URGENT),
        (0.95, CONTEXT_TERMINAL), (1.20, CONTEXT_TERMINAL),
    ]:
        got = _context_level(usage_at(pct), 100_000)
        check(f"{pct:.0%} -> {expected}", got == expected, f"got {got}")

    print("\n-- no window, no guard --")
    # Every provider that won't report a context window must behave exactly as
    # it did before these guards existed.
    check("unknown window never fires", _context_level(usage_at(0.99), None) is None)
    check("no usage yet never fires", _context_level(None, 100_000) is None)


def test_notes():
    print("\n-- the notes escalate --")
    notes = {
        level: context_pressure_note(level, int(100_000 * level), 100_000)
        for level in (CONTEXT_WIND_DOWN, CONTEXT_URGENT, CONTEXT_TERMINAL)
    }
    for level, note in notes.items():
        check(f"{level:.0%} states the numbers", "100,000" in note and "%" in note)
    check("75% advises", "wrapping up" in notes[CONTEXT_WIND_DOWN].lower())
    check("85% orders", "MUST wrap up this session NOW" in notes[CONTEXT_URGENT])
    check("95% announces the end", "session is over" in notes[CONTEXT_TERMINAL].lower())
    check("all three differ", len(set(notes.values())) == 3)


def test_message_shape():
    print("\n-- the note joins the pending tool-result message --")
    # The providers reject two user messages in a row, and require an assistant
    # turn's tool calls to be answered immediately -- so the note has to ride
    # inside the tool-result message, not follow it.
    messages = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": [{"type": "tool_call", "id": "t1",
                                           "name": "bash", "args": {}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_name": "bash", "content": "ok",
             "tool_use_id": "t1", "tool_call_id": "t1"},
        ]},
    ]
    before = len(messages)
    _append_user_text(messages, "NOTE")
    check("no new message is appended", len(messages) == before)
    check("the note lands as a text item",
          messages[-1]["content"][-1] == {"type": "text", "text": "NOTE"})
    check("the tool result is untouched",
          messages[-1]["content"][0]["type"] == "tool_result")
    check("roles still alternate",
          [m["role"] for m in messages] == ["user", "assistant", "user"])

    # After an assistant turn (the ordinary end-of-session path) it becomes its
    # own message, as before.
    messages2 = [{"role": "assistant", "content": [{"type": "text", "text": "done"}]}]
    _append_user_text(messages2, "commit message please")
    check("after an assistant turn it is a new user message",
          len(messages2) == 2 and messages2[-1] == {"role": "user",
                                                    "content": "commit message please"})

    # A plain-string user message merges textually.
    messages3 = [{"role": "user", "content": "first"}]
    _append_user_text(messages3, "second")
    check("string content merges", messages3 == [{"role": "user",
                                                  "content": "first\n\nsecond"}])
    check("empty conversation is fine",
          (lambda m: (_append_user_text(m, "x"), m)[1])([]) == [{"role": "user", "content": "x"}])


def test_providers_transmit_the_note():
    print("\n-- every provider actually sends it --")
    # A text item riding with tool results is new to the unified format; a
    # converter that drops it would disable the guard silently, which is the
    # worst possible failure for this feature.
    messages = [
        {"role": "assistant", "content": [{"type": "tool_call", "id": "t1",
                                           "name": "bash", "args": {"cmd": "ls"}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_name": "bash", "content": "output",
             "tool_use_id": "t1", "tool_call_id": "t1"},
            {"type": "text", "text": "PRESSURE-NOTE"},
        ]},
    ]

    # OpenAI: tool messages must answer the assistant turn first, then the note
    # as a plain user message.
    import json as _json

    captured = {}

    class StubStream:
        def __iter__(self):
            return iter(())

    class StubCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return StubStream()

    class StubClient:
        chat = type("C", (), {"completions": StubCompletions()})()

    import contextlib

    from gerbil import providers

    # The stubs yield no chunks, which the empty-turn guard rightly rejects.
    # Only the request that was built matters here.
    with contextlib.suppress(providers.TransientProviderError):
        list(providers._stream_openai_chat(StubClient(), "m", "sys", messages, []))
    sent = captured.get("messages", [])
    roles = [m["role"] for m in sent]
    check("openai keeps the note", any("PRESSURE-NOTE" in str(m.get("content"))
                                       for m in sent), _json.dumps(sent))
    check("openai puts the tool reply before it",
          roles == ["system", "assistant", "tool", "user"], str(roles))

    # Anthropic: passes non-tool_result items through, so the note stays in the
    # same user message, after the tool_result block.
    captured_a = {}

    class StubCtx:
        def __enter__(self):
            return iter(())

        def __exit__(self, *a):
            return False

    class StubMessages:
        def stream(self, **kwargs):
            captured_a.update(kwargs)
            return StubCtx()

    class StubAnthropicClient:
        messages = StubMessages()

    with contextlib.suppress(providers.TransientProviderError):
        list(providers._stream_anthropic_chat(
            StubAnthropicClient(), "m", "sys", messages, []))
    content = captured_a["messages"][-1]["content"]
    check("anthropic keeps the note",
          any(c.get("type") == "text" and "PRESSURE-NOTE" in c.get("text", "")
              for c in content), str(content))
    check("anthropic sends the tool_result first",
          content[0]["type"] == "tool_result", str(content))


def test_commit_diff_budget():
    print("\n-- the forced ending still fits a commit message --")
    diff = "\n".join(f"+line {i}" for i in range(20_000))  # ~200k chars
    check("plenty of room -> untouched",
          _commit_diff(diff, 1_000_000, usage_at(0.10, 1_000_000)) == diff)
    check("unknown window -> untouched", _commit_diff(diff, None, usage_at(0.99)) == diff)

    # At the terminal threshold there are ~5,000 tokens left of a 100k window.
    tight = _commit_diff(diff, 100_000, usage_at(0.95))
    check("tight room -> shortened", len(tight) < len(diff), f"{len(tight)} vs {len(diff)}")
    check("...to something that fits", len(tight) < 5_000 * 3.1, str(len(tight)))
    check("...keeping head and tail",
          "+line 0" in tight and "+line 19999" in tight)

    # Past the threshold entirely (the window overshot): still asks, minimally.
    over = _commit_diff(diff, 100_000, usage_at(1.10))
    check("no room left -> minimal diff, not empty", 0 < len(over) <= 2_200, str(len(over)))


class FakeSandbox:
    """The bits of LeanSandbox run_session touches."""

    submodule_paths: list[str] = []

    def read_file(self, path):
        return ""

    def diff_since(self, base):
        return "--- a/Foo.lean\n+++ b/Foo.lean\n+theorem foo : True := trivial\n"

    def get_diff(self):
        return self.diff_since("")

    def wip_patch(self, base):
        return ""


def test_session_escalates_and_stops():
    print("\n-- a whole session under context pressure --")
    import json
    import tempfile

    import gerbil.agent as agent
    from gerbil.session import Session
    from gerbil.tools import Toolset

    # A session whose context fills steadily: each turn reports a larger share
    # of a 100k window, crossing 75%, 85% and then 95%.
    percentages = [0.50, 0.76, 0.80, 0.86, 0.90, 0.96, 0.99]
    state = {"n": 0}
    real_turn = agent._run_turn_with_retry
    real_window = agent.get_context_window

    def stub(model, system, messages, tools, provider, read_file=None, session=None):
        i = min(state["n"], len(percentages) - 1)
        state["n"] += 1
        usage = usage_at(percentages[i])
        if not tools:  # the commit-message turn: no tools offered
            return [{"type": "text", "text": "Fix foo\n\nBody."}], [], "Fix foo\n\nBody.", usage
        call = {"name": "bash", "args": {"command": "ls"}, "id": f"t{i}",
                "raw_part": None}
        parts = [{"type": "text", "text": "working"},
                 {"type": "tool_call", "name": "bash", "args": {"command": "ls"},
                  "id": f"t{i}", "raw_part": None}]
        return parts, [call], "working", usage

    class FakeToolset(Toolset):
        def __init__(self):
            super().__init__(FakeSandbox())

        def dispatch(self, name, args):
            from gerbil.tools import ToolResult

            return ToolResult(content="ok", is_error=False)

    tmp = Path(tempfile.mkdtemp())
    session = Session(path=tmp / "s.jsonl", model="claude-opus-4-8",
                      project_dir=Path("/repo"), prompt_file=Path("/p.md"))
    agent._run_turn_with_retry = stub
    agent.get_context_window = lambda *a, **k: 100_000
    try:
        result = agent.run_session(
            FakeSandbox(), session, "do the task", "claude-opus-4-8",
            FakeToolset(), provider="anthropic", max_turns=50,
        )
    finally:
        agent._run_turn_with_retry = real_turn
        agent.get_context_window = real_window

    events = [json.loads(line) for line in session.path.read_text().splitlines()]
    notes = [e["message"] for e in events if e.get("event") == "warning"]

    check("three notes, one per threshold", len(notes) == 3, str(len(notes)))
    check("in escalating order",
          len(notes) == 3
          and "wrapping up" in notes[0].lower()
          and "MUST wrap up this session NOW" in notes[1]
          and "session is over" in notes[2].lower(),
          str(notes))
    # 0.96 crosses the terminal threshold on turn 6; the 0.99 entry must never
    # be reached, and no turn may run after the break except the commit message.
    check("the session stopped at the terminal threshold", state["n"] == 7,
          f"{state['n']} turns ran")
    check("the commit message was still produced",
          result.commit_message == "Fix foo\n\nBody.", repr(result.commit_message))

    # The forced ending must leave a conversation the providers would accept.
    check("the log records the commit request",
          any(e.get("event") == "turn" and e.get("role") == "user"
              and "Write a git commit message" in e.get("content", "")
              for e in events))


def test_prompt_constants_are_the_spec():
    print("\n-- the documented percentages --")
    check("75%", prompts.CONTEXT_WIND_DOWN == 0.75)
    check("85%", prompts.CONTEXT_URGENT == 0.85)
    check("95%", prompts.CONTEXT_TERMINAL == 0.95)


if __name__ == "__main__":
    test_thresholds()
    test_notes()
    test_message_shape()
    test_providers_transmit_the_note()
    test_commit_diff_budget()
    test_session_escalates_and_stops()
    test_prompt_constants_are_the_spec()
    print()
    if failures:
        print(f"\033[31m{len(failures)} failed:\033[0m " + ", ".join(failures))
        sys.exit(1)
    print("\033[32mall passed\033[0m")
