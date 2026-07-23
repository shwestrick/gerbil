"""Tests for resume.parse_session -- rebuilding a conversation from a session log.

Pure parsing, no Docker. Run with: uv run python tests/test_resume.py
"""

import json
import tempfile
from pathlib import Path

from gerbil.resume import parse_session


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


def write_log(events: list[dict], trailing: str = "") -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "gerbil-test.jsonl"
    with p.open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        if trailing:
            f.write(trailing)  # a partial/garbled final line (simulated crash)
    return p


START = {
    "event": "session_start", "model": "claude-sonnet-4-6",
    "base_commit": "abc123def456", "project_dir": "/repo", "prompt_file": "/p.md",
    "gerbil_version": "v1",
}
USER = {"event": "turn", "role": "user", "content": "Prove foo."}


def test_normal_crash_at_turn_boundary() -> None:
    """The common case: crash during the next LLM call, log ends on a tool_result.
    A truncated final line must be tolerated."""
    p = write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "Looking."},
        {"event": "tool_call", "name": "bash", "args": {"command": "ls"}},
        {"event": "tool_result", "name": "bash", "result": "Foo.lean"},
        {"event": "turn", "role": "assistant", "content": "Editing."},
        {"event": "tool_call", "name": "write_file",
         "args": {"path": "Foo.lean", "content": "x"}},
        {"event": "tool_result", "name": "write_file", "result": "wrote 1 byte"},
    ], trailing='{"event":"turn","role":"assist')

    ps = parse_session(p)
    check("model recovered", ps.model == "claude-sonnet-4-6")
    check("base recovered", ps.base_commit == "abc123def456")
    check("prompt recovered", ps.prompt == "Prove foo.")
    check("not complete", ps.complete is False)
    check("ends on user tool_result", ps.messages[-1]["role"] == "user")

    # tool_use ids and tool_result ids line up
    asst = ps.messages[1]
    call_id = next(c["id"] for c in asst["content"] if c["type"] == "tool_call")
    res_id = ps.messages[2]["content"][0]["tool_use_id"]
    check("ids match across call/result", call_id == res_id, f"{call_id} vs {res_id}")
    check("replay carries 7 events", len(ps.events) == 7, str(len(ps.events)))


def test_dangling_tool_call() -> None:
    """Crash mid-tool-execution: a tool_call with no recorded result must get a
    synthesized result so the conversation is API-valid."""
    p = write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "calling"},
        {"event": "tool_call", "name": "bash", "args": {"command": "lake build"}},
    ])
    ps = parse_session(p)
    check("dangling -> ends on user", ps.messages[-1]["role"] == "user")
    res = ps.messages[-1]["content"][0]
    check("dangling result is synthesized", "not recorded" in res["content"])
    check("dangling not complete", ps.complete is False)


def test_already_complete() -> None:
    """A session that ended on the model (no pending tools) is 'complete'."""
    p = write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "done"},
        {"event": "session_end", "total_usage": {}},
    ])
    ps = parse_session(p)
    check("complete flag set", ps.complete is True)
    check("complete ends on assistant", ps.messages[-1]["role"] == "assistant")


def test_thought_signature_roundtrip() -> None:
    """A logged Gemini thought_signature must survive into the rebuilt tool_call
    part, and the agent's extractor must produce the base64 that gets logged."""
    import base64

    from gerbil.agent import _thought_sig

    # Extractor: pulls base64 from a raw_part's thought_signature; None otherwise.
    class FakePart:
        thought_signature = b"\x01\x02\xff"

    sig_b64 = _thought_sig(FakePart())
    check("extractor encodes signature", sig_b64 == base64.b64encode(b"\x01\x02\xff").decode())
    check("extractor None for no raw_part", _thought_sig(None) is None)

    # Parser: carries the logged signature onto the tool_call message part.
    p = write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "calling"},
        {"event": "tool_call", "name": "bash", "args": {"command": "ls"},
         "thought_signature": sig_b64},
        {"event": "tool_result", "name": "bash", "result": "ok"},
    ])
    ps = parse_session(p)
    call = next(c for c in ps.messages[1]["content"] if c["type"] == "tool_call")
    check("parser carries signature", call.get("thought_signature") == sig_b64)


def test_ralph_metadata() -> None:
    """A ralph session's chain metadata round-trips through the parser; a plain
    session reports no ralph metadata."""
    ralph_start = dict(START)
    ralph_start["ralph"] = {
        "iteration": 3, "total": 5, "chain_base": "abc123def456",
        "ancestors": ["gerbil-x-01.patch", "gerbil-x-02.patch"],
    }
    p = write_log([
        ralph_start, USER,
        {"event": "turn", "role": "assistant", "content": "work"},
        {"event": "tool_call", "name": "bash", "args": {"command": "ls"}},
        {"event": "tool_result", "name": "bash", "result": "ok"},
    ])
    ps = parse_session(p)
    check("ralph metadata present", ps.ralph is not None)
    check("ralph iteration", ps.ralph["iteration"] == 3)
    check("ralph total", ps.ralph["total"] == 5)
    check("ralph chain_base", ps.ralph["chain_base"] == "abc123def456")
    check("ralph ancestors", ps.ralph["ancestors"] == ["gerbil-x-01.patch", "gerbil-x-02.patch"])

    plain = parse_session(write_log([START, USER]))
    check("non-ralph has no metadata", plain.ralph is None)


def test_ralph_done_script_roundtrip() -> None:
    """The --ralph_done check script is recovered from the log so a resumed ralph
    chain keeps its termination check; absent when the run didn't use one."""
    script = "#!/usr/bin/env bash\nlake build && ! grep -rq sorry .\n"
    with_script = dict(START)
    with_script["ralph_done_script"] = script
    ps = parse_session(write_log([with_script, USER]))
    check("ralph_done script recovered", ps.ralph_done_script == script)

    plain = parse_session(write_log([START, USER]))
    check("no ralph_done script when unused", plain.ralph_done_script is None)


def test_include_session_roundtrip() -> None:
    """The session-log setting (include_session; off via --omit-session-log) is
    recovered from the log so a resume inherits it. Defaults to False for an
    absent field (older logs, which predate the setting and never folded it in)."""
    on = dict(START)
    on["include_session"] = True
    check("include_session recovered", parse_session(write_log([on, USER])).include_session is True)

    off = dict(START)
    off["include_session"] = False
    check("include_session false honored", parse_session(write_log([off, USER])).include_session is False)

    # An older log without the field parses as False (no crash).
    check("include_session defaults False when absent",
          parse_session(write_log([START, USER])).include_session is False)


def test_commit_message_extraction() -> None:
    """The generated commit message is recovered from the commit-message
    exchange; empty when the session crashed before that phase."""
    done = parse_session(write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "did it"},
        {"event": "turn", "role": "user",
         "content": "The task is complete. Here is the final git diff of all your "
                    "changes:\n\ndiff --git ..."},
        {"event": "turn", "role": "assistant",
         "content": "Fix the widget\n\nThe widget was broken; now it isn't."},
    ]))
    check("commit message recovered",
          done.commit_message == "Fix the widget\n\nThe widget was broken; now it isn't.")

    crashed = parse_session(write_log([
        START, USER,
        {"event": "turn", "role": "assistant", "content": "working"},
        {"event": "tool_call", "name": "bash", "args": {"command": "ls"}},
    ]))
    check("no commit message when crashed early", crashed.commit_message == "")


def test_missing_session_start() -> None:
    p = write_log([USER])
    try:
        parse_session(p)
        check("missing session_start raises", False)
    except ValueError:
        check("missing session_start raises", True)


def main() -> None:
    test_normal_crash_at_turn_boundary()
    test_dangling_tool_call()
    test_already_complete()
    test_thought_signature_roundtrip()
    test_ralph_metadata()
    test_ralph_done_script_roundtrip()
    test_include_session_roundtrip()
    test_commit_message_extraction()
    test_missing_session_start()
    print("\nAll resume parser tests passed.")


if __name__ == "__main__":
    main()
