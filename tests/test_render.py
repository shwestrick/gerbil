"""Unit tests for the cosmetic tool-call rendering in agent.py.

Pure functions, no Docker. term.style() is a no-op off a TTY, so output is plain
text here and we can assert on substrings. Run: uv run python tests/test_render.py
"""

from gerbil.agent import _format_tool_call


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


FILE = "import Mathlib\n\nnamespace Foo\ndef keep := 0\ndef foo := 1\ndef tail := 9\nend Foo\n"


def test_write_file_line_numbers() -> None:
    out = _format_tool_call("write_file", {"path": "A.lean", "content": "a\nb\nc\n"})
    lines = out.splitlines()
    check("write_file numbers from 1", lines[1].strip().startswith("1 a"), out)
    check("write_file numbers increment", lines[3].strip().startswith("3 c"), out)


def test_write_file_summary_when_large() -> None:
    out = _format_tool_call(
        "write_file", {"path": "B.lean", "content": "\n".join(f"l{i}" for i in range(40))}
    )
    check("large write summarized", "lines," in out and "bytes)" in out, out)
    check("large write hides content", "l39" not in out, out)


def test_edit_file_real_line_numbers() -> None:
    args = {
        "path": "Foo.lean",
        "old_string": "def keep := 0\ndef foo := 1\ndef tail := 9",
        "new_string": "def keep := 0\ndef foo := 42\ndef tail := 9",
    }
    out = _format_tool_call("edit_file", args, lambda p: FILE)
    # old_string starts at file line 4; the changed line is file line 5.
    check("edit hunk header uses real lines", "@@ -4,3 +4,3 @@" in out, out)
    check("removed line numbered 5", "5 -def foo := 1" in out, out)
    check("added line numbered 5", "5 +def foo := 42" in out, out)
    check("context line numbered 6", "6  def tail := 9" in out, out)


def test_edit_file_midline_old_string() -> None:
    args = {"path": "Foo.lean", "old_string": "foo := 1", "new_string": "foo := 42"}
    out = _format_tool_call("edit_file", args, lambda p: FILE)
    check("midline old_string resolves to line 5", "5 -foo := 1" in out, out)


def test_edit_file_fallback_without_file() -> None:
    args = {
        "path": "Foo.lean",
        "old_string": "def foo := 1",
        "new_string": "def foo := 42",
    }
    # No reader, or a reader that fails -> fragment-relative (line 1), never crash.
    out_none = _format_tool_call("edit_file", args, None)
    check("no reader -> fragment line 1", "1 -def foo := 1" in out_none, out_none)

    def boom(_):
        raise FileNotFoundError

    out_boom = _format_tool_call("edit_file", args, boom)
    check("reader error -> fragment line 1", "1 -def foo := 1" in out_boom, out_boom)


def test_other_tool_unchanged() -> None:
    out = _format_tool_call("bash", {"command": "lake build"})
    check("bash keeps name(args)", out.strip() == "-> bash({'command': 'lake build'})", out)


def main() -> None:
    test_write_file_line_numbers()
    test_write_file_summary_when_large()
    test_edit_file_real_line_numbers()
    test_edit_file_midline_old_string()
    test_edit_file_fallback_without_file()
    test_other_tool_unchanged()
    print("\nAll render tests passed.")


if __name__ == "__main__":
    main()
