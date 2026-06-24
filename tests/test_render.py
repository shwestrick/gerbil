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


def test_lean_multi_attempt() -> None:
    # Several short snippets: each shown, count in the header, file:line:col.
    out = _format_tool_call("lean_multi_attempt", {
        "file_path": "Foo.lean", "line": 10, "column": 5,
        "snippets": ["simp [foo]", "omega", "ring_nf"],
    })
    check("location is file:line:col", "Foo.lean:10:5" in out, out)
    check("snippet count shown", "(3 snippets)" in out, out)
    check("short snippets shown verbatim",
          "[1] simp [foo]" in out and "[2] omega" in out, out)

    # A large snippet is summarized, not dumped.
    big = "\n".join(f"tac_{i} := by simp" for i in range(41))
    out2 = _format_tool_call("lean_multi_attempt", {
        "file_path": "Big.lean", "line": 1, "column": 1, "snippets": [big],
    })
    check("large snippet summarized", "(41 lines," in out2 and "chars)" in out2, out2)
    check("large snippet not dumped", "tac_40" not in out2, out2)
    check("single snippet -> no count", "snippets)" not in out2, out2)

    # A small multi-line snippet is shown inline.
    out3 = _format_tool_call("lean_multi_attempt", {
        "file_path": "B.lean", "line": 3, "column": 1, "snippets": ["by\n  simp\n  omega"],
    })
    check("small multiline snippet inline", "simp" in out3 and "omega" in out3, out3)


def test_lean_run_code() -> None:
    code = ("import Defunc.Source\nimport Defunc.Lemmas\n\n"
            "lemma foo : True := by trivial\n")
    out = _format_tool_call("lean_run_code", {"code": code})
    check("run_code small shown with line count", "(4 lines)" in out, out)
    check("run_code numbers lines", "1 import Defunc.Source" in out, out)
    check("run_code shows body", "lemma foo" in out, out)

    big = "\n".join(f"def x{i} := {i}" for i in range(60))
    out2 = _format_tool_call("lean_run_code", {"code": big})
    check("run_code large summarized", "(60 lines," in out2 and "chars)" in out2, out2)
    check("run_code large not dumped", "def x59" not in out2, out2)

    out3 = _format_tool_call("lean_run_code", {"code": "#check Nat"})
    check("run_code single line inline", out3.strip() == "-> lean_run_code #check Nat", out3)


def test_lean_goal_position() -> None:
    src = "namespace D\n\ntheorem big (n : Nat) :\n    n + 0 = n := by\n  simp\n"
    out = _format_tool_call(
        "lean_goal", {"file_path": "D.lean", "line": 4, "column": 5}, lambda p: src
    )
    check("location shown", "D.lean:4:5" in out, out)
    check("target source line shown", "n + 0 = n := by" in out, out)
    check("context line shown", "theorem big" in out, out)
    # caret line = 5 (body indent) + 5 (gutter) + (col-1) spaces, then '^'
    caret = next(l for l in out.splitlines() if l.strip() == "^")
    check("caret aligned to column", caret == " " * (5 + 5 + 4) + "^", repr(caret))


def test_lean_goal_fallbacks() -> None:
    out = _format_tool_call(
        "lean_goal", {"file_path": "X.lean", "line": 4, "column": 5}, None
    )
    check("no reader -> just location", out.strip() == "-> lean_goal X.lean:4:5", out)
    out2 = _format_tool_call(
        "lean_goal", {"file_path": "X.lean", "line": 2}, lambda p: "a\nb\nc\n"
    )
    check("no column -> no caret", "^" not in out2 and "b" in out2, out2)
    # lean_term_goal / lean_hover_info use the same renderer.
    out3 = _format_tool_call(
        "lean_term_goal", {"file_path": "X.lean", "line": 1, "column": 1},
        lambda p: "abc\ndef\n",
    )
    check("term_goal also rendered", "X.lean:1:1" in out3 and "^" in out3, out3)


def test_lean_build() -> None:
    plain = _format_tool_call(
        "lean_build", {"fetch_cache": False, "output_lines": 20, "clean": False}
    )
    check("plain build is just the name", plain.strip() == "-> lean_build", plain)
    check("clean flag shown",
          _format_tool_call("lean_build", {"clean": True}).strip()
          == "-> lean_build (clean)")
    check("both flags shown",
          _format_tool_call("lean_build", {"clean": True, "fetch_cache": True}).strip()
          == "-> lean_build (clean, fetch cache)")


def test_lean_diagnostic_messages() -> None:
    out = _format_tool_call("lean_diagnostic_messages", {"file_path": "Foo.lean"})
    check("diagnostics shows just the path",
          out.strip() == "-> lean_diagnostic_messages Foo.lean", out)
    out2 = _format_tool_call(
        "lean_diagnostic_messages", {"file_path": "Foo.lean", "line": 10}
    )
    check("diagnostics appends extra args",
          out2.strip() == "-> lean_diagnostic_messages Foo.lean (line=10)", out2)


def test_other_tool_unchanged() -> None:
    out = _format_tool_call("bash", {"command": "lake build"})
    check("bash keeps name(args)", out.strip() == "-> bash({'command': 'lake build'})", out)


def main() -> None:
    test_write_file_line_numbers()
    test_write_file_summary_when_large()
    test_edit_file_real_line_numbers()
    test_edit_file_midline_old_string()
    test_edit_file_fallback_without_file()
    test_lean_multi_attempt()
    test_lean_run_code()
    test_lean_goal_position()
    test_lean_goal_fallbacks()
    test_lean_build()
    test_lean_diagnostic_messages()
    test_other_tool_unchanged()
    print("\nAll render tests passed.")


if __name__ == "__main__":
    main()
