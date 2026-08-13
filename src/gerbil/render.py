"""All human-readable terminal rendering, in one place.

Everything here is purely cosmetic -- it decides what the *user* sees, never
what is dispatched to a tool, sent to the model, or recorded to the session
log. That invariant is the point of this module: agent.py calls in with the
real data and prints whatever comes back.

Contents:
  - style():           tiny ANSI color helper (respects NO_COLOR / non-TTY)
  - turn_header:       the timestamped rule that opens every turn (box-drawing
                       characters where the terminal can encode them, dashes
                       where it can't)
  - format_tool_call:  pretty one-or-many-line rendering of a tool invocation
  - format_tool_result: the "  <- ..." preview of a tool's result
  - context_suffix / print_usage: the turn-header context gauge and the
    end-of-session usage line
plus the private _render_* helpers they dispatch to.
"""

import difflib
import json
import os
import re
import shutil
import sys
import textwrap
from datetime import datetime

# https://no-color.org/ -- any non-empty NO_COLOR disables color.
# GERBIL_FORCE_STYLE keeps styling on when stdout is not a tty: a detached
# runner's stdout is the display file a live viewer tails and re-renders, so
# the escapes are wanted there. Only gerbil's own spawner sets it (cli.
# _spawn_and_attach); NO_COLOR still wins.
ENABLED = (
    sys.stdout.isatty() or bool(os.environ.get("GERBIL_FORCE_STYLE"))
) and not os.environ.get("NO_COLOR")


_BOX_GLYPHS = {
    "rule": "─", "sep": "·",
    # File-tree connectors (view.render_file_tree): entry, last entry, a
    # continuing ancestor level, a finished one. All the same width so the
    # ASCII layout lines up column for column.
    "tee": "├── ", "corner": "└── ", "pipe": "│   ", "blank": "    ",
}
_ASCII_GLYPHS = {
    "rule": "-", "sep": "|",
    "tee": "|-- ", "corner": "`-- ", "pipe": "|   ", "blank": "    ",
}


def _supports_unicode(stream=None) -> bool:
    """Whether `stream` (default stdout) can carry box-drawing characters.

    Asked of the stream's own encoding rather than guessed from the locale: the
    only thing that matters is whether the write will succeed. A terminal that
    can't (a C/POSIX locale, a legacy code page, a pipe into something ASCII)
    gets the dashed fallback instead of mojibake or a UnicodeEncodeError from a
    print statement. GERBIL_ASCII forces the fallback everywhere."""
    if os.environ.get("GERBIL_ASCII"):
        return False
    encoding = getattr(sys.stdout if stream is None else stream, "encoding", None)
    if not encoding:
        return False
    try:
        for glyph in _BOX_GLYPHS.values():
            glyph.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


UNICODE = _supports_unicode()
GLYPHS = _BOX_GLYPHS if UNICODE else _ASCII_GLYPHS

# Escape sequences style() may have inserted, for measuring a line's *visible*
# width. Padding math must not count them -- they occupy no columns.
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")

# Below this, a header just puts its pieces next to each other; padding a rule
# out to a sliver of a terminal looks worse than not trying.
_MIN_RULE_WIDTH = 40


def _visible_len(text: str) -> int:
    return len(_ANSI_RE.sub("", text))


def _terminal_width() -> int:
    """Columns available for a full-width rule. Falls back to 80 when the output
    isn't a terminal (a piped log still reads fine at that width)."""
    try:
        return shutil.get_terminal_size(fallback=(80, 24)).columns
    except Exception:
        return 80


def turn_header(label: str, *styles: str, max_context=None, usage=None) -> str:
    """A turn header: a rule carrying the label and the wall-clock time, run out
    to the terminal width, with the context gauge right-aligned at the end.

        ---- turn 3 | 14:32:07 --------------  [context: 96,000 / 100,000 (96.0%)]

    The rule is drawn with box-drawing characters where the terminal can encode
    them (see _supports_unicode) and dashes where it can't -- the two layouts are
    identical column for column, so nothing shifts between them.

    The timestamp is local wall-clock, deliberately: it is read against the
    user's own sense of how long a session has been running, and against other
    things on their screen. The session log keeps the UTC record."""
    stamp = datetime.now().strftime("%H:%M:%S")
    rule, sep = GLYPHS["rule"], GLYPHS["sep"]

    left = f"{rule * 4} {label} {sep} {stamp} "
    right = context_suffix(max_context, usage)
    width = _terminal_width()
    # The gauge is styled by its own severity color, so the rule is styled apart
    # from it and the two are concatenated already-colored.
    fill = width - _visible_len(left) - _visible_len(right)
    if width < _MIN_RULE_WIDTH or fill < 1:
        return style(left.rstrip(), *styles) + right
    return style(left + rule * fill, *styles) + right

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "dark_red": "\033[38;5;88m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
}


def style(text: str, *names: str) -> str:
    """Wrap text in the given styles (e.g. style("hi", "bold", "cyan"))."""
    if not ENABLED or not names:
        return text
    prefix = "".join(_CODES[n] for n in names)
    return f"{prefix}{text}{_CODES['reset']}"


# Single accent color for every tool invocation.
TOOL_COLOR = "cyan"


# Cosmetic limits for rendering tool calls in the terminal. Display-only: they
# never affect what is written, dispatched, or recorded to the session log.
PREVIEW_HEAD_LINES = 10              # write_file/lean_run_code: lines shown at the top...
PREVIEW_TAIL_LINES = 10             # ...and at the bottom of a truncated preview
# At/below this many lines the whole thing is shown; above it, head+tail with an
# elision marker. Equals head+tail so a truncated preview never overlaps itself.
PREVIEW_FULL_MAX_LINES = PREVIEW_HEAD_LINES + PREVIEW_TAIL_LINES
EDIT_FILE_DIFF_MAX_LINES = 30        # show the diff at/below this; else summarize
SNIPPET_INLINE_MAX_LINES = 12        # lean_multi_attempt: show a snippet inline below this
SNIPPET_INLINE_MAX_CHARS = 800       # ...and only if not too large overall
POSITION_CONTEXT_LINES = 2           # lines of context shown around a queried position
_BODY_INDENT = "     "               # aligns body lines under "  -> "
_LINE_CLIP = 200                     # max width of a shown content/diff line
PROSE_WRAP_WIDTH = 88                # zoom prompt/summary: wrap prose to this width
ZOOM_PROSE_HEAD_LINES = 20           # zoom prompt/summary: generous head+tail bound
ZOOM_PROSE_TAIL_LINES = 10           # (these texts are the point, so show more)

# lean_* tools that query the language server at a (file_path, line, column) and
# read nicely with the source line + a caret at the column.
_POSITION_TOOLS = {"lean_goal", "lean_term_goal", "lean_hover_info"}


def _clip(s: str, width: int = _LINE_CLIP) -> str:
    return s if len(s) <= width else s[: width - 3] + "..."


def _gutter(n: int | None) -> str:
    """A right-aligned, dim line-number gutter (blank when n is None)."""
    return style(f"{'' if n is None else n:>4} ", "dim")


def _render_file_preview(lines: list[str]) -> str:
    """A gutter-numbered preview of file/snippet contents, shown no matter the
    size. At/below PREVIEW_FULL_MAX_LINES the whole thing is shown; above it, the
    first PREVIEW_HEAD_LINES and last PREVIEW_TAIL_LINES with an elision marker in
    between noting how many lines were omitted (the tail keeps real line numbers).
    Each line is clipped to _LINE_CLIP, so the output is bounded regardless of
    input. Display-only."""
    def row(n: int, text: str) -> str:
        return _BODY_INDENT + _gutter(n) + style(_clip(text), "gray")

    total = len(lines)
    if total <= PREVIEW_FULL_MAX_LINES:
        return "\n".join(row(i, ln) for i, ln in enumerate(lines, 1))
    omitted = total - PREVIEW_HEAD_LINES - PREVIEW_TAIL_LINES
    marker = _BODY_INDENT + _gutter(None) + style(
        f"... ({omitted} line{'' if omitted == 1 else 's'} omitted)", "dim"
    )
    head = [row(i, ln) for i, ln in enumerate(lines[:PREVIEW_HEAD_LINES], 1)]
    tail_start = total - PREVIEW_TAIL_LINES + 1
    tail = [row(tail_start + i, ln) for i, ln in enumerate(lines[-PREVIEW_TAIL_LINES:])]
    return "\n".join(head + [marker] + tail)


def _render_read_result(content: str) -> str:
    """A line-numbered preview of a read_file *result* (head+tail when long),
    shown in place of the generic truncated-to-200-chars tool-result preview --
    same head/tail elision as write_file/lean_run_code. `content` is whatever the
    model sees (already run through truncate_tool_output), so the preview reflects
    exactly that. Display-only."""
    lines = content.splitlines()
    if not lines:
        return style("(empty)", "gray")
    n = len(lines)
    header = style(f"({n} line{'' if n == 1 else 's'})", "gray")
    return f"{header}\n{_render_file_preview(lines)}"


def _elide_middle(
    lines: list[str],
    head: int = PREVIEW_HEAD_LINES,
    tail: int = PREVIEW_TAIL_LINES,
) -> list[str]:
    """Apply the head+tail elision policy to a list of already-rendered display
    lines (which carry their own prefix/color -- unlike _render_file_preview,
    which numbers raw text). At/below head+tail lines they are returned
    unchanged; above it, the first `head` and last `tail` with an elision
    marker between."""
    total = len(lines)
    if total <= head + tail:
        return lines
    omitted = total - head - tail
    marker = _BODY_INDENT + style(
        f"... ({omitted} line{'' if omitted == 1 else 's'} omitted)", "dim"
    )
    return lines[:head] + [marker] + lines[-tail:]


# Diagnostics by severity: the leading symbol and color used to render each line.
# The symbol reinforces the color (and survives NO_COLOR).
_SEVERITY_STYLE = {
    "error":   ("✗", "red"),
    "warning": ("⚠", "yellow"),
    "info":    ("ℹ", "cyan"),
    "hint":    ("·", "gray"),
}
_SEVERITY_DEFAULT = ("•", "magenta")   # unknown(N) severities
_SEVERITY_ORDER = ("error", "warning", "info", "hint")   # header summary order


def _diagnostic_lines(diags: list) -> list[str]:
    """Render a list of {severity,message,line,column} diagnostics to display
    lines: one severity symbol + color per line, multi-line messages keeping the
    symbol on every line so a truncated tail stays legible. Shared by the
    diagnostics-result and hover-info previews."""
    out: list[str] = []
    for d in diags:
        if not isinstance(d, dict):
            continue
        symbol, color = _SEVERITY_STYLE.get(d.get("severity", ""), _SEVERITY_DEFAULT)
        loc = f"{d.get('line', '?')}:{d.get('column', '?')}"
        msg_lines = str(d.get("message", "")).splitlines() or [""]
        out.append(_BODY_INDENT + style(_clip(f"{symbol} {loc}: {msg_lines[0]}"), color))
        for ln in msg_lines[1:]:
            out.append(_BODY_INDENT + style(_clip(f"{symbol} {ln}"), color))
    return out


def _render_diagnostics_result(content: str) -> str | None:
    """Human-readable preview of a lean_run_code / lean_diagnostic_messages
    *result*. Both return a JSON object carrying a compile status and a list of
    diagnostics ({severity,message,line,column}) -- under "diagnostics" for
    lean_run_code, "items" for lean_diagnostic_messages, which may also list
    "failed_dependencies". Raw JSON is noisy to read; render a status header plus
    one block per diagnostic, each line carrying a severity symbol and color, with
    the same head+tail elision as the other previews. Returns None when `content`
    isn't the expected JSON shape, so the caller falls back to the generic
    preview. Display-only -- the model still sees the raw JSON."""
    try:
        data = json.loads(content)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    diags = data.get("diagnostics")
    if diags is None:
        diags = data.get("items")
    if not isinstance(diags, list):
        return None
    deps = [d for d in (data.get("failed_dependencies") or []) if isinstance(d, str)]

    header = style("✓ compiled", "green") if data.get("success") else style("✗ failed", "red")
    if data.get("timed_out"):
        header += " " + style("(timed out -- partial)", "yellow")
    counts: dict[str, int] = {}
    for d in diags:
        sev = d.get("severity", "?") if isinstance(d, dict) else "?"
        counts[sev] = counts.get(sev, 0) + 1
    ordered = [s for s in _SEVERITY_ORDER if s in counts] + \
              [s for s in counts if s not in _SEVERITY_ORDER]
    summary = [f"{counts[s]} {s}{'' if counts[s] == 1 else 's'}" for s in ordered]
    if deps:
        summary.append(f"{len(deps)} failed dependenc{'y' if len(deps) == 1 else 'ies'}")
    if summary:
        header += " " + style(f"({', '.join(summary)})", "dim")
    elif data.get("success"):
        header += " " + style("(no diagnostics)", "dim")

    # Failed dependencies first (they are usually the root cause), then each
    # diagnostic. Every line carries its severity symbol so a truncated tail stays
    # legible.
    body: list[str] = [
        _BODY_INDENT + style(_clip(f"✗ failed dependency: {dep}"), "red") for dep in deps
    ]
    body += _diagnostic_lines(diags)
    body = _elide_middle(body)
    return header + ("\n" + "\n".join(body) if body else "")


def _render_build_result(content: str) -> str | None:
    """Human-readable preview of a lean_build *result* -- a JSON BuildResult
    {success, output, errors}. Render a pass/fail header, any error strings (red,
    one symbol per line), then the build log (dimmed), with the same head+tail
    elision as the other previews. Errors come first since they are the point of a
    failed build; the log is trailing context. Returns None when `content` isn't
    the expected shape, so the caller falls back to the generic preview.
    Display-only -- the model still sees the raw JSON."""
    try:
        data = json.loads(content)
    except ValueError:
        return None
    # "output" keys this apart from the diagnostics shape and from arbitrary JSON.
    if not isinstance(data, dict) or "output" not in data:
        return None
    errors = [e for e in (data.get("errors") or []) if isinstance(e, str)]

    header = (
        style("✓ build succeeded", "green") if data.get("success")
        else style("✗ build failed", "red")
    )
    if errors:
        header += " " + style(f"({len(errors)} error{'' if len(errors) == 1 else 's'})", "dim")

    body: list[str] = []
    for err in errors:
        for ln in (err.splitlines() or [""]):
            body.append(_BODY_INDENT + style(_clip(f"✗ {ln}"), "red"))
    for ln in str(data.get("output", "")).splitlines():
        body.append(_BODY_INDENT + style(_clip(ln), "gray"))

    body = _elide_middle(body)
    return header + ("\n" + "\n".join(body) if body else "")


# The three standard Lean 4 kernel axioms -- trusted and expected in most
# mathlib-based proofs. Anything else in a lean_verify report is worth
# flagging: sorryAx means the proof (transitively) contains a sorry, and any
# other name is a custom or TCB-extending axiom (e.g. Lean.ofReduceBool from
# native_decide).
_STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def _render_verify_result(content: str) -> str | None:
    """Human-readable preview of a lean_verify *result* -- a JSON
    {axioms, warnings}. The tool exists to answer one question: does the proof
    rest only on the standard kernel axioms, or does it smuggle in sorryAx (an
    unfinished proof) or a custom axiom? Lead with that verdict as a colored
    header, then list each axiom (standard dimmed, sorryAx red, custom yellow)
    and any source-scan warnings. Returns None when `content` isn't the
    expected shape, so the caller falls back to the generic preview.
    Display-only -- the model still sees the raw JSON."""
    try:
        data = json.loads(content)
    except ValueError:
        return None
    if not isinstance(data, dict) or "axioms" not in data:
        return None
    axioms = [a for a in (data.get("axioms") or []) if isinstance(a, str)]
    warnings = [w for w in (data.get("warnings") or []) if isinstance(w, str)]

    custom = [a for a in axioms if a != "sorryAx" and a not in _STANDARD_AXIOMS]
    if "sorryAx" in axioms:
        header = style("✗ depends on sorryAx -- proof incomplete", "red")
    elif custom:
        header = style(
            f"⚠ nonstandard axiom{'' if len(custom) == 1 else 's'}: "
            f"{', '.join(custom)}", "yellow",
        )
    elif axioms:
        header = style("✓ standard axioms only", "green")
    else:
        header = style("✓ no axioms", "green")
    if warnings:
        header += " " + style(
            f"({len(warnings)} warning{'' if len(warnings) == 1 else 's'})", "dim"
        )

    body: list[str] = []
    for a in axioms:
        if a == "sorryAx":
            body.append(_BODY_INDENT + style(_clip(f"✗ {a}"), "red"))
        elif a in _STANDARD_AXIOMS:
            body.append(_BODY_INDENT + style(_clip(f"· {a}"), "gray"))
        else:
            body.append(_BODY_INDENT + style(_clip(f"⚠ {a}"), "yellow"))
    for w in warnings:
        for ln in (w.splitlines() or [""]):
            body.append(_BODY_INDENT + style(_clip(f"⚠ {ln}"), "yellow"))
    body = _elide_middle(body)
    return header + ("\n" + "\n".join(body) if body else "")


def _goal_blocks(goals: list, pad: str = "") -> list[str]:
    """Render pretty-printed goals to display lines: the ⊢ target line highlighted,
    hypotheses dimmed, with a 'goal i/n' separator when there is more than one.
    `pad` adds indentation (used to nest goals under before/after). A goal is the
    pretty text (default format); a structured goal dict falls back to its
    'pretty'/'goal' field."""
    out: list[str] = []
    for i, g in enumerate(goals, 1):
        text = g if isinstance(g, str) else (g.get("pretty") or g.get("goal") or "")
        if len(goals) > 1:
            out.append(_BODY_INDENT + pad + style(f"goal {i}/{len(goals)}", "dim"))
        for ln in (str(text).splitlines() or [""]):
            styles = ("bold", "cyan") if ln.lstrip().startswith("⊢") else ("gray",)
            out.append(_BODY_INDENT + pad + style(_clip(ln), *styles))
    return out


def _render_goal_result(content: str) -> str | None:
    """Human-readable preview of a lean_goal *result* -- a JSON GoalState. With a
    column it carries `goals`; without, `goals_before`/`goals_after` showing how
    the line's tactic transforms the state. Render each goal's pretty text with the
    ⊢ target highlighted; an empty goal list is "✓ no goals" (proof complete).
    head+tail elision as elsewhere. Returns None on an unexpected shape, so the
    caller falls back to the generic preview. Display-only."""
    try:
        data = json.loads(content)
    except ValueError:
        return None
    if not isinstance(data, dict) or "line_context" not in data:
        return None
    if not any(data.get(k) is not None for k in ("goals", "goals_before", "goals_after")):
        return None  # e.g. a term-goal shape; let the generic preview handle it

    # Column given: a single goal list.
    if data.get("goals") is not None:
        goals = data["goals"]
        if not goals:
            return style("✓ no goals", "green")
        header = style(f"{len(goals)} goal{'' if len(goals) == 1 else 's'}", "cyan")
        body = _elide_middle(_goal_blocks(goals))
        return header + "\n" + "\n".join(body)

    # Column omitted: goals at line start and end (the tactic's effect).
    header = style("goals before → after", "cyan")
    body: list[str] = []
    for label, goals in (("before", data.get("goals_before") or []),
                         ("after", data.get("goals_after") or [])):
        if goals:
            body.append(_BODY_INDENT + style(
                f"{label} ({len(goals)} goal{'' if len(goals) == 1 else 's'})", "dim"))
            body += _goal_blocks(goals, pad="  ")
        else:
            body.append(_BODY_INDENT + style(f"{label}: ", "dim") + style("✓ no goals", "green"))
    body = _elide_middle(body)
    return header + ("\n" + "\n".join(body) if body else "")


def _render_hover_result(content: str) -> str | None:
    """Human-readable preview of a lean_hover_info *result* -- a JSON HoverInfo
    {symbol, info, diagnostics}. Show the hovered symbol as a header, its type/doc
    text (the signature line highlighted, docs dimmed), and any diagnostics at the
    position with their severity symbols. head+tail elision as elsewhere. Returns
    None on an unexpected shape, so the caller falls back to the generic preview.
    Display-only."""
    try:
        data = json.loads(content)
    except ValueError:
        return None
    if not isinstance(data, dict) or "info" not in data:
        return None

    symbol = str(data.get("symbol", "")).strip()
    header = style(symbol, "bold", "cyan") if symbol else style("hover", "cyan")
    diags = data.get("diagnostics") or []
    if diags:
        header += " " + style(f"({len(diags)} diagnostic{'' if len(diags) == 1 else 's'})", "dim")

    info_lines = str(data.get("info", "")).splitlines()
    while info_lines and not info_lines[0].strip():   # trim leading/trailing blanks
        info_lines.pop(0)
    while info_lines and not info_lines[-1].strip():
        info_lines.pop()
    # First line is the type signature (highlighted); the rest is documentation.
    body = [
        _BODY_INDENT + style(_clip(ln), *(("cyan",) if i == 0 else ("gray",)))
        for i, ln in enumerate(info_lines)
    ]
    body += _diagnostic_lines(diags)

    body = _elide_middle(body)
    return header + ("\n" + "\n".join(body) if body else "")


def _prose_block(text: str) -> list[str]:
    """Free prose (a zoom_in prompt / zoom_out summary) as indented display
    lines: each paragraph wrapped to PROSE_WRAP_WIDTH -- these texts are written
    for a human, so wrap them readably instead of clipping at _LINE_CLIP --
    blank lines kept as paragraph breaks, and a generous head+tail elision
    bounding a very long text. Display-only."""
    out: list[str] = []
    for para in text.splitlines():
        if not para.strip():
            out.append(_BODY_INDENT.rstrip())
            continue
        for ln in textwrap.wrap(
            para, PROSE_WRAP_WIDTH, drop_whitespace=True
        ) or [""]:
            out.append(_BODY_INDENT + ln)
    return _elide_middle(out, ZOOM_PROSE_HEAD_LINES, ZOOM_PROSE_TAIL_LINES)


def _render_zoom_in(args: dict, read_file=None) -> str:
    """zoom_in hands one sorry to the smaller model. Show the sorry's position
    with the source line + caret (like the position-query lean_* tools), then
    the full task prompt as wrapped prose under a 'prompt:' label -- the prompt
    is the interesting part for a human following along."""
    loc = _render_position(
        {
            "file_path": args.get("file", "?"),
            "line": args.get("line"),
            "column": args.get("column"),
        },
        read_file,
    )
    prompt = str(args.get("prompt", "")).strip()
    if not prompt:
        return f"{loc}\n{_BODY_INDENT}{style('(no prompt)', 'gray')}"
    label = _BODY_INDENT + style("prompt:", "bold", "magenta")
    return "\n".join([loc, label] + _prose_block(prompt))


def _render_zoom_out(args: dict) -> str:
    """zoom_out ends a sub-session; its summary is the whole report the outer
    model receives, so show it in full as wrapped prose under a 'summary:'
    label."""
    summary = str(args.get("summary", "")).strip()
    if not summary:
        return " " + style("(empty summary)", "gray")
    label = _BODY_INDENT + style("summary:", "bold", "magenta")
    return "\n".join(["", label] + _prose_block(summary))


def format_tool_call(name: str, args: dict, read_file=None) -> str:
    """A pretty, single- or multi-line rendering of a tool call for the terminal.

    Purely cosmetic: write_file shows its contents (when small) or a summary, and
    edit_file shows a diff (or a summary). Every other tool keeps the plain
    `name(args)` form. The dict passed to the tool and recorded to the session is
    unaffected -- this only changes what is printed."""
    arrow = style("->", TOOL_COLOR)
    label = style(name, "bold", TOOL_COLOR)
    head = f"  {arrow} {label}"
    if name == "write_file" and isinstance(args.get("content"), str):
        return f"{head} {_render_write_file(args)}"
    if (
        name == "edit_file"
        and isinstance(args.get("old_string"), str)
        and isinstance(args.get("new_string"), str)
    ):
        return f"{head} {_render_edit_file(args, read_file)}"
    if name == "lean_multi_attempt" and isinstance(args.get("snippets"), list):
        return f"{head} {_render_lean_multi_attempt(args)}"
    if name == "lean_run_code" and isinstance(args.get("code"), str):
        return f"{head} {_render_lean_run_code(args)}"
    if name == "zoom_in":
        return f"{head} {_render_zoom_in(args, read_file)}"
    if name == "zoom_out":
        return f"{head}{_render_zoom_out(args)}"
    if name in _POSITION_TOOLS:
        return f"{head} {_render_position(args, read_file)}"
    if name == "lean_build":
        extra = _render_lean_build(args)
        return f"{head} {extra}" if extra else head
    if name == "lean_diagnostic_messages" and isinstance(args.get("file_path"), str):
        return f"{head} {_render_path_with_extras(args)}"
    if name == "lean_verify" and isinstance(args.get("theorem_name"), str):
        return f"{head} {_render_lean_verify(args)}"
    return f"{head}({args})"


def _render_write_file(args: dict) -> str:
    path = style(str(args.get("path", "?")), TOOL_COLOR)
    content = args["content"]
    lines = content.splitlines()
    if not lines:
        return f"{path} {style('(empty)', 'gray')}"
    # Always show a preview (head+tail when long); for a long file also keep the
    # total size on the path line, since the elision marker only counts lines.
    head = path
    if len(lines) > PREVIEW_FULL_MAX_LINES:
        n = len(lines)
        summary = f"({n} lines, {len(content.encode('utf-8', 'replace'))} bytes)"
        head = f"{path} {style(summary, 'gray')}"
    return f"{head}\n{_render_file_preview(lines)}"


def _edit_line_offset(read_file, path: str, old_string: str) -> int:
    """How many lines precede old_string in the file -- the amount to add to the
    fragment-relative diff numbers to get real file line numbers. 0 if the file
    can't be read or old_string isn't found (best-effort, display-only)."""
    if read_file is None:
        return 0
    try:
        content = read_file(path)
    except Exception:
        return 0
    idx = content.find(old_string)
    return content.count("\n", 0, idx) if idx >= 0 else 0


def _render_edit_file(args: dict, read_file=None) -> str:
    path_str = str(args.get("path", "?"))
    path = style(path_str, TOOL_COLOR)
    old_string = args["old_string"]
    diff = [
        d for d in difflib.unified_diff(
            old_string.splitlines(), args["new_string"].splitlines(),
            lineterm="", n=2,
        )
        if not d.startswith(("---", "+++"))
    ]
    if not diff:
        return f"{path} {style('(no textual change)', 'gray')}"
    if len(diff) > EDIT_FILE_DIFF_MAX_LINES:
        adds = sum(1 for d in diff if d.startswith("+"))
        dels = sum(1 for d in diff if d.startswith("-"))
        return f"{path} {style(f'(+{adds} -{dels} lines)', 'gray')}"
    # The diff is relative to old_string (starting at line 1). Locate old_string
    # in the actual file to offset the gutter to real file line numbers. The edit
    # has not run yet, so the file still contains old_string. Falls back to
    # fragment-relative numbers if the file can't be read or old_string isn't found.
    offset = _edit_line_offset(read_file, path_str, old_string)
    # Removed lines show their position in the original, added lines in the new.
    rendered = []
    old_ln = new_ln = 0
    for d in diff:
        if d.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", d)
            if m:
                old_ln, new_ln = int(m.group(1)) + offset, int(m.group(3)) + offset
                # Rewrite the hunk header to the same real line numbers as the
                # gutter (the raw header is relative to old_string).
                ocnt = f",{m.group(2)}" if m.group(2) else ""
                ncnt = f",{m.group(4)}" if m.group(4) else ""
                d = f"@@ -{old_ln}{ocnt} +{new_ln}{ncnt} @@"
            rendered.append(_BODY_INDENT + _gutter(None) + style(d, "dim"))
        elif d.startswith("-"):
            rendered.append(_BODY_INDENT + _gutter(old_ln) + style(_clip(d), "red"))
            old_ln += 1
        elif d.startswith("+"):
            rendered.append(_BODY_INDENT + _gutter(new_ln) + style(_clip(d), "green"))
            new_ln += 1
        else:  # context line (leading space)
            rendered.append(_BODY_INDENT + _gutter(new_ln) + style(_clip(d), "gray"))
            old_ln += 1
            new_ln += 1
    return f"{path}\n" + "\n".join(rendered)


def _render_lean_multi_attempt(args: dict) -> str:
    """lean_multi_attempt tries candidate snippets (often many lines of Lean) at a
    position. Show file:line:col and each snippet -- inline when short, summarized
    when long -- instead of dumping the whole snippets list."""
    loc = str(args.get("file_path", "?"))
    line, col = args.get("line"), args.get("column")
    if line is not None:
        loc += f":{line}" + (f":{col}" if col is not None else "")
    head = style(loc, TOOL_COLOR)
    snippets = args.get("snippets") or []
    if len(snippets) != 1:
        head += " " + style(f"({len(snippets)} snippets)", "gray")
    return "\n".join([head] + [_render_snippet(i, str(s)) for i, s in enumerate(snippets, 1)])


def _render_path_with_extras(args: dict) -> str:
    """Show a tool's file_path prominently, with any other args appended compactly
    (e.g. a line range) -- instead of echoing a full args dict."""
    out = style(str(args["file_path"]), TOOL_COLOR)
    extras = [f"{k}={v}" for k, v in args.items() if k != "file_path"]
    if extras:
        out += " " + style(f"({', '.join(extras)})", "gray")
    return out


def _render_lean_build(args: dict) -> str:
    """lean_build just builds the project; its args are mostly default booleans
    (output_lines is only a display cap). Surface the flags that are actually on,
    or nothing for a plain build -- no need to echo a dict of defaults."""
    flags = []
    if args.get("clean"):
        flags.append("clean")
    if args.get("fetch_cache"):
        flags.append("fetch cache")
    return style(f"({', '.join(flags)})", "gray") if flags else ""


def _render_position(args: dict, read_file=None) -> str:
    """For a position-query lean_* tool: show file:line:col, then the source line
    at that position (with a few lines of context) and a caret under the column.
    Falls back to just the location when the file can't be read."""
    path = str(args.get("file_path", "?"))
    line, col = args.get("line"), args.get("column")
    loc = path
    if isinstance(line, int):
        loc += f":{line}" + (f":{col}" if isinstance(col, int) else "")
    loc = style(loc, TOOL_COLOR)
    if read_file is None or not isinstance(line, int):
        return loc
    try:
        lines = read_file(path).splitlines()
    except Exception:
        return loc
    if not 1 <= line <= len(lines):
        return loc
    out = [loc]
    lo = max(1, line - POSITION_CONTEXT_LINES)
    hi = min(len(lines), line + POSITION_CONTEXT_LINES)
    for n in range(lo, hi + 1):
        text = _clip(lines[n - 1])
        out.append(_BODY_INDENT + _gutter(n) + style(text, "bold" if n == line else "gray"))
        if n == line and isinstance(col, int) and col >= 1:
            pad = " " * (min(col, len(text) + 1) - 1)
            out.append(_BODY_INDENT + _gutter(None) + pad + style("^", TOOL_COLOR))
    return "\n".join(out)


def _render_lean_verify(args: dict) -> str:
    """lean_verify checks a theorem's axiom dependencies. Show the (fully
    qualified) theorem name prominently with the file appended dimly, instead
    of echoing the args dict."""
    out = style(str(args["theorem_name"]), TOOL_COLOR)
    if isinstance(args.get("file_path"), str):
        out += " " + style(f"({args['file_path']})", "gray")
    return out


def _render_lean_run_code(args: dict) -> str:
    """lean_run_code runs a standalone Lean snippet. Always show a line-numbered
    preview of the code (head+tail when long), under a line-count header, instead
    of dumping the whole `code` string."""
    code = args["code"]
    lines = code.splitlines()
    if not lines:
        return style("(empty)", "gray")
    n = len(lines)
    header = style(f"({n} line{'' if n == 1 else 's'})", "dim")
    return f"{header}\n{_render_file_preview(lines)}"


def _render_snippet(i: int, snip: str) -> str:
    label = style(f"[{i}]", "dim")
    lines = snip.splitlines()
    if not lines:
        return f"{_BODY_INDENT}{label} {style('(empty)', 'gray')}"
    if len(lines) == 1:
        return f"{_BODY_INDENT}{label} {style(_clip(lines[0]), 'gray')}"
    if len(lines) <= SNIPPET_INLINE_MAX_LINES and len(snip) <= SNIPPET_INLINE_MAX_CHARS:
        inner = "\n".join(_BODY_INDENT + "    " + style(_clip(ln), "gray") for ln in lines)
        return f"{_BODY_INDENT}{label} {style(f'({len(lines)} lines)', 'dim')}\n{inner}"
    first = next((ln for ln in lines if ln.strip()), lines[0])
    summary = f"({len(lines)} lines, {len(snip)} chars)"
    return f"{_BODY_INDENT}{label} {style(summary, 'dim')} {style(_clip(first), 'gray')}"

def format_tool_result(name: str, content: str, raw_content: str, is_error: bool) -> str:
    """The '  <- ...' line(s) shown for a tool result. Tool-specific previews
    (display-only; the model still sees `content`) render `raw_content` -- the
    untruncated result -- so a long output still previews fully; their own
    head+tail elision bounds the size. Anything else falls back to a generic
    200-char preview of `content` (exactly what the model sees)."""
    color = "red" if is_error else "gray"
    rendered = None
    if not is_error:
        if name == "read_file":
            rendered = _render_read_result(content)
        elif name in ("lean_run_code", "lean_diagnostic_messages"):
            rendered = _render_diagnostics_result(raw_content)
        elif name == "lean_build":
            rendered = _render_build_result(raw_content)
        elif name == "lean_goal":
            rendered = _render_goal_result(raw_content)
        elif name == "lean_hover_info":
            rendered = _render_hover_result(raw_content)
        elif name == "lean_verify":
            rendered = _render_verify_result(raw_content)
    if rendered is None:
        preview = content[:200] + "..." if len(content) > 200 else content
        # Align continuation lines under the content (after "  <- ").
        rendered = style(
            preview.rstrip("\n").replace("\n", "\n     "), color
        )
    return f"  {style('<-', color)} {rendered}"


def context_suffix(max_context: int | None, usage) -> str:
    """A ' [context: ...]' fragment appended to a turn header, showing how full
    the window is entering the turn. `usage` is the previous turn's usage (a
    providers.Usage, duck-typed to keep this module free of gerbil imports): its
    `input_tokens` is the whole conversation fed to the model and `output_tokens`
    what it generated -- together, the tokens that had to fit in the window at
    once. Empty before the first turn lands (no measurement yet). When the window
    is known, show the percentage (color escalating toward the limit); when it
    isn't (provider doesn't report it), show the raw total."""
    if usage is None:
        return ""
    used = usage.context_tokens
    if not max_context:
        return style(f"  [context: {used:,} tokens]", "gray")
    pct = used / max_context * 100
    color = "red" if pct >= 80 else "yellow" if pct >= 50 else "gray"
    return style(f"  [context: {used:,} / {max_context:,} ({pct:.1f}%)]", color)


def usage_line(turns: int, usage, cost: float | None) -> str:
    """The end-of-session summary line with token counts and estimated cost.
    `usage` is a providers.Usage (duck-typed to keep this module free of gerbil
    imports); `cost` comes from pricing.estimate_cost, None = unknown."""
    cost_str = "cost: N/A" if cost is None else f"~${cost:.4f}"
    total = (
        usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
        + usage.output_tokens
    )
    # Cache reads/writes are extra prompt tokens on top of input_tokens (the
    # uncached remainder), each billed at its own rate -- break them out so the
    # discount is visible.
    inp = f"in: {usage.input_tokens:,}"
    if usage.cache_read_tokens or usage.cache_write_tokens:
        inp += (
            f" + {usage.cache_read_tokens:,} cache-read"
            f" + {usage.cache_write_tokens:,} cache-write"
        )
    # thinking_tokens is a subset of output_tokens, so show it as a breakdown.
    out = f"out: {usage.output_tokens:,}"
    if usage.thinking_tokens:
        out += f" incl. {usage.thinking_tokens:,} thinking"
    line = (
        f"--- {turns} turns, {total:,} tokens "
        f"({inp}, {out}), "
        f"{cost_str} ---"
    )
    return style(line, "bold")


def print_usage(turns: int, usage, cost: float | None) -> None:
    """Print the end-of-session summary line (see usage_line)."""
    print("\n" + usage_line(turns, usage, cost), flush=True)
