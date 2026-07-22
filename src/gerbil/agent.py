"""Agent loop — drives turns between the LLM and the sandbox until the model
stops requesting tools.

run_session() is called by the gerbil CLI once the sandbox is ready and the
mathlib cache is warm. It uses the unified provider stream (providers.stream)
and the sandbox tools (tools.dispatch), recording every turn, tool call, and
tool result to the Session as it goes.
"""

import difflib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .providers import (
    Done,
    TextDelta,
    ToolCall,
    TransientProviderError,
    Usage,
    _ToolMeta,
    get_context_window,
    stream,
)
from .sandbox import LeanSandbox
from .session import Session
from .term import style
from .tools import Toolset, truncate_tool_output

# Single accent color for every tool invocation.
TOOL_COLOR = "cyan"

# Known models and per-million-token pricing (input, output). Best-effort
# estimates, used only for the cost summary. Prices as of July 2026, per
# https://benchlm.ai/llm-pricing. Keys are API-ID-shaped so that gateway model
# strings (e.g. a Portkey `@provider/anthropic.claude-opus-4-8`) can be priced
# by substring -- see pricing_match. Mini/pro variants of a family must appear
# alongside the base key, or the base key would silently (mis)price them.
MODEL_PRICING = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3-pro-preview": (2.0, 12.0),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-3-flash": (0.50, 3.0),
    "gemini-3.5-flash": (1.50, 9.0),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-pro": (30.0, 180.0),
    "gpt-5.4-pro-2026-03-05": (30.0, 180.0),
    "gpt-5.5": (5.0, 30.0),
    "gpt-5.5-pro": (30.0, 180.0),
    "o3": (2.0, 8.0),
    "o3-pro": (20.0, 80.0),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
}

def pricing_match(model: str) -> str | None:
    """The MODEL_PRICING key that prices `model`, or None when we don't know.

    An exact key wins. Otherwise fall back to substring matching, which is what
    prices a gateway model: a Portkey catalog name like
    `@vertexai-foo/anthropic.claude-opus-4-7` embeds the real model name, so a
    MODEL_PRICING key found inside the string identifies the pricing.

    The table nests keys within a model family (`o3` inside `o3-mini`,
    `gpt-5.4` inside `gpt-5.4-pro`), so a string like `@x/o3-mini` matches
    several keys. That isn't real ambiguity: when the longest matching key
    itself contains every other match, the shorter ones are just its prefixes
    riding along, and the longest -- most specific -- key is the answer.
    Anything else (several matches, none subsuming the rest) means we'd be
    guessing, and a guessed price is worse than an honest N/A -> None."""
    if model in MODEL_PRICING:
        return model
    matches = [key for key in MODEL_PRICING if key in model]
    if not matches:
        return None
    longest = max(matches, key=len)
    return longest if all(key in longest for key in matches) else None


# Models already warned about by model_pricing, so a summary spanning many
# sessions of the same unknown model warns once, not once per session.
_pricing_warned: set[str] = set()


def model_pricing(model: str) -> tuple[float, float] | None:
    """Per-million-token (input, output) pricing for a model, or None when the
    pricing is unknown -- callers must then report cost as N/A rather than
    invent a number. ollama models run locally and cost nothing, so they price
    at (0, 0). An unknown model warns once (per process) on stderr."""
    if model.startswith("ollama:"):
        return (0.0, 0.0)
    key = pricing_match(model)
    if key is not None:
        return MODEL_PRICING[key]
    if model not in _pricing_warned:
        _pricing_warned.add(model)
        matches = [k for k in MODEL_PRICING if k in model]
        reason = (
            f"ambiguous match: {', '.join(matches)}"
            if len(matches) > 1
            else "no known model matches"
        )
        print(
            style(f"warning: unknown pricing for '{model}' ({reason}); "
                  "cost will be reported as N/A", "yellow"),
            file=sys.stderr,
        )
    return None

SYSTEM_PROMPT = """\
You are gerbil, an autonomous agent working inside a sandboxed Lean 4 / Lake \
project. Your job is to carry out the user's task by editing files and running \
commands in the project.

You have these tools:
  - bash: run shell commands
  - read_file: read a file's contents
  - write_file: create or overwrite a file
  - edit_file: replace an exact string in a file

If you can, prefer using read_file, write_file, and edit_file for all
file manipulations (instead of bash commands).

Guidelines:
  - Explore the project before editing: read the relevant files first.
  - After making changes, fix any new Lean errors.
  - Do not leave `sorry` in proofs unless the task explicitly allows it.
  - When the task is complete and the project builds, stop and give a short \
summary of what you did. Do not call any more tools once you are done.
  - NEVER DO `import Mathlib`. This is extremely expensive and causes the \
whole system to hang. If you need to import something, only import exactly \
what you need, and no more.
"""

# Appended to the system prompt when lean-lsp (MCP) tools are available.
LSP_TOOLS_NOTE = """\

You also have lean_* tools backed by the Lean language server. Prefer them for \
understanding the proof state instead of guessing:
  - lean_build: full build of the project, refreshes .olean files
  - lean_goal / lean_term_goal: the proof state at a position (line/col are 1-indexed)
  - lean_diagnostic_messages: compiler errors/warnings for a file
  - lean_hover_info: type signature and docs for an identifier
  - lean_multi_attempt: try candidate tactics at a position WITHOUT editing the file
  - lean_run_code: run a code snippet without needing to write it to a file
  - lean_local_search: search the LOCAL Lean/mathlib source for declarations and \
lemmas (ripgrep-backed) -- use it before guessing a lemma name
  - reset_lean_server: restart the language server if the lean_* tools start \
timing out or acting stuck/hung; the next lean_* call re-initializes it (and may \
be slow). It does not touch your files.
The lean_* tools never modify files; keep using edit_file / write_file for changes. \
After editing a file, re-run lean_build (or a diagnostics call) so the language \
server sees your changes.

If you can, prefer using `lean_build` instead of running the bash command \
`lake build`.
"""


# Appended to the system prompt in --ralph mode.
RALPH_NOTE = """\

You are running in a repeating loop: after this session ends, the same task \
prompt runs again in a fresh session that builds on the changes you commit now. \
Focus on solid, incremental progress that the next session can build on; do not \
try to do everything at once.
"""


# Appended to the system prompt to pin down how the final state must be left.
# gerbil reads the result purely as `git format-patch <base>..HEAD`, so the
# agent's work must end up reachable from that range. {base} is the commit the
# session starts on.
GIT_STATE_NOTE = """\

You are working inside of a git repository, starting from commit {base}. Before \
you finish, ensure that all of your changes are visible via the command \
`git format-patch {base}..HEAD`. This is the only way we will be able to see your \
changes; anything not reachable from that range is lost. You do not need to \
commit -- any uncommitted changes you leave in the working tree are committed for \
you -- but do not hide or discard your work: do not run `git reset`, `git \
checkout`/`git restore`, `git stash`, or `git init`, and do not create another \
git repository inside this one.
"""


def build_system_prompt(
    has_lsp_tools: bool, ralph: bool = False, base_commit: str = ""
) -> str:
    """The system prompt, with notes appended for active features."""
    prompt = SYSTEM_PROMPT
    if has_lsp_tools:
        prompt += LSP_TOOLS_NOTE
    if ralph:
        prompt += RALPH_NOTE
    if base_commit:
        prompt += GIT_STATE_NOTE.format(base=base_commit)
    return prompt


@dataclass
class SessionResult:
    final_text: str       # the model's last task-phase message
    diff: str             # git patch of all changes made during the session
    commit_message: str   # generated commit title + body ("" if no changes)


def _commit_request(diff: str) -> str:
    """The user message that asks the model to write the commit message."""
    return (
        "The task is complete. Here is the final git diff of all your changes:\n\n"
        f"{diff}\n\n"
        "Write a git commit message for these changes. Output ONLY the commit "
        "message, with no code fences or preamble:\n"
        "  - First line: a concise imperative title, at most ~72 characters.\n"
        "  - Then one blank line.\n"
        "  - Then a short body explaining what changed and why, wrapped at ~72 columns."
    )


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


def _elide_middle(lines: list[str]) -> list[str]:
    """Apply the head+tail elision policy to a list of already-rendered display
    lines (which carry their own prefix/color -- unlike _render_file_preview,
    which numbers raw text). At/below PREVIEW_FULL_MAX_LINES the lines are returned
    unchanged; above it, the first PREVIEW_HEAD_LINES and last PREVIEW_TAIL_LINES
    with an elision marker between."""
    total = len(lines)
    if total <= PREVIEW_FULL_MAX_LINES:
        return lines
    omitted = total - PREVIEW_HEAD_LINES - PREVIEW_TAIL_LINES
    marker = _BODY_INDENT + style(
        f"... ({omitted} line{'' if omitted == 1 else 's'} omitted)", "dim"
    )
    return lines[:PREVIEW_HEAD_LINES] + [marker] + lines[-PREVIEW_TAIL_LINES:]


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


def _format_tool_call(name: str, args: dict, read_file=None) -> str:
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
    if name in _POSITION_TOOLS:
        return f"{head} {_render_position(args, read_file)}"
    if name == "lean_build":
        extra = _render_lean_build(args)
        return f"{head} {extra}" if extra else head
    if name == "lean_diagnostic_messages" and isinstance(args.get("file_path"), str):
        return f"{head} {_render_path_with_extras(args)}"
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


def _run_turn(model, system, messages, tools, provider, read_file=None):
    """Stream a single turn, printing text and tool calls live.

    Returns (assistant_parts, tool_calls, text, usage).
    """
    assistant_parts = []
    current_text = ""
    tool_calls = []  # {name, args, id, raw_part}
    usage = Usage()

    for event in stream(model, system, messages, tools, provider):
        if isinstance(event, TextDelta):
            sys.stdout.write(event.text)
            sys.stdout.flush()
            current_text += event.text
        elif isinstance(event, ToolCall):
            if current_text:
                assistant_parts.append({"type": "text", "text": current_text})
                current_text = ""
            print(
                "\n" + _format_tool_call(event.name, event.args, read_file),
                flush=True,
            )
            tool_calls.append({
                "name": event.name,
                "args": event.args,
                "id": None,
                "raw_part": event.raw_part,
            })
        elif isinstance(event, _ToolMeta):
            if tool_calls:
                tool_calls[-1]["id"] = event.tool_use_id
        elif isinstance(event, Done):
            usage = event.usage

    if current_text:
        assistant_parts.append({"type": "text", "text": current_text})
    for tc in tool_calls:
        assistant_parts.append({
            "type": "tool_call",
            "name": tc["name"],
            "args": tc["args"],
            "id": tc["id"],
            "raw_part": tc["raw_part"],
        })

    text = "".join(p["text"] for p in assistant_parts if p["type"] == "text")
    return assistant_parts, tool_calls, text, usage


# How long to wait between retries of a transient provider failure.
RETRY_DELAY_SECONDS = 5

# HTTP status codes (and Google RPC status names) that mark a transient, retryable
# provider failure -- the service is up but momentarily can't serve us.
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_RETRYABLE_STATUS_NAMES = {
    "unavailable", "resource_exhausted", "internal", "aborted",
    "deadline_exceeded", "cancelled",
}
# Phrases (lowercased) that mark a transient failure when no status is exposed.
# Kept specific so genuinely permanent errors (bad request, auth, context length)
# are NOT matched and still abort the run.
_TRANSIENT_MARKERS = (
    "unavailable", "overloaded", "rate limit", "ratelimit",
    "resource exhausted", "too many requests", "try again",
    "temporarily", "internal error", "internal server error",
    "bad gateway", "gateway timeout", "service is currently",
    "connection reset", "connection aborted", "connection error",
    "server disconnected", "remote end closed", "timed out", "timeout",
)


def _is_transient_error(exc: BaseException) -> bool:
    """Whether `exc` is a transient provider failure worth retrying (503/
    UNAVAILABLE, rate limits, 5xx, dropped connections). Conservative: anything
    not recognized returns False so permanent errors still abort the run."""
    # A malformed/empty Gemini turn surfaces as no SDK exception, only a finish
    # reason; providers.py converts it to this, which is always retryable.
    if isinstance(exc, TransientProviderError):
        return True
    for attr in ("status_code", "code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, bool):  # bool is an int subclass; never a status code
            continue
        if isinstance(val, int) and val in _RETRYABLE_STATUS_CODES:
            return True
        if isinstance(val, str) and val.strip().isdigit() and int(val) in _RETRYABLE_STATUS_CODES:
            return True
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS_CODES:
        return True
    status = getattr(exc, "status", None)
    if isinstance(status, str) and status.strip().lower() in _RETRYABLE_STATUS_NAMES:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


def _run_turn_with_retry(
    model, system, messages, tools, provider, read_file=None, session=None
):
    """Run one streamed turn, retrying transient provider failures every
    RETRY_DELAY_SECONDS until the call succeeds. Everything stays warm across the
    wait -- the session, the sandbox, and the conversation are untouched; only this
    one model call is repeated (from the same `messages`, so no state is lost).
    Non-transient errors propagate immediately, so the run still aborts (and stays
    resumable) on a real failure. A Ctrl-C during the wait aborts cleanly (it is a
    BaseException, not caught here)."""
    attempt = 0
    while True:
        try:
            return _run_turn(model, system, messages, tools, provider, read_file)
        except Exception as exc:
            if not _is_transient_error(exc):
                raise
            attempt += 1
            detail = _clip(f"{type(exc).__name__}: {exc}", 200)
            msg = (
                f"[provider unavailable: {detail}; retrying in "
                f"{RETRY_DELAY_SECONDS}s (attempt {attempt})]"
            )
            print("\n" + style(msg, "bold", "yellow"), flush=True)
            if session is not None:
                session.record_warning(msg)
            time.sleep(RETRY_DELAY_SECONDS)


def _thought_sig(raw_part) -> str | None:
    """Base64 of a Gemini tool-call's thought_signature, for the session log.

    raw_part is the provider's original tool-call part: a google-genai Part for
    Gemini (which may carry a thought_signature), and None for every other
    provider. Returns None when there is no signature to record."""
    sig = getattr(raw_part, "thought_signature", None)
    if not sig:
        return None
    import base64

    return base64.b64encode(sig).decode()


def _update_wip_patch(sandbox: LeanSandbox, path: Path | None, base: str) -> None:
    """Refresh the live resume snapshot -- a format-patch from `base` to the
    current state (committed + uncommitted), so a crash can be resumed or the
    patch applied directly. Best-effort: a checkpoint must never be the thing that
    crashes the session."""
    if path is None:
        return
    try:
        path.write_text(sandbox.wip_patch(base))
    except Exception:
        pass


def run_session(
    sandbox: LeanSandbox,
    session: Session,
    prompt: str,
    model: str,
    toolset: Toolset,
    provider: str | None = None,
    max_turns: int | None = None,
    messages: list | None = None,
    wip_patch_path: Path | None = None,
) -> SessionResult:
    """Run the agent loop until the model stops calling tools (or max_turns).

    When the model finishes the task naturally, one more turn is appended to the
    same conversation asking it to write a commit message for its diff; that turn
    is recorded and counted like any other.

    `messages`, when given, is a pre-built conversation to continue from (used by
    --resume to pick up a crashed session); the initial prompt is then assumed to
    already live in that history and is not re-recorded. `wip_patch_path`, when
    given, receives the working-tree diff after every turn, so a crash leaves a
    patch that reconstructs the tree exactly.
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
        session.record_turn("user", prompt)

    system = build_system_prompt(
        bool(toolset.mcp_tool_names()), ralph=toolset.ralph,
        base_commit=session.base_commit,
    )
    tools = toolset.schemas()

    # Query the model's maximum context window once at session start; every turn
    # then reports how close the running conversation is to filling it. None means
    # the provider doesn't report it (OpenAI), so we show raw token totals instead.
    max_context = get_context_window(model, provider)
    if max_context:
        banner = f"[context window: {max_context:,} tokens ({model})]"
    else:
        banner = f"[context window: unknown for {model}; reporting token totals only]"
    print(style(banner, "gray"), flush=True)

    total = Usage()
    turn = 0
    # Continue the turn counter across a resume: the seeded history already holds
    # this many assistant turns, so the displayed count picks up where it left off
    # instead of restarting at 1. It's display-only -- the max_turns budget below
    # still counts just this run's new turns. Zero for a fresh (non-resumed) run.
    turn_offset = sum(1 for m in messages if m.get("role") == "assistant")
    final_text = ""
    stopped_at_max = False
    # The most recent turn's usage, shown in the next turn's header so it reflects
    # how full the context is *entering* the turn. None until the first turn lands.
    last_usage: Usage | None = None

    # In a ralph loop, tag every turn header with the session counter so it stays
    # visible while scrolling, not just in the once-per-session banner.
    ralph_tag = (
        f"[ralph {session.ralph['iteration']}/{session.ralph['total']}] "
        if session.ralph else ""
    )

    # A resumed conversation that already ends on an assistant message had no
    # pending tool calls -- the model was done. Skip the loop (calling the model
    # again would append a second assistant turn, which the APIs reject) and go
    # straight to the commit-message phase.
    done_already = messages[-1]["role"] == "assistant" if messages else False
    if done_already:
        final_text = "".join(
            p.get("text", "")
            for p in messages[-1]["content"]
            if isinstance(p, dict) and p.get("type") == "text"
        )

    while not done_already:
        if max_turns and turn >= max_turns:
            stopped_at_max = True
            break
        turn += 1

        header = style(
            f"--- {ralph_tag}turn {turn + turn_offset} ---", "bold", "dark_red"
        )
        print("\n" + header + _context_suffix(max_context, last_usage), flush=True)

        assistant_parts, tool_calls, final_text, usage = _run_turn_with_retry(
            model, system, messages, tools, provider, sandbox.read_file, session
        )
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens
        total.thinking_tokens += usage.thinking_tokens
        last_usage = usage

        messages.append({"role": "assistant", "content": assistant_parts})
        session.record_turn(
            "assistant", final_text, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens,
        )

        # No tool calls => the model is done with the task.
        if not tool_calls:
            print(flush=True)
            break

        # Execute tool calls and feed results back.
        tool_results = []
        for tc in tool_calls:
            session.record_tool_call(
                tc["name"], tc["args"], thought_signature=_thought_sig(tc["raw_part"])
            )
            result = toolset.dispatch(tc["name"], tc["args"])
            # Truncate once and use the same text everywhere: the session log
            # records exactly what the model sees, no more, no less.
            content = truncate_tool_output(result.content)
            session.record_tool_result(tc["name"], content)

            result_color = "red" if result.is_error else "gray"
            # Tool-specific human-readable previews (display-only; the model still
            # sees `content`). They render the untruncated result so a long output
            # still previews fully -- their own head+tail elision bounds the size.
            rendered = None
            if not result.is_error:
                if tc["name"] == "read_file":
                    rendered = _render_read_result(content)
                elif tc["name"] in ("lean_run_code", "lean_diagnostic_messages"):
                    rendered = _render_diagnostics_result(result.content)
                elif tc["name"] == "lean_build":
                    rendered = _render_build_result(result.content)
                elif tc["name"] == "lean_goal":
                    rendered = _render_goal_result(result.content)
                elif tc["name"] == "lean_hover_info":
                    rendered = _render_hover_result(result.content)
            if rendered is None:
                preview = content[:200] + "..." if len(content) > 200 else content
                # Align continuation lines under the content (after "  <- ").
                rendered = style(
                    preview.rstrip("\n").replace("\n", "\n     "), result_color
                )
            print(f"  {style('<-', result_color)} {rendered}", flush=True)

            tr = {
                "type": "tool_result",
                "tool_name": tc["name"],
                "content": content,
            }
            # Anthropic keys on tool_use_id, OpenAI on tool_call_id; set both.
            if tc["id"]:
                tr["tool_use_id"] = tc["id"]
                tr["tool_call_id"] = tc["id"]
            tool_results.append(tr)

        messages.append({"role": "user", "content": tool_results})

        # Snapshot the working tree after every turn that ran tools, so an
        # interruption before the next turn leaves a patch that recreates it.
        _update_wip_patch(sandbox, wip_patch_path, session.base_commit)

    if stopped_at_max:
        print(
            "\n" + style(f"[stopped: reached max_turns={max_turns}]", "bold", "yellow"),
            flush=True,
        )

    # Final turn: ask for a commit message as a true continuation of the
    # conversation. Skipped if nothing changed, or if we bailed on max_turns
    # (the work is incomplete and the history ends on a tool result). The diff is
    # taken from the session base, so the message describes the whole session even
    # when the agent committed some of it internally.
    diff = (
        sandbox.diff_since(session.base_commit)
        if session.base_commit else sandbox.get_diff()
    )
    commit_message = ""
    if diff.strip() and not stopped_at_max:
        turn += 1
        header = style(
            f"--- {ralph_tag}turn {turn + turn_offset} (commit message) ---",
            "bold", "dark_red",
        )
        print("\n" + header + _context_suffix(max_context, last_usage), flush=True)

        request = _commit_request(diff)
        messages.append({"role": "user", "content": request})
        session.record_turn("user", request)

        parts, _calls, text, usage = _run_turn_with_retry(
            model, system, messages, [], provider, session=session
        )
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens
        total.thinking_tokens += usage.thinking_tokens
        last_usage = usage

        messages.append({"role": "assistant", "content": parts})
        session.record_turn(
            "assistant", text, usage.input_tokens, usage.output_tokens,
            usage.thinking_tokens,
        )
        commit_message = text.strip()
        print(flush=True)

    _print_usage(model, turn + turn_offset, total)
    return SessionResult(
        final_text=final_text, diff=diff, commit_message=commit_message
    )


def _context_suffix(max_context: int | None, usage: Usage | None) -> str:
    """A ' [context: ...]' fragment appended to a turn header, showing how full
    the window is entering the turn. `usage` is the previous turn's usage: its
    `input_tokens` is the whole conversation fed to the model and `output_tokens`
    what it generated -- together, the tokens that had to fit in the window at
    once. Empty before the first turn lands (no measurement yet). When the window
    is known, show the percentage (color escalating toward the limit); when it
    isn't (provider doesn't report it), show the raw total."""
    if usage is None:
        return ""
    used = usage.input_tokens + usage.output_tokens
    if not max_context:
        return style(f"  [context: {used:,} tokens]", "gray")
    pct = used / max_context * 100
    color = "red" if pct >= 80 else "yellow" if pct >= 50 else "gray"
    return style(f"  [context: {used:,} / {max_context:,} ({pct:.1f}%)]", color)


def _print_usage(model: str, turns: int, usage: Usage) -> None:
    """Print a summary line with token counts and estimated cost."""
    pricing = model_pricing(model)
    if pricing is None:
        cost_str = "cost: N/A"
    else:
        price_in, price_out = pricing
        cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
        cost_str = f"~${cost:.4f}"
    total = usage.input_tokens + usage.output_tokens
    # thinking_tokens is a subset of output_tokens, so show it as a breakdown.
    out = f"out: {usage.output_tokens:,}"
    if usage.thinking_tokens:
        out += f" incl. {usage.thinking_tokens:,} thinking"
    line = (
        f"--- {turns} turns, {total:,} tokens "
        f"(in: {usage.input_tokens:,}, {out}), "
        f"{cost_str} ---"
    )
    print("\n" + style(line, "bold"), flush=True)
