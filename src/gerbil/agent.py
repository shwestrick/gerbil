"""Agent loop — drives turns between the LLM and the sandbox until the model
stops requesting tools.

run_session() is called by the gerbil CLI once the sandbox is ready and the
mathlib cache is warm. It uses the unified provider stream (providers.stream)
and the sandbox tools (tools.dispatch), recording every turn, tool call, and
tool result to the Session as it goes.
"""

import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .providers import (
    Done,
    TextDelta,
    ToolCall,
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
# estimates, used only for the cost summary. Ported from lea-prover.
MODEL_PRICING = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-3-pro-preview": (1.25, 10.0),
    "gemini-3.1-pro-preview": (1.25, 10.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-4o": (2.50, 10.0),
    "gpt-5.4-pro-2026-03-05": (2.5, 15.0),
    "o3": (2.0, 8.0),
    "o4-mini": (1.10, 4.40),
}
DEFAULT_PRICING = (2.0, 10.0)

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


# Cosmetic limits for rendering write_file / edit_file tool calls in the
# terminal. Display-only: they never affect what is written, dispatched, or
# recorded to the session log.
WRITE_FILE_INLINE_MAX_LINES = 10     # show contents inline at/below this
WRITE_FILE_INLINE_MAX_CHARS = 2000   # ...and only if not too large overall
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
    if len(lines) <= WRITE_FILE_INLINE_MAX_LINES and len(content) <= WRITE_FILE_INLINE_MAX_CHARS:
        body = "\n".join(
            _BODY_INDENT + _gutter(i) + style(_clip(ln), "gray")
            for i, ln in enumerate(lines, 1)
        )
        return f"{path}\n{body}"
    n = len(lines)
    summary = f"({n} line{'' if n == 1 else 's'}, " \
              f"{len(content.encode('utf-8', 'replace'))} bytes)"
    return f"{path} {style(summary, 'gray')}"


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
    """lean_run_code runs a standalone Lean snippet. Show the code (with a
    line-number gutter) when short, or a summary when long, instead of dumping
    the whole `code` string."""
    code = args["code"]
    lines = code.splitlines()
    if not lines:
        return style("(empty)", "gray")
    if len(lines) == 1:
        return style(_clip(lines[0]), "gray")
    if len(lines) <= SNIPPET_INLINE_MAX_LINES and len(code) <= SNIPPET_INLINE_MAX_CHARS:
        body = "\n".join(
            _BODY_INDENT + _gutter(i) + style(_clip(ln), "gray")
            for i, ln in enumerate(lines, 1)
        )
        return f"{style(f'({len(lines)} lines)', 'dim')}\n{body}"
    first = next((ln for ln in lines if ln.strip()), lines[0])
    summary = f"({len(lines)} lines, {len(code)} chars)"
    return f"{style(summary, 'dim')} {style(_clip(first), 'gray')}"


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

        assistant_parts, tool_calls, final_text, usage = _run_turn(
            model, system, messages, tools, provider, sandbox.read_file
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

            preview = content[:200] + "..." if len(content) > 200 else content
            result_color = "red" if result.is_error else "gray"
            # Align continuation lines under the content (after "  <- ").
            indented = preview.rstrip("\n").replace("\n", "\n     ")
            print(
                f"  {style('<-', result_color)} {style(indented, result_color)}",
                flush=True,
            )

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

        parts, _calls, text, usage = _run_turn(
            model, system, messages, [], provider
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
    price_in, price_out = MODEL_PRICING.get(model, DEFAULT_PRICING)
    cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
    total = usage.input_tokens + usage.output_tokens
    # thinking_tokens is a subset of output_tokens, so show it as a breakdown.
    out = f"out: {usage.output_tokens:,}"
    if usage.thinking_tokens:
        out += f" incl. {usage.thinking_tokens:,} thinking"
    line = (
        f"--- {turns} turns, {total:,} tokens "
        f"(in: {usage.input_tokens:,}, {out}), "
        f"~${cost:.4f} ---"
    )
    print("\n" + style(line, "bold"), flush=True)
