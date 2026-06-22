"""Agent loop — drives turns between the LLM and the sandbox until the model
stops requesting tools.

run_session() is called by the gerbil CLI once the sandbox is ready and the
mathlib cache is warm. It uses the unified provider stream (providers.stream)
and the sandbox tools (tools.dispatch), recording every turn, tool call, and
tool result to the Session as it goes.
"""

import sys
from dataclasses import dataclass

from .providers import Done, TextDelta, ToolCall, Usage, _ToolMeta, stream
from .sandbox import LeanSandbox
from .session import Session
from .term import style

# Single accent color for every tool invocation.
TOOL_COLOR = "cyan"
from .tools import TOOLS, dispatch

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
  - bash: run shell commands (use `lake build` to check that the project compiles)
  - read_file: read a file's contents
  - write_file: create or overwrite a file
  - edit_file: replace an exact string in a file

Guidelines:
  - Explore the project before editing: read the relevant files first.
  - After making changes, run `lake build` and fix any errors it reports.
  - Do not leave `sorry` in proofs unless the task explicitly allows it.
  - When the task is complete and the project builds, stop and give a short \
summary of what you did. Do not call any more tools once you are done.
"""


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


def _run_turn(model, system, messages, tools, provider):
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
                f"\n  {style('->', TOOL_COLOR)} "
                f"{style(event.name, 'bold', TOOL_COLOR)}({event.args})",
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


def run_session(
    sandbox: LeanSandbox,
    session: Session,
    prompt: str,
    model: str,
    provider: str | None = None,
    max_turns: int | None = None,
) -> SessionResult:
    """Run the agent loop until the model stops calling tools (or max_turns).

    When the model finishes the task naturally, one more turn is appended to the
    same conversation asking it to write a commit message for its diff; that turn
    is recorded and counted like any other.
    """
    messages = [{"role": "user", "content": prompt}]
    session.record_turn("user", prompt)

    total = Usage()
    turn = 0
    final_text = ""
    stopped_at_max = False

    while True:
        if max_turns and turn >= max_turns:
            stopped_at_max = True
            break
        turn += 1

        print("\n" + style(f"--- turn {turn} ---", "bold", "dark_red"), flush=True)

        assistant_parts, tool_calls, final_text, usage = _run_turn(
            model, SYSTEM_PROMPT, messages, TOOLS, provider
        )
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens

        messages.append({"role": "assistant", "content": assistant_parts})
        session.record_turn(
            "assistant", final_text, usage.input_tokens, usage.output_tokens
        )

        # No tool calls => the model is done with the task.
        if not tool_calls:
            print(flush=True)
            break

        # Execute tool calls and feed results back.
        tool_results = []
        for tc in tool_calls:
            session.record_tool_call(tc["name"], tc["args"])
            result = dispatch(sandbox, tc["name"], tc["args"])
            session.record_tool_result(tc["name"], result.content)

            preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
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
                "content": result.content,
            }
            # Anthropic keys on tool_use_id, OpenAI on tool_call_id; set both.
            if tc["id"]:
                tr["tool_use_id"] = tc["id"]
                tr["tool_call_id"] = tc["id"]
            tool_results.append(tr)

        messages.append({"role": "user", "content": tool_results})

    if stopped_at_max:
        print(
            "\n" + style(f"[stopped: reached max_turns={max_turns}]", "bold", "yellow"),
            flush=True,
        )

    # Final turn: ask for a commit message as a true continuation of the
    # conversation. Skipped if nothing changed, or if we bailed on max_turns
    # (the work is incomplete and the history ends on a tool result).
    diff = sandbox.get_diff()
    commit_message = ""
    if diff.strip() and not stopped_at_max:
        turn += 1
        print(
            "\n" + style(f"--- turn {turn} (commit message) ---", "bold", "dark_red"),
            flush=True,
        )

        request = _commit_request(diff)
        messages.append({"role": "user", "content": request})
        session.record_turn("user", request)

        parts, _calls, text, usage = _run_turn(
            model, SYSTEM_PROMPT, messages, [], provider
        )
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens

        messages.append({"role": "assistant", "content": parts})
        session.record_turn(
            "assistant", text, usage.input_tokens, usage.output_tokens
        )
        commit_message = text.strip()
        print(flush=True)

    _print_usage(model, turn, total)
    return SessionResult(
        final_text=final_text, diff=diff, commit_message=commit_message
    )


def _print_usage(model: str, turns: int, usage: Usage) -> None:
    """Print a summary line with token counts and estimated cost."""
    price_in, price_out = MODEL_PRICING.get(model, DEFAULT_PRICING)
    cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
    total = usage.input_tokens + usage.output_tokens
    line = (
        f"--- {turns} turns, {total:,} tokens "
        f"(in: {usage.input_tokens:,}, out: {usage.output_tokens:,}), "
        f"~${cost:.4f} ---"
    )
    print("\n" + style(line, "bold"), flush=True)
