"""Agent loop — drives turns between the LLM and the sandbox until the model
stops requesting tools.

run_session() is called by the gerbil CLI once the sandbox is ready and the
mathlib cache is warm. It uses the unified provider stream (providers.stream)
and the sandbox tools (tools.dispatch), recording every turn, tool call, and
tool result to the Session as it goes.
"""

import sys

from providers import Done, TextDelta, ToolCall, Usage, _ToolMeta, stream
from sandbox import LeanSandbox
from session import Session
from tools import TOOLS, dispatch

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


def run_session(
    sandbox: LeanSandbox,
    session: Session,
    prompt: str,
    model: str,
    provider: str | None = None,
    max_turns: int | None = None,
) -> str:
    """Run the agent loop until the model stops calling tools (or max_turns).

    Returns the final assistant text.
    """
    messages = [{"role": "user", "content": prompt}]
    session.record_turn("user", prompt)

    total = Usage()
    turn = 0
    final_text = ""
    while True:
        turn += 1
        if max_turns and turn > max_turns:
            print(f"\n[stopped: reached max_turns={max_turns}]", flush=True)
            _print_usage(model, turn - 1, total)
            return final_text

        print(f"\n--- turn {turn} ---", flush=True)

        assistant_parts = []
        current_text = ""
        tool_calls = []  # {name, args, id, raw_part}
        usage = Usage()

        for event in stream(model, SYSTEM_PROMPT, messages, TOOLS, provider):
            if isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()
                current_text += event.text
            elif isinstance(event, ToolCall):
                if current_text:
                    assistant_parts.append({"type": "text", "text": current_text})
                    current_text = ""
                print(f"\n  -> {event.name}({event.args})", flush=True)
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
        total.input_tokens += usage.input_tokens
        total.output_tokens += usage.output_tokens

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

        messages.append({"role": "assistant", "content": assistant_parts})
        final_text = "".join(
            p["text"] for p in assistant_parts if p["type"] == "text"
        )
        session.record_turn(
            "assistant", final_text, usage.input_tokens, usage.output_tokens
        )

        # No tool calls => the model is done.
        if not tool_calls:
            print(flush=True)
            _print_usage(model, turn, total)
            return final_text

        # Execute tool calls and feed results back.
        tool_results = []
        for tc in tool_calls:
            session.record_tool_call(tc["name"], tc["args"])
            result = dispatch(sandbox, tc["name"], tc["args"])
            session.record_tool_result(tc["name"], result.content)

            preview = result.content[:200] + "..." if len(result.content) > 200 else result.content
            print(f"  <- {preview}", flush=True)

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


COMMIT_SYSTEM_PROMPT = """\
You write a single git commit message for a set of changes to a Lean project.

Output format (and nothing else -- no code fences, no preamble):
  - First line: a concise imperative title, at most ~72 characters.
  - Then one blank line.
  - Then a short body explaining what changed and why, wrapped at ~72 columns.
"""


def generate_commit_message(
    model: str,
    task: str,
    diff: str,
    provider: str | None = None,
) -> str:
    """Make a single tool-less LLM call to produce a commit title + message
    describing the given diff. Returns the message text."""
    user = (
        f"The requested task was:\n{task}\n\n"
        f"The git diff of the changes that were made:\n{diff}"
    )
    messages = [{"role": "user", "content": user}]
    text = ""
    for event in stream(model, COMMIT_SYSTEM_PROMPT, messages, [], provider):
        if isinstance(event, TextDelta):
            text += event.text
    return text.strip()


def _print_usage(model: str, turns: int, usage: Usage) -> None:
    """Print a summary line with token counts and estimated cost."""
    price_in, price_out = MODEL_PRICING.get(model, DEFAULT_PRICING)
    cost = (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
    total = usage.input_tokens + usage.output_tokens
    print(
        f"\n--- {turns} turns, {total:,} tokens "
        f"(in: {usage.input_tokens:,}, out: {usage.output_tokens:,}), "
        f"~${cost:.4f} ---",
        flush=True,
    )
