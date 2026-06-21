"""Live test of the Gemini backend in providers.py.

Exercises the full event interface and message round-trip:
  1. stream a turn where the model must call a tool
  2. replay the tool_call (with Gemini's raw_part) into the assistant message
  3. feed a tool_result back
  4. stream the follow-up turn and confirm the model uses the result

Requires GOOGLE_API_KEY in the environment and the gemini extra installed:

    uv sync --extra gemini
    GOOGLE_API_KEY=...  uv run python tests/test_gemini.py
    # optional: choose a model (default: gemini-2.5-flash)
    GOOGLE_API_KEY=...  uv run python tests/test_gemini.py gemini-2.5-pro
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from providers import Done, TextDelta, ToolCall, Usage, _ToolMeta, stream

MODEL = sys.argv[1] if len(sys.argv) > 1 else "gemini-2.5-flash"

# One fake tool, in gerbil's tool-schema shape.
TOOLS = [
    {
        "name": "multiply",
        "description": "Multiply two integers and return the product.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    }
]


def run_turn(system, messages):
    """Consume one streamed turn into a unified assistant message.

    Returns (assistant_message, tool_calls, usage). Mirrors the accumulation
    logic the real agent loop will use.
    """
    parts = []
    text = ""
    tool_calls = []  # {name, args, id, raw_part}
    usage = Usage()

    for event in stream(MODEL, system, messages, TOOLS, provider="gemini"):
        if isinstance(event, TextDelta):
            sys.stdout.write(event.text)
            sys.stdout.flush()
            text += event.text
        elif isinstance(event, ToolCall):
            if text:
                parts.append({"type": "text", "text": text})
                text = ""
            print(f"\n  -> {event.name}({event.args})")
            tool_calls.append(
                {"name": event.name, "args": event.args, "id": None, "raw_part": event.raw_part}
            )
        elif isinstance(event, _ToolMeta):
            if tool_calls:
                tool_calls[-1]["id"] = event.tool_use_id
        elif isinstance(event, Done):
            usage = event.usage

    if text:
        parts.append({"type": "text", "text": text})
    for tc in tool_calls:
        parts.append({
            "type": "tool_call",
            "name": tc["name"],
            "args": tc["args"],
            "id": tc["id"],
            "raw_part": tc["raw_part"],
        })

    return {"role": "assistant", "content": parts}, tool_calls, usage


def main():
    if "GOOGLE_API_KEY" not in os.environ:
        sys.exit("error: set GOOGLE_API_KEY in the environment")

    system = "You are a helpful assistant. Use the multiply tool when asked to multiply."
    messages = [{"role": "user", "content": "Use the multiply tool to compute 21 times 2, then state the result."}]

    print(f"=== model: {MODEL} ===")
    print("\n--- turn 1 ---")
    assistant_msg, tool_calls, usage1 = run_turn(system, messages)
    messages.append(assistant_msg)

    assert tool_calls, "FAIL: model did not call the tool"
    tc = tool_calls[0]
    assert tc["name"] == "multiply", f"FAIL: unexpected tool {tc['name']}"
    a, b = tc["args"]["a"], tc["args"]["b"]
    product = a * b
    print(f"  <- {product}")

    # Feed the tool result back (Gemini path keys off tool_name).
    messages.append({
        "role": "user",
        "content": [{"type": "tool_result", "tool_name": tc["name"], "content": str(product)}],
    })

    print("\n--- turn 2 ---")
    final_msg, more_calls, usage2 = run_turn(system, messages)
    final_text = "".join(p["text"] for p in final_msg["content"] if p["type"] == "text")

    print("\n\n--- checks ---")
    ok_tool = (a, b) in [(21, 2), (2, 21)]
    ok_answer = "42" in final_text
    print(f"[{'PASS' if ok_tool else 'FAIL'}] tool called with 21 and 2 (got a={a}, b={b})")
    print(f"[{'PASS' if ok_answer else 'FAIL'}] final answer mentions 42")
    total_in = usage1.input_tokens + usage2.input_tokens
    total_out = usage1.output_tokens + usage2.output_tokens
    print(f"[{'PASS' if total_in and total_out else 'FAIL'}] usage reported (in={total_in}, out={total_out})")

    if ok_tool and ok_answer and total_in and total_out:
        print("\nGemini backend OK.")
    else:
        sys.exit("\nGemini backend test FAILED.")


if __name__ == "__main__":
    main()
