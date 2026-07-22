"""Provider abstraction — thin wrappers over LLM APIs.

Every provider yields the same stream of events from a unified interface:

    TextDelta(text)   incremental assistant text
    ToolCall(name, args, raw_part)   a requested tool invocation
    _ToolMeta(id)     internal: provider-specific id for the preceding ToolCall
    Done(usage)       end of turn, with token usage

This lets the agent loop stay provider-agnostic. The unified message format is:

    {"role": "user", "content": str | [ {"type": "tool_result", ...} ]}
    {"role": "assistant", "content": [ {"type": "text"|"tool_call", ...} ]}

Tool schemas are dicts with name, description, and input_schema (JSON Schema) --
the same shape gerbil's tools.py produces.

Provider SDKs (anthropic, openai, google-genai, portkey-ai) are optional and
imported only when their provider is selected.

Ported from lea-prover (lea/providers.py).
"""

import os
import sys
from dataclasses import dataclass
from functools import lru_cache

ANTHROPIC_MAX_TOKENS = 16384


class TransientProviderError(RuntimeError):
    """A provider failure that is safe to retry by re-running the same turn.

    Some failures arrive not as an SDK exception but as a *finish reason* on an
    otherwise-empty final chunk -- notably Gemini's MALFORMED_RESPONSE and the
    botched-function-call reasons. With no exception raised, the stream just ends,
    and the agent loop reads a content-free, tool-call-free turn as "the model is
    done" -- silently mistaking a model failure for task completion. We raise this
    instead so _run_turn_with_retry re-streams the same turn (nothing is lost; the
    retry runs from the unchanged `messages`)."""


@lru_cache(maxsize=None)
def get_context_window(model: str, provider: str | None = None) -> int | None:
    """The model's maximum context window in total tokens, queried live from the
    provider's model-info endpoint -- or None if it can't be determined.

    Gemini (`input_token_limit`) and Anthropic (`max_input_tokens`) report it; the
    OpenAI models endpoint does not, so it returns None there (and the caller just
    reports raw token totals). No static table -- we ask the provider so the
    number can't drift out of date. Cached, so a --ralph chain queries at most
    once; never raises (any failure -> None)."""
    try:
        provider = provider or detect_provider(model)
    except ValueError:
        return None
    try:
        if provider == "gemini":
            from google import genai

            client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
            limit = getattr(client.models.get(model=model), "input_token_limit", None)
            return int(limit) if limit else None
        if provider == "anthropic":
            import anthropic

            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            limit = getattr(client.models.retrieve(model), "max_input_tokens", None)
            return int(limit) if limit else None
    except Exception:
        return None
    return None  # openai: no endpoint exposes the context window


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    # Reasoning/thinking tokens. A SUBSET of output_tokens (which stays the
    # inclusive, output-rate-billed total) -- tracked separately for reporting.
    # Populated only for providers/models that expose the breakdown; 0 otherwise,
    # in which case any thinking is silently folded into output_tokens.
    thinking_tokens: int = 0
    # Prompt-cache token counts, Anthropic semantics: these are IN ADDITION to
    # input_tokens (which then covers only the uncached remainder of the prompt).
    # Total prompt size = input + cache_read + cache_write. Reads bill at ~0.1x
    # the input rate and writes at ~1.25x (see agent.CACHE_READ_MULTIPLIER /
    # CACHE_WRITE_MULTIPLIER). Zero for providers that cache implicitly (Gemini,
    # OpenAI) or not at all -- their caching discounts are invisible here and
    # cost estimates stay conservative.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    name: str
    args: dict
    raw_part: object = None  # Provider-specific part, for faithful replay (Gemini)


@dataclass
class Done:
    usage: Usage


@dataclass
class _ToolMeta:
    """Internal: carries a provider-specific tool-call id (Anthropic tool_use_id /
    OpenAI tool_call_id) so the agent can build matching tool_result messages."""

    tool_use_id: str


def detect_provider(model: str) -> str:
    """Guess the provider from a model name."""
    if model.startswith("gemini"):
        return "gemini"
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt", "o3", "o4")):
        return "openai"
    # ollama:<NAME> selects a local model served by ollama (OpenAI-compatible).
    if model.startswith("ollama:"):
        return "ollama"
    # portkey:<MODEL> routes through a Portkey AI gateway. A bare @provider/model
    # name is Portkey's own model-catalog syntax, so it also selects portkey.
    if model.startswith(("portkey:", "@")):
        return "portkey"
    raise ValueError(
        f"Can't detect provider for model '{model}'. Pass provider explicitly."
    )


def stream(
    model: str,
    system: str,
    messages: list,
    tools: list,
    provider: str | None = None,
):
    """Yield TextDelta, ToolCall, _ToolMeta, and Done events from the model.

    messages: unified message dicts (see module docstring)
    tools: tool schema dicts (name, description, input_schema)
    """
    provider = provider or detect_provider(model)
    if provider == "gemini":
        yield from _stream_gemini(model, system, messages, tools)
    elif provider == "anthropic":
        yield from _stream_anthropic(model, system, messages, tools)
    elif provider == "openai":
        yield from _stream_openai(model, system, messages, tools)
    elif provider == "ollama":
        yield from _stream_ollama(model, system, messages, tools)
    elif provider == "portkey":
        yield from _stream_portkey(model, system, messages, tools)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _stream_gemini(model, system, messages, tools):
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    config_kwargs = {"system_instruction": system}
    if tools:
        declarations = [
            {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}
            for t in tools
        ]
        config_kwargs["tools"] = [types.Tool(function_declarations=declarations)]
    config = types.GenerateContentConfig(**config_kwargs)

    contents = []
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg["content"])]))
            elif isinstance(msg["content"], list):
                parts = []
                for item in msg["content"]:
                    if item.get("type") == "tool_result":
                        parts.append(types.Part.from_function_response(
                            name=item["tool_name"], response={"result": item["content"]},
                        ))
                    else:
                        parts.append(types.Part.from_text(text=str(item)))
                contents.append(types.Content(role="user", parts=parts))
        elif msg["role"] == "assistant":
            parts = []
            for item in msg["content"]:
                if item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item["text"]))
                elif item.get("type") == "tool_call":
                    # raw_part preserves Gemini's thought_signature when present.
                    if item.get("raw_part") is not None:
                        parts.append(item["raw_part"])
                    else:
                        part = types.Part(function_call=types.FunctionCall(
                            name=item["name"], args=item["args"],
                        ))
                        # On --resume there is no raw_part, but the signature was
                        # logged (base64); restore it so Gemini accepts the call.
                        sig = item.get("thought_signature")
                        if sig:
                            import base64

                            part.thought_signature = base64.b64decode(sig)
                        parts.append(part)
            contents.append(types.Content(role="model", parts=parts))

    usage = Usage()
    emitted = False
    finish_reason = None
    for chunk in client.models.generate_content_stream(model=model, contents=contents, config=config):
        if chunk.usage_metadata:
            um = chunk.usage_metadata
            usage.input_tokens = um.prompt_token_count or 0
            # Gemini reports thinking tokens SEPARATELY from the visible output:
            # total = prompt + candidates + tool_use_prompt + thoughts, so
            # candidates_token_count does NOT include them. Thinking tokens bill at
            # the output rate, so fold them into output_tokens (keeping that the
            # inclusive total) and record the breakdown in thinking_tokens --
            # otherwise a thinking model (e.g. the default gemini-*-pro)
            # undercounts cost.
            usage.thinking_tokens = um.thoughts_token_count or 0
            usage.output_tokens = (
                (um.candidates_token_count or 0) + usage.thinking_tokens
            )
        if not chunk.candidates:
            continue
        candidate = chunk.candidates[0]
        # The completion status rides on the final chunk's candidate. Remember it
        # so we can tell a real stop from a malformed/failed turn once the stream
        # ends (see _check_gemini_finish).
        if candidate.finish_reason is not None:
            finish_reason = candidate.finish_reason
        # A final chunk may carry only a finish reason / usage, with no content.
        content = candidate.content
        if content is None or content.parts is None:
            continue
        for part in content.parts:
            if part.text:
                emitted = True
                yield TextDelta(part.text)
            elif part.function_call:
                emitted = True
                yield ToolCall(part.function_call.name, dict(part.function_call.args), raw_part=part)

    # A malformed/empty finish must raise BEFORE Done, so the agent loop never
    # sees it as a completed (content-free) turn -- it gets retried instead.
    _check_gemini_finish(finish_reason, emitted)
    yield Done(usage)


# Gemini finish reasons that mean the turn FAILED rather than completed: the model
# produced a malformed response or bungled a function call. To the agent loop these
# look identical to an empty, tool-call-free "the model is done" turn, so we must
# catch them here and surface them as retryable.
_GEMINI_BAD_FINISH = {
    "MALFORMED_RESPONSE",
    "MALFORMED_FUNCTION_CALL",
    "UNEXPECTED_TOOL_CALL",
    "OTHER",
    "FINISH_REASON_UNSPECIFIED",
}


def _finish_reason_name(finish_reason) -> str | None:
    """Bare uppercase name of a google-genai FinishReason, e.g. 'MALFORMED_RESPONSE'.

    Handles both a normal enum member and -- for a reason the installed SDK
    predates -- the synthetic member google-genai fabricates around the raw string
    (which is exactly the MALFORMED_RESPONSE case that prompted this). Returns None
    when there is no finish reason."""
    if finish_reason is None:
        return None
    name = getattr(finish_reason, "name", None) or str(finish_reason)
    return name.rsplit(".", 1)[-1].upper()


def _check_gemini_finish(finish_reason, emitted: bool) -> None:
    """Raise TransientProviderError when Gemini ended a turn on a malformed/failed
    finish reason, or produced an empty stream with no finish reason at all. Both
    are recoverable by re-running the turn; without this they would be misread as
    a completed turn. A clean STOP/MAX_TOKENS (even with no content) is left alone
    so a genuine empty completion still ends the run as before."""
    name = _finish_reason_name(finish_reason)
    if name in _GEMINI_BAD_FINISH:
        raise TransientProviderError(f"gemini finish reason {name}")
    if not emitted and name is None:
        raise TransientProviderError("gemini returned an empty response")


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _stream_anthropic(model, system, messages, tools):
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    yield from _stream_anthropic_chat(client, model, system, messages, tools)


def _stream_anthropic_chat(client, model, system, messages, tools):
    """The Messages-API streaming core, shared by the Anthropic provider and the
    Portkey provider's native route (the gateway speaks /v1/messages too). The
    caller supplies an already-constructed anthropic client and the resolved
    model name; everything below -- message conversion, cache breakpoints,
    tool-call accumulation, usage tracking -- is identical regardless of which
    endpoint the client points at."""
    import json

    anthropic_tools = [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["input_schema"],
        }
        for t in tools
    ]

    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                anthropic_messages.append({"role": "user", "content": msg["content"]})
            elif isinstance(msg["content"], list):
                content = []
                for item in msg["content"]:
                    if item.get("type") == "tool_result":
                        content.append({
                            "type": "tool_result",
                            "tool_use_id": item["tool_use_id"],
                            "content": item["content"],
                        })
                    else:
                        content.append(item)
                anthropic_messages.append({"role": "user", "content": content})
        elif msg["role"] == "assistant":
            content = []
            for item in msg["content"]:
                if item.get("type") == "text":
                    content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "tool_call":
                    content.append({
                        "type": "tool_use",
                        "id": item["id"],
                        "name": item["name"],
                        "input": item["args"],
                    })
            anthropic_messages.append({"role": "assistant", "content": content})

    # Prompt caching. Anthropic caching is an explicit prefix match (unlike
    # Gemini/OpenAI, which cache server-side with no opt-in): content up to a
    # `cache_control` breakpoint is cached for ~5 minutes and re-served at ~0.1x
    # the input rate, with a one-time ~1.25x write premium. Two breakpoints (max
    # allowed is 4):
    #
    #   1. The system prompt -- the request renders tools -> system -> messages,
    #      so this one breakpoint caches the tool schemas AND the system prompt.
    #      Both are byte-identical across every turn of a session, which is what
    #      makes the prefix cacheable at all -- keep it that way.
    #   2. The last content block of the final message -- the "moving"
    #      breakpoint. Each turn re-sends the whole conversation; last turn's
    #      breakpoint is a prefix of this turn's request, so the entire history
    #      is a cache read and only the new suffix is written. (Anthropic finds
    #      the prior entry by scanning back up to 20 blocks from a breakpoint; a
    #      single turn would need >9 parallel tool calls to outrun that.)
    #
    # Prefixes under a model-dependent minimum (~4k tokens on Opus) silently
    # don't cache -- no error, just zero cache usage -- so this is safe even for
    # tiny conversations.
    cache_breakpoint = {"type": "ephemeral"}
    anthropic_system = [
        {"type": "text", "text": system, "cache_control": cache_breakpoint}
    ]
    if anthropic_messages:
        last = anthropic_messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        if last["content"]:
            # Stamp a copy, not the block itself: some blocks pass through from
            # the caller's conversation by reference, and a marker stuck to one
            # would still be there next turn when the block is no longer last --
            # breakpoints would accumulate until the API's limit of 4 rejects
            # the request. Rebuilding the marked block keeps each request's
            # markers to exactly this turn's two.
            last["content"] = list(last["content"])
            last["content"][-1] = {
                **last["content"][-1], "cache_control": cache_breakpoint,
            }

    usage = Usage()
    current_tool_name = None
    current_tool_json = ""
    current_tool_id = None

    stream_kwargs = dict(
        model=model,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        system=anthropic_system,
        messages=anthropic_messages,
    )
    if anthropic_tools:
        stream_kwargs["tools"] = anthropic_tools

    with client.messages.stream(**stream_kwargs) as s:
        for event in s:
            if event.type == "content_block_start":
                if event.content_block.type == "tool_use":
                    current_tool_name = event.content_block.name
                    current_tool_id = event.content_block.id
                    current_tool_json = ""
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    yield TextDelta(event.delta.text)
                elif event.delta.type == "input_json_delta":
                    current_tool_json += event.delta.partial_json
            elif event.type == "content_block_stop":
                if current_tool_name:
                    args = json.loads(current_tool_json) if current_tool_json else {}
                    yield ToolCall(current_tool_name, args)
                    yield _ToolMeta(current_tool_id)
                    current_tool_name = None
                    current_tool_json = ""
                    current_tool_id = None
            elif event.type == "message_delta":
                if hasattr(event, "usage") and event.usage:
                    # output_tokens is the inclusive billing total -- extended
                    # thinking tokens are already counted in it; the details
                    # object breaks out the thinking subset (when present).
                    usage.output_tokens = event.usage.output_tokens
                    details = getattr(event.usage, "output_tokens_details", None)
                    if details is not None:
                        usage.thinking_tokens = (
                            getattr(details, "thinking_tokens", 0) or 0
                        )
            elif event.type == "message_start":
                if hasattr(event.message, "usage") and event.message.usage:
                    mu = event.message.usage
                    # input_tokens excludes cached tokens; the cache counts are
                    # reported separately and billed at their own rates.
                    usage.input_tokens = mu.input_tokens
                    usage.cache_read_tokens = (
                        getattr(mu, "cache_read_input_tokens", 0) or 0
                    )
                    usage.cache_write_tokens = (
                        getattr(mu, "cache_creation_input_tokens", 0) or 0
                    )

    yield Done(usage)


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _stream_openai(model, system, messages, tools):
    # "-pro" reasoning models require the Responses API.
    if "-pro" in model:
        yield from _stream_openai_responses(model, system, messages, tools)
        return

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ.get("OPENAI_BASE_URL", None))
    yield from _stream_openai_chat(client, model, system, messages, tools)


def _stream_openai_chat(client, model, system, messages, tools):
    """The Chat Completions streaming core, shared by the OpenAI provider and the
    ollama provider (which speaks the same OpenAI-compatible API). The caller
    supplies an already-constructed client and the resolved model name; everything
    below -- message conversion, tool-call accumulation, usage tracking -- is
    identical regardless of which endpoint the client points at."""
    import json

    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]

    openai_messages = [{"role": "system", "content": system}]
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                openai_messages.append({"role": "user", "content": msg["content"]})
            elif isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "tool_result":
                        openai_messages.append({
                            "role": "tool",
                            "tool_call_id": item["tool_call_id"],
                            "content": item["content"],
                        })
        elif msg["role"] == "assistant":
            oai_msg = {"role": "assistant", "content": None}
            tool_calls = []
            text_parts = []
            for item in msg["content"]:
                if item.get("type") == "text":
                    text_parts.append(item["text"])
                elif item.get("type") == "tool_call":
                    tool_calls.append({
                        "id": item["id"],
                        "type": "function",
                        "function": {"name": item["name"], "arguments": json.dumps(item["args"])},
                    })
            if text_parts:
                oai_msg["content"] = "\n".join(text_parts)
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            openai_messages.append(oai_msg)

    usage = Usage()
    tool_calls_acc = {}  # index -> {id, name, args_json}

    def flush_tool_calls():
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            args = json.loads(tc["args_json"]) if tc["args_json"] else {}
            yield ToolCall(tc["name"], args)
            yield _ToolMeta(tc["id"])
        tool_calls_acc.clear()

    create_kwargs = dict(
        model=model,
        messages=openai_messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    if openai_tools:
        create_kwargs["tools"] = openai_tools

    response = client.chat.completions.create(**create_kwargs)

    for chunk in response:
        if chunk.usage:
            usage.input_tokens = chunk.usage.prompt_tokens or 0
            # completion_tokens already includes reasoning tokens (they are a
            # breakdown of it, billed at the output rate); surface that subset.
            usage.output_tokens = chunk.usage.completion_tokens or 0
            details = getattr(chunk.usage, "completion_tokens_details", None)
            if details is not None:
                usage.thinking_tokens = getattr(details, "reasoning_tokens", 0) or 0
            # Gateways translating Anthropic responses (Portkey) pass the
            # Anthropic cache counters through with their exclusive semantics
            # (prompt_tokens covers only the uncached remainder). Absent on real
            # OpenAI/ollama responses, which is correct: OpenAI's own
            # prompt_tokens_details.cached_tokens is a SUBSET of prompt_tokens
            # with different billing, and must not be double-counted here.
            usage.cache_read_tokens = (
                getattr(chunk.usage, "cache_read_input_tokens", 0) or 0
            )
            usage.cache_write_tokens = (
                getattr(chunk.usage, "cache_creation_input_tokens", 0) or 0
            )

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if delta.content:
            yield TextDelta(delta.content)

        if delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": tc.id or "", "name": "", "args_json": ""}
                if tc.id:
                    tool_calls_acc[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    tool_calls_acc[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls_acc[idx]["args_json"] += tc.function.arguments

        # Flush on ANY terminal chunk, not just finish_reason == "tool_calls".
        # Gateways translating other providers' responses can mislabel the
        # finish reason -- Portkey in its default non-strict-compliance mode
        # passes Anthropic's raw "tool_use" through verbatim -- and gating on
        # the exact label silently dropped the accumulated calls, making the
        # agent look done after its very first turn. Accumulated calls are
        # themselves the proof that tools were requested; the label is not.
        if chunk.choices[0].finish_reason is not None:
            yield from flush_tool_calls()

    # Safety net: some servers end the stream without ever setting a
    # finish_reason (or set it on a choices-less usage chunk we skip above).
    yield from flush_tool_calls()

    yield Done(usage)


def _stream_ollama(model, system, messages, tools):
    """Local model served by ollama. ollama exposes an OpenAI-compatible Chat
    Completions endpoint (default http://localhost:11434/v1), so we point an
    OpenAI client at it and reuse the shared streaming core. The `ollama:` prefix
    is stripped so the real model name reaches ollama; it stays attached
    everywhere else (logs, pricing, banners) as the model's identity. No API key
    is required -- a non-empty placeholder keeps the SDK happy."""
    from openai import OpenAI

    from .ollama import ollama_base_url, ollama_model_name

    client = OpenAI(api_key="ollama", base_url=ollama_base_url() + "/v1")
    yield from _stream_openai_chat(
        client, ollama_model_name(model), system, messages, tools
    )


def portkey_model_name(model: str) -> str:
    """The model name the Portkey gateway expects, with any `portkey:` prefix
    stripped. Only the first `portkey:` is removed, so a catalog name like
    `portkey:@vertexai-foo/anthropic.claude-opus-4-8` survives intact as
    `@vertexai-foo/anthropic.claude-opus-4-8`. A bare `@provider/model` name
    (already in gateway syntax) passes through unchanged."""
    return model[len("portkey:") :] if model.startswith("portkey:") else model


def _portkey_gateway_root(base_url: str) -> str:
    """The gateway's root URL, for SDKs that append their own path. The
    OpenAI-compatible route lives under `{root}/v1/chat/completions` and
    PORTKEY_BASE_URL conventionally includes the `/v1`; the anthropic SDK
    appends `/v1/messages` itself, so it needs the bare root."""
    root = (base_url or "https://api.portkey.ai").rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def _portkey_split_catalog(name: str) -> tuple[str | None, str]:
    """Split a Portkey model-catalog name `@provider/model` into its provider
    slug and bare model name. Non-catalog names pass through with no provider."""
    if name.startswith("@") and "/" in name:
        provider, bare = name.split("/", 1)
        return provider, bare
    return None, name


def _portkey_route_missing(exc) -> bool:
    """Whether an error from the gateway means the /v1/messages route itself is
    absent (older gateway build) rather than the request being bad -- only then
    is falling back to the chat-completions dialect the right move."""
    import anthropic

    return isinstance(exc, anthropic.APIStatusError) and exc.status_code in (
        404, 405, 501,
    )


def _stream_portkey_anthropic(api_key, base_url, name, system, messages, tools):
    """Anthropic-family model through the gateway's native /v1/messages route.
    Portkey serves the Anthropic Messages API directly (it's how Claude Code
    runs through Portkey), so we point an anthropic client at the gateway and
    reuse the Anthropic streaming core. This matters for two reasons: the
    chat-completions translation drops `cache_control` on tool_result messages
    (so the moving cache breakpoint -- the one that makes the growing
    conversation cheap -- can't survive it), and the native route needs no
    finish_reason translation at all. Auth follows Portkey's documented shape:
    the Portkey key in `x-portkey-api-key`, the provider slug from the catalog
    name in `x-portkey-provider`, and the bare model name in the request."""
    import anthropic

    provider, bare_model = _portkey_split_catalog(name)
    headers = {"x-portkey-api-key": api_key}
    if provider:
        headers["x-portkey-provider"] = provider
    client = anthropic.Anthropic(
        auth_token=api_key,
        base_url=_portkey_gateway_root(base_url),
        default_headers=headers,
    )
    yield from _stream_anthropic_chat(client, bare_model, system, messages, tools)


def _stream_portkey(model, system, messages, tools):
    """Model served through a Portkey AI gateway (https://portkey.ai).

    Anthropic-family models take the gateway's native /v1/messages route (see
    _stream_portkey_anthropic -- prompt caching and native stop reasons); if
    the gateway predates that route, they fall back to the OpenAI-compatible
    dialect below with a warning. Everything else speaks the OpenAI Chat
    Completions dialect: the portkey-ai SDK's streaming chunks are
    field-compatible with openai's, so we hand a Portkey client to the shared
    streaming core unchanged.

    Auth comes from PORTKEY_API_KEY; a self-hosted/enterprise gateway (the
    usual reason to use Portkey at all) is selected with PORTKEY_BASE_URL --
    unset, the SDK targets Portkey's hosted service. The `portkey:` prefix (if
    any) is stripped so the gateway sees its own model name; the full prefixed
    name stays attached everywhere else (logs, pricing, banners) as the
    model's identity."""
    try:
        api_key = os.environ["PORTKEY_API_KEY"]
    except KeyError:
        raise RuntimeError(
            "portkey models need the PORTKEY_API_KEY environment variable"
        ) from None
    base_url = os.environ.get("PORTKEY_BASE_URL", "").strip()
    name = portkey_model_name(model)

    if "claude" in name.lower():
        native = _stream_portkey_anthropic(
            api_key, base_url, name, system, messages, tools
        )
        started = False
        try:
            for event in native:
                started = True
                yield event
            return
        except Exception as exc:
            # Fall back only when the route itself is missing and nothing has
            # been yielded yet -- once events are out, a retry would duplicate
            # them, and any other error would just recur on the fallback.
            if started or not _portkey_route_missing(exc):
                raise
            print(
                "warning: gateway has no /v1/messages route; falling back to "
                "the chat-completions dialect (prompt caching unavailable)",
                file=sys.stderr,
            )

    from portkey_ai import Portkey

    # strict_open_ai_compliance makes the gateway normalize provider-native
    # finish reasons to the OpenAI vocabulary (e.g. Anthropic's "tool_use" ->
    # "tool_calls"). The SDK defaults it to False, which leaks raw provider
    # labels into the stream; the shared core tolerates that now, but ask for
    # the spec-compliant dialect anyway since that is what we parse against.
    client_kwargs = {"api_key": api_key, "strict_open_ai_compliance": True}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = Portkey(**client_kwargs)
    yield from _stream_openai_chat(client, name, system, messages, tools)


def _stream_openai_responses(model, system, messages, tools):
    """OpenAI Responses API path — required for gpt-*-pro reasoning models."""
    import json

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Responses API tool schema: flat, no "function" wrapper.
    openai_tools = [
        {
            "type": "function",
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        }
        for t in tools
    ]

    input_items = []
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                input_items.append({"role": "user", "content": msg["content"]})
            elif isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "tool_result":
                        content = item["content"]
                        if not isinstance(content, str):
                            content = json.dumps(content)
                        input_items.append({
                            "type": "function_call_output",
                            "call_id": item["tool_call_id"],
                            "output": content,
                        })
        elif msg["role"] == "assistant":
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text" and item.get("text"):
                        input_items.append({"role": "assistant", "content": item["text"]})
                    elif item.get("type") == "tool_call":
                        input_items.append({
                            "type": "function_call",
                            "call_id": item["id"],
                            "name": item["name"],
                            "arguments": json.dumps(item["args"]),
                        })

    usage = Usage()
    pending_calls = {}  # item_id -> {call_id, name, args_json}

    create_kwargs = dict(
        model=model,
        instructions=system,
        input=input_items,
        stream=True,
    )
    if openai_tools:
        create_kwargs["tools"] = openai_tools

    s = client.responses.create(**create_kwargs)

    for event in s:
        etype = getattr(event, "type", "")

        if etype == "response.output_text.delta":
            yield TextDelta(event.delta)

        elif etype == "response.output_item.added":
            item = getattr(event, "item", None)
            if item is not None and getattr(item, "type", "") == "function_call":
                pending_calls[item.id] = {
                    "call_id": item.call_id,
                    "name": item.name,
                    "args_json": "",
                }

        elif etype == "response.function_call_arguments.delta":
            item_id = getattr(event, "item_id", None)
            if item_id in pending_calls:
                pending_calls[item_id]["args_json"] += event.delta

        elif etype == "response.output_item.done":
            item = getattr(event, "item", None)
            if item is not None and getattr(item, "type", "") == "function_call":
                info = pending_calls.pop(item.id, None)
                if info is not None:
                    args_json = info["args_json"] or getattr(item, "arguments", "") or ""
                    args = json.loads(args_json) if args_json else {}
                    yield ToolCall(info["name"], args)
                    yield _ToolMeta(info["call_id"])

        elif etype == "response.completed":
            resp = getattr(event, "response", None)
            u = getattr(resp, "usage", None) if resp is not None else None
            if u is not None:
                usage.input_tokens = getattr(u, "input_tokens", 0) or 0
                # output_tokens already includes reasoning tokens (a breakdown of
                # it), so it is the correct output-rate total for these models;
                # surface the reasoning subset from the details breakdown.
                usage.output_tokens = getattr(u, "output_tokens", 0) or 0
                details = getattr(u, "output_tokens_details", None)
                if details is not None:
                    usage.thinking_tokens = getattr(details, "reasoning_tokens", 0) or 0

    yield Done(usage)
