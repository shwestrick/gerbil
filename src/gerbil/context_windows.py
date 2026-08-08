"""Context-window sizes for known models -- the fallback for when the provider
won't tell us.

`providers.get_context_window` asks the provider's own model-info endpoint
first, so the number can't drift out of date. But only Gemini and Anthropic
expose one: the OpenAI models endpoint doesn't, gateways don't, and a local
ollama model has no endpoint to ask. Those used to report nothing at all, which
means the session banner and the context-usage line had no denominator.

This table is the backup for exactly those cases. It is a snapshot, so it *can*
go stale -- which is why it is consulted only after the live query has failed,
never instead of it.

Source: https://benchlm.ai/llm-pricing (its `/api/data/pricing` endpoint),
snapshot of August 7, 2026 -- the same source pricing.py's table comes from. To
refresh, re-read that endpoint; `contextWindow` arrives as a display string
("200K", "1.05M") which is expanded to tokens here.

Keys are API-ID-shaped, not benchlm's display names, so that the model strings
users actually pass to `--model` match them -- see model_match.table_match for
the rule. benchlm writes some Anthropic models "<version> <tier>" ("Claude 4.1
Opus") where the API is always "<tier>-<version>" (`claude-opus-4-1`); those
are transcribed to the API's order. Entries the source lists twice with
different windows (it splits "GPT-5" into a 128K and a 400K variant) are left
out entirely rather than guessed at.

Known gaps, i.e. models gerbil prices but this source does not list: OpenAI's
o-series (`o3`, `o3-pro`, `o3-mini`, `o4-mini`) and `claude-opus-4-20250514`.
They report None, as they did before this table existed. Nothing is entered
here by hand from memory -- an invented context window is worse than no
denominator, and it would silently survive every refresh of the snapshot.
"""

from .model_match import table_match

# Model name -> context window in tokens.
CONTEXT_WINDOWS = {
    # Aion Labs
    "aion-2.0": 128000,

    # Anthropic
    "claude-3-5-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3-opus": 200000,
    "claude-fable-5": 1000000,
    "claude-haiku-4-5": 200000,
    "claude-mythos-5": 1000000,
    "claude-opus-4-1": 200000,
    "claude-opus-4-5": 200000,
    "claude-opus-4-6": 1000000,
    "claude-opus-4-7": 1000000,
    "claude-opus-4-8": 1000000,
    "claude-opus-5": 1000000,
    "claude-sonnet-4": 200000,
    "claude-sonnet-4-5": 200000,
    "claude-sonnet-4-6": 200000,
    "claude-sonnet-5": 1000000,

    # Celeris
    "celeris-1": 128000,

    # Cohere
    "command-a+": 128000,

    # Cursor
    "composer-2": 200000,
    "composer-2.5": 200000,

    # Databricks
    "dbrx-instruct": 32000,

    # DeepSeek
    "deepseek-coder-2.0": 128000,
    "deepseek-llm-2.0": 128000,
    "deepseek-r1": 128000,
    "deepseek-r1-distill-qwen-32b": 128000,
    "deepseek-v3": 128000,
    "deepseek-v3.1": 128000,
    "deepseek-v3.2": 128000,
    "deepseek-v4-flash": 1000000,
    "deepseek-v4-flash-base": 1000000,
    "deepseek-v4-pro": 1000000,
    "deepseek-v4-pro-base": 1000000,
    "deepseekmath-v2": 128000,

    # Google
    "gemini-1.0-pro": 32000,
    "gemini-1.5-pro": 1000000,
    "gemini-2.5-flash": 1000000,
    "gemini-2.5-flash-lite": 1000000,
    "gemini-2.5-pro": 1000000,
    "gemini-3-flash": 1000000,
    "gemini-3-pro": 2000000,
    "gemini-3-pro-deep-think": 2000000,
    "gemini-3.1-flash-lite": 1000000,
    "gemini-3.1-pro": 1000000,
    "gemini-3.5-flash": 1000000,
    "gemini-3.5-flash-lite": 1000000,
    "gemini-3.6-flash": 1000000,
    "gemma-3-27b": 32000,
    "gemma-4-26b-a4b": 256000,
    "gemma-4-31b": 256000,
    "gemma-4-e2b": 128000,
    "gemma-4-e4b": 128000,

    # H Company
    "holo3-122b-a10b": 64000,
    "holo3-35b-a3b": 64000,
    "holo3.1-0.8b": 262000,
    "holo3.1-35b-a3b": 64000,
    "holo3.1-35b-a3b-fp8": 262000,
    "holo3.1-35b-a3b-gguf": 262000,
    "holo3.1-35b-a3b-nvfp4": 262000,
    "holo3.1-4b": 262000,
    "holo3.1-9b": 262000,

    # IBM
    "granite-4.0-1b": 128000,
    "granite-4.0-350m": 32000,
    "granite-4.0-h-1b": 128000,
    "granite-4.0-h-350m": 32000,

    # InclusionAI
    "ling-2.6-flash": 262000,
    "ling-3.0-flash": 262000,
    "ling-3.0-flash-fp8": 262000,

    # Interfaze
    "interfaze-beta": 1000000,

    # LiquidAI
    "lfm2-24b-a2b": 32000,
    "lfm2.5-1.2b-instruct": 32000,
    "lfm2.5-1.2b-thinking": 32000,
    "lfm2.5-2.6b": 128000,
    "lfm2.5-230m": 32000,
    "lfm2.5-350m": 32000,
    "lfm2.5-8b-a1b": 128000,
    "lfm2.5-colbert-350m": 32000,
    "lfm2.5-embedding-350m": 32000,
    "lfm2.5-vl-450m": 128000,

    # Meituan
    "longcat-2.0": 1000000,

    # Meta
    "llama-3-70b": 128000,
    "llama-3.1-405b": 128000,
    "llama-4-behemoth": 32000,
    "llama-4-maverick": 1000000,
    "llama-4-scout": 10000000,

    # Mistral
    "leanstral": 256000,

    # Moonshot AI
    "kimi-2.6": 256000,
    "kimi-k2": 128000,
    "kimi-k2.5": 256000,
    "kimi-k2.7-code": 256000,
    "kimi-k3": 1050000,

    # NVIDIA
    "cosmos3-edge": 256000,

    # OpenAI
    "gpt-4-turbo": 128000,
    "gpt-4.1": 1000000,
    "gpt-4.1-mini": 1000000,
    "gpt-4.1-nano": 1000000,
    "gpt-4o": 128000,
    "gpt-4o-audio": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-audio": 128000,
    "gpt-5-mini": 128000,
    "gpt-5-nano": 400000,
    "gpt-5.1": 400000,
    "gpt-5.1-codex-max": 400000,
    "gpt-5.2": 400000,
    "gpt-5.2-codex": 400000,
    "gpt-5.2-instant": 128000,
    "gpt-5.2-pro": 400000,
    "gpt-5.3-codex": 400000,
    "gpt-5.3-codex-spark": 256000,
    "gpt-5.3-instant": 128000,
    "gpt-5.4": 1050000,
    "gpt-5.4-mini": 400000,
    "gpt-5.4-nano": 400000,
    "gpt-5.4-pro": 1050000,
    "gpt-5.5": 1000000,
    "gpt-5.5-pro": 1000000,
    "gpt-5.6-luna": 1050000,
    "gpt-5.6-sol": 1050000,
    "gpt-5.6-terra": 1050000,
    "gpt-oss-120b": 128000,
    "gpt-oss-20b": 128000,
    "gpt-realtime": 32000,
    "gpt-realtime-1.5": 32000,
    "gpt-realtime-2": 128000,
    "gpt-realtime-mini": 32000,

    # Poolside
    "laguna-m.1": 256000,
    "laguna-s-2.1": 1000000,
    "laguna-xs.2": 256000,

    # Prism ML
    "1-bit-bonsai-1.7b": 32000,
    "1-bit-bonsai-4b": 32000,
    "1-bit-bonsai-8b": 64000,

    # Sakana AI
    "fugu-cyber": 1000000,

    # Tencent
    "hy3": 256000,
    "hy3-preview": 256000,

    # Thinking Machines Lab
    "inkling": 1000000,
    "inkling-small": 1000000,

    # Z.AI
    "glm-4.5": 128000,
    "glm-4.5-air": 128000,
    "glm-4.7": 200000,
    "glm-4.7-flash": 200000,
    "glm-5": 200000,
    "glm-5-turbo": 200000,
    "glm-5.1": 203000,
    "glm-5.2": 1000000,
    "glm-5v-turbo": 200000,

    # xAI
    "grok-3": 128000,
    "grok-3-mini": 128000,
    "grok-4": 128000,
    "grok-4.1": 1000000,
    "grok-4.1-fast": 2000000,
    "grok-4.20": 2000000,
    "grok-4.20-multi-agent": 2000000,
    "grok-4.3": 1000000,
    "grok-4.5": 500000,
    "grok-build-0.1": 256000,
    "grok-code-fast-1": 256000,
}


def context_window(model: str) -> int | None:
    """The table's context window for `model`, or None when it isn't listed (or
    the name is ambiguous between entries). Never raises."""
    key = table_match(model, CONTEXT_WINDOWS)
    return CONTEXT_WINDOWS[key] if key is not None else None
