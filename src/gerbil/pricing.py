"""Model pricing table and cost estimation.

Best-effort USD estimates for the end-of-session usage line and `gerbil
summarize` -- never used for anything but reporting. The one rule: when a
model's pricing is unknown, report None so callers show N/A instead of a
made-up number.
"""

import sys

from .render import style


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


# Anthropic prompt-cache billing, as multiples of the model's input rate:
# reading a cached prefix costs ~0.1x, writing one costs ~1.25x (5-minute TTL).
# Applied to Usage.cache_read_tokens / cache_write_tokens, which follow the
# same exclusive semantics (input_tokens covers only the uncached remainder).
CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Estimated USD cost from per-million-token pricing, or None when the
    model's pricing is unknown -- callers must then report N/A rather than a
    made-up number. Cache reads/writes bill at their multiplier of the input
    rate; both counts are zero for providers without explicit caching, which
    reduces this to the plain input+output estimate."""
    pricing = model_pricing(model)
    if pricing is None:
        return None
    price_in, price_out = pricing
    return (
        input_tokens * price_in
        + cache_read_tokens * price_in * CACHE_READ_MULTIPLIER
        + cache_write_tokens * price_in * CACHE_WRITE_MULTIPLIER
        + output_tokens * price_out
    ) / 1_000_000
