"""Matching a `--model` string against a table of known models.

gerbil keeps two hand-curated tables keyed by API-ID-shaped model names --
`pricing.MODEL_PRICING` and `context_windows.CONTEXT_WINDOWS` -- and both face
the same problem: the string a user passes to `--model` is not always the bare
API id. A Portkey catalog name embeds it (`@vertexai-foo/anthropic.claude-opus-4-8`),
an ollama name prefixes it, and providers append dates and `-preview` suffixes.
So the tables are matched by substring, under one rule, defined here so the two
cannot drift apart.

A leaf module: it imports nothing from gerbil, and knows nothing about what the
tables' values mean.
"""


def table_match(model: str, keys) -> str | None:
    """The key in `keys` that identifies `model`, or None when we don't know.

    An exact key wins. Otherwise a key found *inside* the model string
    identifies it, which is what matches a gateway model: a Portkey catalog
    name like `@vertexai-foo/anthropic.claude-opus-4-7` embeds the real name.

    Keys nest within a model family (`o3` inside `o3-mini`, `gpt-5.4` inside
    `gpt-5.4-pro`), so a string like `@x/o3-mini` matches several. That isn't
    real ambiguity: when the longest matching key itself contains every other
    match, the shorter ones are just its substrings riding along, and the
    longest -- most specific -- key is the answer. Anything else (several
    matches, none subsuming the rest) means we'd be guessing, and a guessed
    number is worse than an honest "unknown" -> None.
    """
    keys = list(keys)
    if model in keys:
        return model
    matches = [key for key in keys if key in model]
    if not matches:
        return None
    longest = max(matches, key=len)
    return longest if all(key in longest for key in matches) else None
