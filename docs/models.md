# Models and API keys

The model is chosen with `--model`; the provider is auto-detected from the
name. Run `gerbil run --help` for the list of known models.

```console
$ gerbil run --model gemini-3.1-pro-preview --prompt prompt.md
```

Set the API key for whichever provider you're using:

```console
$ export GOOGLE_API_KEY=...        # gemini-*
$ export ANTHROPIC_API_KEY=...     # claude-*
$ export OPENAI_API_KEY=...        # gpt-*, o3, o4-mini, ...
```

gerbil speaks to every provider through one unified streaming interface, so
tool calling, thinking-token accounting, and the transient-error retry behave
the same regardless of which one you pick. Only the selected provider's SDK is
imported at runtime.

## Local models with ollama

To run against a local model served by [ollama](https://ollama.com), use the
`ollama:<NAME>` model syntax -- no API key required:

```console
$ ollama pull qwen2.5-coder
$ gerbil run --model ollama:qwen2.5-coder --prompt prompt.md
```

The model must already be pulled locally. If no ollama server is running,
gerbil starts `ollama serve` as a child process and stops it when the session
ends. Set `OLLAMA_HOST` to point at a non-default address (for example, a
remote box with a real GPU).

Note that ollama runs on the **host**, not in the sandbox -- the container
holds the Lean toolchain, not the model.

## Portkey gateways

gerbil supports [Portkey](https://app.portkey.ai), which is useful when your
models sit behind an organizational gateway:

```console
$ export PORTKEY_API_KEY=...
$ export PORTKEY_BASE_URL=...    # e.g. https://your.gateway.com/v1/  (self-hosted only)
$ gerbil run --model portkey:<MODEL_CATALOG_STRING> --prompt prompt.md
```

A bare catalog name beginning with `@` (for example `@anthropic/claude-opus-4-5`)
is recognized as Portkey too, so the `portkey:` prefix is optional in that
case. `PORTKEY_BASE_URL` is only needed for a self-hosted gateway.

## Stalled streams

A gateway (or a proxy in front of one) can finish a response upstream and then
sit on it, leaving gerbil's connection open but silent. gerbil gives up on a
stream that produces nothing for **120 seconds** and retries the turn.

The limit is on *silence between chunks*, not on the length of a response -- a
turn that streams for ten minutes is fine. It is deliberately far below the
provider SDKs' own 600s default, because the wait is not just lost time: the
retry re-runs the same turn, and Anthropic's prompt cache expires after about
five minutes, so a ten-minute stall guarantees the retry re-writes the whole
conversation prefix at the cache-write premium instead of re-reading it at a
tenth of the input rate. Catching the stall at 120s keeps the retry inside the
cache's lifetime.

`GERBIL_STREAM_TIMEOUT` overrides the number of seconds; `0` restores the SDK
default of effectively waiting it out. The retry warning reports how long the
failed attempt ran, so a stall is easy to tell apart from an instant `503`:

```
[provider unavailable after 120s: ReadTimeout: The read operation timed out; retrying in 5s (attempt 1)]
```

This does not apply to `ollama:` models, where a local server legitimately goes
quiet for minutes while loading a model and there is no metered cache to lose.

## Context window

gerbil shows how much of the model's context window a session is using. It asks
the provider for the number first, so it can't go stale -- but only Gemini and
Anthropic publish one. For everything else (OpenAI, gateway models, local
ollama models) gerbil falls back to a built-in table snapshotted from
[benchlm.ai/llm-pricing](https://benchlm.ai/llm-pricing), matched against the
model name -- including a name buried inside a Portkey catalog string like
`@vertexai-foo/anthropic.claude-opus-4-8`.

A model the table doesn't list, or one whose name matches several entries
ambiguously, reports no context window at all rather than a guessed one; the
usage line then shows raw token totals. Refresh the table from that site's
`/api/data/pricing` endpoint (see `src/gerbil/context_windows.py`).

## Cost

gerbil prints a running token count and an estimated cost at the end of each
session, using a built-in per-model pricing table. Models missing from that
table simply report `N/A` rather than a wrong number. See
[Usage and cost](summarize.md) for the cross-session view.

## Big-small mode

A large model can drive the session while a smaller, cheaper one grinds through
the mechanical parts of individual proofs. See
[Big-small mode](big-small.md).
