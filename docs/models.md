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

gerbil shows how much of the model's context window a session is using. It
works out the number from three sources, most trustworthy first:

1. **The provider**, asked live. Only Gemini and Anthropic publish a context
   window; the OpenAI models endpoint doesn't, gateways don't, and a local
   ollama model has no endpoint to ask.
2. **`~/.gerbil/context-windows.jsonl`**, gerbil's own record of every answer
   it has ever gotten from step 1. A model queried once stays known afterwards
   -- on a later run with no API key, or through a gateway.
3. **A built-in table**, snapshotted from
   [benchlm.ai/llm-pricing](https://benchlm.ai/llm-pricing), for a model never
   seen live.

All three match against the model name, including a name buried inside a
Portkey catalog string like `@vertexai-foo/anthropic.claude-opus-4-8` -- which
is how an observation logged for `claude-opus-4-8` also answers for the gateway
route to it.

A model none of them covers, or one whose name matches several table entries
ambiguously, reports no context window rather than a guessed one; the usage
line then shows raw token totals.

### The observation log

Each line is one *change*: `{"timestamp", "model", "context_window",
"provider"}`. A model whose window never moves gets one line ever, so the file
stays a readable history of when windows actually changed rather than a
per-run journal.

When a logged observation contradicts the built-in table, every `gerbil run`
and `gerbil resume` says so up front:

```
warning: 1 model(s) report a context window that disagrees with gerbil's built-in table:
  claude-opus-4-8: provider says 500,000, table says 1,000,000 (last seen 2026-08-08)
  using the provider's number; refresh CONTEXT_WINDOWS in context_windows.py to silence this
```

The warning is advisory -- gerbil already prefers the provider's number. It
means the shipped table has gone stale, which is worth knowing because a model
gerbil has never queried live is still being sized against it. Refresh the
table from that site's `/api/data/pricing` endpoint (see
`src/gerbil/context_windows.py`), or delete the log to start the history over.

## Cost

gerbil prints a running token count and an estimated cost at the end of each
session, using a built-in per-model pricing table. Models missing from that
table simply report `N/A` rather than a wrong number. See
[Usage and cost](summarize.md) for the cross-session view.

## Big-small mode

A large model can drive the session while a smaller, cheaper one grinds through
the mechanical parts of individual proofs. See
[Big-small mode](big-small.md).
