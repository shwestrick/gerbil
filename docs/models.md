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

## Cost

gerbil prints a running token count and an estimated cost at the end of each
session, using a built-in per-model pricing table. Models missing from that
table simply report `N/A` rather than a wrong number. See
[Usage and cost](summarize.md) for the cross-session view.

## Big-small mode

A large model can drive the session while a smaller, cheaper one grinds through
the mechanical parts of individual proofs. See
[Big-small mode](big-small.md).
