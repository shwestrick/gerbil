# Usage and cost

`gerbil summarize` scans a project's `.gerbil/*.jsonl` session logs and reports
total token usage, an estimated cost, and breakdowns by session status, tool
call, and model:

```console
$ gerbil summarize
gerbil summary -- 28 session(s) in /path/to/project/.gerbil

Tokens
  input:     141,094,502
  output:        367,974
  thinking:      201,830  (of output)
  total:     141,462,476

Estimated cost
  ~$180.0479

Sessions
  ...
```

## Reading the numbers

**`thinking`** is the reasoning-token portion of `output`, broken out for the
models that report it (Gemini's "thoughts", OpenAI's reasoning tokens, and so
on). Thinking tokens bill at the output rate and are *already included* in
`output` and in the cost, so they are shown as a sub-total, not added on top.
For models that don't expose the breakdown, thinking is silently counted within
`output`.

**Cache reads and writes** are listed separately when a provider reports them,
annotated with their rate relative to input tokens (0.1x for reads, 1.25x for
writes). Long agent sessions replay a growing conversation every turn, so cache
hits dominate the input count and the cost is far lower than the raw input
number suggests.

**Estimated cost** comes from gerbil's built-in per-model pricing table. A
model missing from that table contributes tokens but no cost, and the report
says how many sessions were excluded rather than quietly under-reporting. If no
session has known pricing, the cost is `N/A`.

In a [big-small](big-small.md) session, the inner sub-session's events are
tagged in the log, so each model's tokens are priced at its own rates.

## The per-session table

Below the totals is one row per session: its cost, the `.lean` lines added and
removed (taken from the commit that log was folded into, or from its
uncommitted `.patch`), and a running total of Lean code in the project
(excluding `.lake`). It is a rough answer to "what did all this actually buy
me".

## Where the logs come from

Session logs reach the project's `.gerbil/` by default, carried in the
committed patch. Runs made with `gerbil run --omit-session-log` keep them only
in `~/.gerbil/sessions/`; if the project has no logs, `summarize` points you
there.
