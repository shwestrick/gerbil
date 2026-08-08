# Big-small mode

Proof work splits unevenly. Deciding *how* to attack a theorem -- what to
generalize, which lemma to factor out, where the induction should go -- rewards
the strongest model you can afford. Grinding out the resulting `sorry` with
`simp` variations and `omega` mostly does not.

Big-small mode lets you pay for each separately:

```console
$ gerbil run --model claude-opus-4-8 --small-model claude-haiku-4-5-20251001 \
             --prompt prompt.md
```

`--model` is the **big** model; it drives the session normally and gets one
extra tool:

```
zoom_in(prompt, file, line[, column])
```

Calling it sets the big conversation aside and starts an inner sub-session of
the small model, aimed at that one `sorry`. The sub-session gets the full
toolset plus `zoom_out(summary)`, and no `zoom_in` of its own -- delegation is
at most one level deep. Its initial prompt is whatever the big model supplied,
plus one appended "YOUR TASK" line naming the sorry and the turn budget; its
system prompt is the plain gerbil one.

When the small model calls `zoom_out`, its summary comes back to the big model
as the result of the `zoom_in` call. The two share a working tree, so the small
model's edits are simply there afterwards.

The big model has to write a self-contained prompt: the sub-session cannot ask
it questions, and the summary is the only thing that flows back besides the
file changes themselves.

## Turn budget

`--zoom-max-turns N` caps each sub-session (default 100). Hitting the cap does
not silently drop the work: the small model gets one final forced turn with
`zoom_out` as its only available tool, so the big model always receives a real
report of where things stand.

## Logging and accounting

Inner events go into the same `.jsonl` log, tagged `"zoom": true`. That tag is
what lets:

- [`gerbil resume`](resume.md) rebuild the two conversations separately -- a
  crash that happens mid-zoom resumes *inside* the sub-session;
- [`gerbil summarize`](summarize.md) price each model's tokens at its own
  rates, so the report reflects what the split actually saved you.

## Resuming

Nothing extra is required. `gerbil resume` reads the small model, the zoom turn
cap, and everything else from the log. `--zoom-max-turns` on `gerbil resume`
overrides the recorded cap if you want to give the sub-sessions more room the
second time around.
