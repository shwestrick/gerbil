"""Session views -- where a running session's output goes.

agent.py and cli.py keep building their display strings exactly as they always
have (render.py does the formatting, style() the coloring) and hand the
finished string to a SessionView; the view supplies only the *destination* and
the print conventions (leading blank line or not, stdout vs stderr). Two
implementations exist:

- PrintView (here): the classic scrolling stream -- byte-for-byte the output
  gerbil has always produced. It is the default everywhere (run_session
  constructs one when no view is passed), so existing tests, piped runs, and
  --plain behave exactly as before.
- RunnerView (here): PrintView plus live statistics, for the detached runner
  process behind a TUI run. Its stdout/stderr are already redirected to the
  run's display.ansi file, so the inherited prints ARE the stream the viewer
  (tui.ViewerApp) tails; on top it keeps a SessionStats and persists it to the
  run's stats.json after every stats-relevant event (see runs.py for the
  registry layout).

The full-screen app itself lives in tui.py -- the only module that imports
textual; this one never does.

This module also owns the pure state behind the TUI's left pane:

- SessionStats: everything the left pane shows (turns, token buckets, per-file
  +/- line counts, zoom state, wall-clock anchors), updated by plain methods
  that mirror the view events.
- patch_file_stats(): per-file added/removed line counts parsed from the wip
  patch text the agent loop already produces every tool-running turn -- the
  file list costs no extra container round-trips.
- render_stats(): SessionStats -> displayable text.

Everything here is plain functions over plain data, testable without a
terminal (tests/test_tui.py), and deliberately free of any textual import.
"""

import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from . import render, runs
from .pricing import estimate_cost
from .providers import Usage


class SessionView(Protocol):
    """What the agent loop and the CLI need from an output destination.

    Free-text methods (banner, notice, zoom_end, result_line) receive their
    text *already styled* -- the caller keeps composing render.style(...)
    exactly as it did when it printed directly, so PrintView reproduces
    today's bytes and TuiView feeds the same ANSI text to rich. The data-feed
    methods (turn_complete, wip_patch) carry raw values and are no-ops for
    PrintView; they exist so a live view can keep statistics without the
    agent loop knowing anything about statistics.
    """

    # Whether the view wants the wip patch text even when no snapshot file is
    # being written. False for PrintView so `wip_patch_path=None` keeps its
    # long-standing meaning of "compute nothing" (tests rely on that); True
    # for TuiView, whose file table is built from the patch text.
    wants_wip_patch: bool

    # -- structure ---------------------------------------------------------
    def session_begin(
        self, *, name: str, model: str, small_model: str | None,
        ralph: dict | None, resumed_from: str | None,
    ) -> None: ...
    def banner(self, text: str) -> None: ...
    def turn_header(
        self, label: str, *styles: str,
        max_context: int | None = None, usage=None,
    ) -> None: ...
    def assistant_delta(self, text: str) -> None: ...
    def tool_call(self, name: str, args: dict, read_file=None) -> None: ...
    def tool_result(
        self, name: str, content: str, raw_content: str, is_error: bool,
    ) -> None: ...
    def notice(
        self, text: str, *, newline_before: bool = True, stderr: bool = False,
    ) -> None: ...
    def turn_end(self) -> None: ...
    def zoom_begin(
        self, pos: str, small_model: str, max_context: int | None,
    ) -> None: ...
    def zoom_end(self, text: str) -> None: ...
    # -- data feeds (PrintView no-ops) ---------------------------------------
    def turn_complete(self, usage: Usage, *, zoom: bool) -> None: ...
    def wip_patch(self, patch_text: str) -> None: ...
    # -- results -------------------------------------------------------------
    def usage_summary(self, turns: int, usage: Usage, cost: float | None) -> None: ...
    def result_line(self, text: str) -> None: ...


class PrintView:
    """The classic scrolling stream: exactly the print()/stdout.write() calls
    the agent loop and CLI used to make inline, moved behind the view methods
    verbatim. Nothing here may drift from what those call sites historically
    printed -- tests/test_tui.py pins the conventions."""

    wants_wip_patch = False

    def session_begin(self, *, name, model, small_model, ralph, resumed_from):
        pass  # the classic stream has no per-session state

    def banner(self, text):
        print(text, flush=True)

    def turn_header(self, label, *styles, max_context=None, usage=None):
        print("\n" + render.turn_header(
            label, *styles, max_context=max_context, usage=usage,
        ), flush=True)

    def assistant_delta(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()

    def tool_call(self, name, args, read_file=None):
        print("\n" + render.format_tool_call(name, args, read_file), flush=True)

    def tool_result(self, name, content, raw_content, is_error):
        print(render.format_tool_result(name, content, raw_content, is_error),
              flush=True)

    def notice(self, text, *, newline_before=True, stderr=False):
        print(("\n" + text) if newline_before else text,
              file=sys.stderr if stderr else sys.stdout, flush=True)

    def turn_end(self):
        print(flush=True)

    def zoom_begin(self, pos, small_model, max_context):
        print("\n" + render.style(
            f"===== zoom in: {pos} ({small_model}) =====", "bold", "magenta",
        ), flush=True)

    def zoom_end(self, text):
        print("\n" + text, flush=True)

    def turn_complete(self, usage, *, zoom):
        pass

    def wip_patch(self, patch_text):
        pass

    def usage_summary(self, turns, usage, cost):
        render.print_usage(turns, usage, cost)

    def result_line(self, text):
        print(text, flush=True)


# ---------------------------------------------------------------------------
# Left-pane state: pure data + pure functions, shared with tests.
# ---------------------------------------------------------------------------


def patch_file_stats(patch_text: str) -> dict[str, tuple[int, int] | None]:
    """Per-file (added, removed) line counts from a `git format-patch --stdout`
    text, in first-appearance order. A None value marks a binary change.

    Follows the same line conventions as cli._patch_lean_delta, with one
    difference that matters here: the wip patch is `format-patch base..wip
    --stdout` and can hold SEVERAL commits when the agent commits its work
    incrementally. The `-- ` line that ends each commit (the mail-signature
    separator) therefore *pauses* counting rather than stopping it -- counting
    resumes at the next `diff --git`, and the in-between commit-message lines
    (which legitimately start with `-`, e.g. bullet points) stay uncounted.
    The first commit's message is equally safe: no file is current until its
    first `diff --git`.

    A file touched in several commits gets its counts summed, which reads as
    churn rather than net when later commits rework earlier lines -- accepted
    for a live display, and swappable for a `git diff --numstat <base>` (one
    extra container exec per turn) behind this same signature if it ever
    matters."""
    files: dict[str, tuple[int, int] | None] = {}
    current: str | None = None
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            # The b/ side is the post-image path, which is also the right name
            # for a rename (the rename from/to header lines carry no +/-).
            current = line.rsplit(" b/", 1)[-1]
            files.setdefault(current, (0, 0))
        elif line == "-- ":
            current = None
        elif current is None or files[current] is None:
            continue
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            # Mark binary; the base85 payload lines that may follow can start
            # with +/- and must not be counted (the None value shields them).
            files[current] = None
        elif line.startswith("+++") or line.startswith("---"):
            continue  # file headers, not content
        elif line.startswith("+"):
            added, removed = files[current]
            files[current] = (added + 1, removed)
        elif line.startswith("-"):
            added, removed = files[current]
            files[current] = (added, removed + 1)
    return files


@dataclass
class SessionStats:
    """Everything the live left pane displays. Mutated only by the on_*
    methods below (driven by view events on the UI thread) and read by
    render_stats(); no locking needed because both happen on one thread.

    Two lifetimes coexist: per-session fields reset on every on_session_begin
    (a ralph chain runs several sessions under one TUI), while the chain_*
    fields and chain_started accumulate across the whole run."""

    # Identity (per session; run_name spans the whole run and is what
    # `gerbil grab` takes -- shown in the pane so the user always sees it)
    run_name: str = ""
    model: str = ""
    small_model: str | None = None
    session_name: str = ""
    ralph_iteration: int | None = None
    ralph_total: int | None = None
    resumed_from: str | None = None

    # Wall clock (time.monotonic values; render_stats takes `now`)
    chain_started: float = 0.0
    session_started: float = 0.0

    # Per-session counters
    turns: int = 0
    zoom_turns: int = 0
    max_context: int | None = None
    zoom_max_context: int | None = None
    last_context_tokens: int = 0        # the outer conversation's latest turn
    last_zoom_context_tokens: int = 0   # the inner conversation's latest turn
    zoom_active: str | None = None      # "File.lean:42" while zoomed in

    # Token buckets. Unlike agent.py's `total` (which folds zoom usage in),
    # `outer` here is the big model's own spend only -- outer + inner is the
    # session total, and each bucket prices at its own model's rates.
    outer: Usage = field(default_factory=Usage)
    inner: Usage = field(default_factory=Usage)
    chain_outer: Usage = field(default_factory=Usage)
    chain_inner: Usage = field(default_factory=Usage)

    # path -> (added, removed), None = binary; first-appearance order.
    files: dict[str, tuple[int, int] | None] = field(default_factory=dict)
    # The finished ralph sessions' file stats, accumulated at each
    # session_begin (a sum of the per-session diffs, the same "sum of the
    # patches" metric `gerbil summarize` reports). The live chain view is
    # merge_file_stats(chain_files, files), computed at render time.
    chain_files: dict[str, tuple[int, int] | None] = field(default_factory=dict)

    # Set when the user asked to interrupt; render_stats shows a banner.
    interrupt_requested: bool = False

    # Set while the runner is SIGSTOPped (viewer-owned, like the fields below:
    # the viewer derives it from meta.json each poll; it never crosses the
    # stats wire).
    paused: bool = False

    # Set once the whole run (every ralph session) has ended and the TUI is
    # holding the screen for the user to read: "complete", "interrupted", or
    # "error". finished_at pins the clocks so elapsed stops counting the time
    # spent looking at the finished screen.
    finished: str | None = None
    finished_at: float | None = None

    def on_session_begin(self, *, name: str, model: str,
                         small_model: str | None, ralph: dict | None,
                         resumed_from: str | None, now: float) -> None:
        self.model = model
        self.small_model = small_model
        self.session_name = name
        self.ralph_iteration = ralph["iteration"] if ralph else None
        self.ralph_total = ralph["total"] if ralph else None
        self.resumed_from = resumed_from
        if not self.chain_started:
            self.chain_started = now
        self.session_started = now
        self.turns = 0
        self.zoom_turns = 0
        self.last_context_tokens = 0
        self.last_zoom_context_tokens = 0
        self.zoom_active = None
        self.outer = Usage()
        self.inner = Usage()
        # Fold the finished session's diff into the chain totals before the
        # new session starts from a clean slate.
        self.chain_files = merge_file_stats(self.chain_files, self.files)
        self.files = {}

    def on_turn_header(self, max_context: int | None) -> None:
        # The zoom sub-session's headers carry the SMALL model's window; only
        # adopt a window for the outer gauge while no zoom is in progress (the
        # zoom's own window arrives explicitly via on_zoom_begin).
        if self.zoom_active is None and max_context:
            self.max_context = max_context

    def on_turn_complete(self, usage: Usage, *, zoom: bool) -> None:
        for dst in ((self.inner, self.chain_inner) if zoom
                    else (self.outer, self.chain_outer)):
            dst.input_tokens += usage.input_tokens
            dst.output_tokens += usage.output_tokens
            dst.thinking_tokens += usage.thinking_tokens
            dst.cache_read_tokens += usage.cache_read_tokens
            dst.cache_write_tokens += usage.cache_write_tokens
        if zoom:
            self.zoom_turns += 1
            self.last_zoom_context_tokens = usage.context_tokens
        else:
            self.turns += 1
            self.last_context_tokens = usage.context_tokens

    def on_zoom_begin(self, pos: str, small_model: str,
                      max_context: int | None) -> None:
        self.zoom_active = pos
        self.small_model = small_model
        self.zoom_max_context = max_context
        self.last_zoom_context_tokens = 0

    def on_zoom_end(self) -> None:
        self.zoom_active = None

    def on_wip_patch(self, patch_text: str) -> None:
        self.files = patch_file_stats(patch_text)


def merge_file_stats(
    base: dict[str, tuple[int, int] | None],
    extra: dict[str, tuple[int, int] | None],
) -> dict[str, tuple[int, int] | None]:
    """Per-file sums of two diff-stat dicts (binary None absorbs). Used for
    the chain view: each session's numbers are exact, and the sum across
    sessions is the sum-of-patches metric summarize also reports."""
    merged = dict(base)
    for path, counts in extra.items():
        prev = merged.get(path)
        if counts is None or (path in merged and prev is None):
            merged[path] = None
        else:
            pa, pr = prev or (0, 0)
            merged[path] = (pa + counts[0], pr + counts[1])
    return merged


def _bucket_cost(model: str, small_model: str | None,
                 outer: Usage, inner: Usage) -> float | None:
    """Estimated cost of an (outer, inner) bucket pair -- the same split
    agent.py prices at session end, so the live figure converges on the final
    one. Any unpriceable side makes the whole answer None: a guessed number is
    worse than an honest N/A."""
    if not model:
        # Before the first session_begin there is no model yet; asking pricing
        # about "" would emit its unknown-model warning for nothing.
        return None
    outer_cost = estimate_cost(
        model, outer.input_tokens, outer.output_tokens,
        outer.cache_read_tokens, outer.cache_write_tokens,
    )
    if not _usage_any(inner):
        return outer_cost
    inner_cost = estimate_cost(
        small_model or "", inner.input_tokens, inner.output_tokens,
        inner.cache_read_tokens, inner.cache_write_tokens,
    )
    if outer_cost is None or inner_cost is None:
        return None
    return outer_cost + inner_cost


def live_cost(stats: SessionStats) -> float | None:
    """The running cost estimate for the current session."""
    return _bucket_cost(stats.model, stats.small_model, stats.outer, stats.inner)


def chain_cost(stats: SessionStats) -> float | None:
    """The running cost estimate across the whole ralph chain."""
    return _bucket_cost(
        stats.model, stats.small_model, stats.chain_outer, stats.chain_inner,
    )


def _usage_any(u: Usage) -> bool:
    return bool(
        u.input_tokens or u.output_tokens or u.thinking_tokens
        or u.cache_read_tokens or u.cache_write_tokens
    )


def _hms(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _stat_row(label: str, value: str) -> str:
    return f"{render.style(label, 'gray')} {value}"


def render_stats(stats: SessionStats, width: int, now: float | None = None) -> str:
    """The left pane as one newline-joined string. Pure over (stats, width,
    now) -- `now` defaults to the real clock and is a parameter so tests can
    pin it. Colors come from render.style, so the output is plain text
    wherever style() is disabled (piped tests, NO_COLOR)."""
    now = time.monotonic() if now is None else now
    if stats.finished_at is not None:
        now = stats.finished_at  # freeze the clocks on the finished screen
    sep = render.GLYPHS["sep"]
    rule = render.GLYPHS["rule"]
    lines: list[str] = []

    title = f"gerbil {sep} {stats.run_name or stats.model}"
    lines.append(render.style(title, "bold"))
    if stats.run_name:
        lines.append(_stat_row("model", stats.model))
    if stats.small_model:
        lines.append(_stat_row("small:", stats.small_model))

    name = stats.session_name
    if stats.resumed_from:
        name += " (resumed)"
    lines.append(_stat_row("session", name))
    if stats.ralph_total:
        lines.append(_stat_row(
            "ralph", f"{stats.ralph_iteration}/{stats.ralph_total}"))

    elapsed = _hms(now - stats.session_started)
    if stats.ralph_total and stats.ralph_total > 1:
        elapsed += f"  (chain {_hms(now - stats.chain_started)})"
    lines.append(_stat_row("elapsed", elapsed))

    turns = str(stats.turns)
    if stats.zoom_turns:
        turns += f" (+{stats.zoom_turns} zoom)"
    lines.append(_stat_row("turns", turns))

    used = stats.last_context_tokens
    if stats.max_context:
        pct = used / stats.max_context * 100
        color = "red" if pct >= 80 else "yellow" if pct >= 50 else None
        gauge = f"{pct:.1f}%  {used:,} / {stats.max_context:,}"
        lines.append(_stat_row(
            "context", render.style(gauge, color) if color else gauge))
    else:
        lines.append(_stat_row("context", f"{used:,} tokens (window unknown)"))

    tin = sum(
        u.input_tokens + u.cache_read_tokens + u.cache_write_tokens
        for u in (stats.outer, stats.inner)
    )
    tout = stats.outer.output_tokens + stats.inner.output_tokens
    lines.append(_stat_row("tokens", f"in {tin:,} {sep} out {tout:,}"))

    cost = live_cost(stats)
    cost_str = "N/A" if cost is None else f"~${cost:.4f}"
    ccost = chain_cost(stats)
    if stats.ralph_total and stats.ralph_total > 1 and ccost is not None:
        cost_str += f"  (chain ~${ccost:.4f})"
    lines.append(_stat_row("cost", cost_str))

    if stats.zoom_active:
        zoom = stats.zoom_active
        if stats.zoom_max_context:
            zpct = stats.last_zoom_context_tokens / stats.zoom_max_context * 100
            zoom += f" {sep} ctx {zpct:.1f}%"
        lines.append(render.style(f"zoom: {zoom}", "magenta"))

    if stats.finished is not None:
        color = {"complete": "green", "interrupted": "yellow"}.get(
            stats.finished, "red")
        lines.append("")
        lines.append(render.style(f"session {stats.finished}", "bold", color))
        lines.append(render.style("press q or enter to exit", "gray"))
    elif stats.interrupt_requested:
        lines.append("")
        lines.append(render.style(
            "interrupting: finishing the current operation;\n"
            "press again to detach", "bold", "yellow"))
    elif stats.paused:
        lines.append("")
        lines.append(render.style("paused: press c to continue", "bold", "blue"))
        lines.append(render.style(
            "(the sandbox stays alive; d detaches)", "gray"))

    return "\n".join(lines)


def render_file_tree(
    files: dict[str, tuple[int, int] | None], width: int
) -> list[str]:
    """A `tree`-style listing of a diff-stat dict, one line per directory or
    file, +/- figures right-aligned. Chains of single-child directories are
    compressed onto one line (A/B/C/), so depth costs width only where the
    tree actually branches. Connector glyphs come from render.GLYPHS and
    degrade to ASCII exactly like every other decoration."""
    tee, corner = render.GLYPHS["tee"], render.GLYPHS["corner"]
    pipe, blank = render.GLYPHS["pipe"], render.GLYPHS["blank"]

    # path components -> nested dict; a leaf holds its counts under None key.
    tree: dict = {}
    for path, counts in files.items():
        node = tree
        parts = path.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = {None: counts}

    def row(prefix: str, name: str, counts=None, leaf=False) -> str:
        if not leaf:
            return prefix + name
        if counts is None:
            figures = "bin"
        else:
            figures = (render.style(f"+{counts[0]}", "green") + " "
                       + render.style(f"-{counts[1]}", "red"))
        fig_width = render._visible_len(figures)
        label = prefix + name
        avail = max(2, width - fig_width - 1)
        if len(label) > avail:
            label = label[:avail - 1] + "…"
        pad = max(1, width - len(label) - fig_width)
        return f"{label}{' ' * pad}{figures}"

    lines: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        dirs = sorted(k for k, v in node.items() if k and None not in v)
        leaves = sorted(k for k, v in node.items() if k and None in v)
        entries = dirs + leaves
        for i, name in enumerate(entries):
            last = i == len(entries) - 1
            connector = corner if last else tee
            child = node[name]
            if None in child:
                lines.append(row(prefix + connector, name,
                                 child[None], leaf=True))
            else:
                # Compress single-child directory chains: A/B/C/ on one line.
                label = name
                while len(child) == 1 and None not in next(iter(child.values())):
                    (only,) = child
                    label += "/" + only
                    child = child[only]
                lines.append(row(prefix + connector, label + "/"))
                walk(child, prefix + (blank if last else pipe))

    walk(tree, "")
    return lines


def file_summary(stats: SessionStats, width: int) -> str:
    """The scrollable file pane: this session's diff as a tree, and -- for a
    ralph chain past its first session -- the whole chain's summed diff
    below it. Pure over (stats, width), like render_stats."""
    rule = render.GLYPHS["rule"]

    def section(title: str, files: dict) -> list[str]:
        lines = [render.style(f"{rule * 2} {title} {rule * 2}", "gray")]
        if not files:
            lines.append(render.style("(none yet)", "gray"))
            return lines
        lines += render_file_tree(files, width)
        counted = [c for c in files.values() if c is not None]
        if len(files) > 1:
            total = (render.style(f"+{sum(a for a, _ in counted)}", "green")
                     + " "
                     + render.style(f"-{sum(r for _, r in counted)}", "red"))
            label = f"total ({len(files)} files)"
            pad = max(1, width - len(label) - render._visible_len(total))
            lines.append(render.style(f"{label}{' ' * pad}", "gray") + total)
        return lines

    chain = stats.ralph_total is not None and stats.ralph_total > 1
    lines = section("files (this session)" if chain else "files", stats.files)
    if chain:
        lines.append("")
        lines += section(
            "files (chain)", merge_file_stats(stats.chain_files, stats.files))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The wire format: SessionStats across a process boundary.
#
# The runner process owns the stats; the viewer re-renders them. Everything
# JSON-native crosses as-is; the two things that don't are converted here:
# time.monotonic() anchors (meaningless in another process -- shipped as
# elapsed seconds, re-anchored against the reader's own clock so its 1s tick
# keeps counting between writes) and the files dict's tuple/None values.
# The re-anchoring contract: the reader anchors ONCE per fresh doc (tui._poll
# gates on written_at) and its own clock carries the time in between --
# re-anchoring the same stale doc again would pin the clocks to the doc's
# values and make elapsed advance only in event-sized jumps. `age` is how
# stale the doc already is at anchoring time (reader wall clock minus the
# doc's written_at), so a viewer attaching mid-turn shows the run's true
# elapsed rather than the last event's.
# `finished`/`finished_at`/`interrupt_requested`/`paused` are deliberately NOT
# carried: meta.json's status is the authority for how a run ended (and for
# pause), and the interrupt banner belongs to the viewer that pressed the key.
# ---------------------------------------------------------------------------


def stats_to_wire(stats: SessionStats, now: float | None = None) -> dict:
    now = time.monotonic() if now is None else now
    return {
        "run_name": stats.run_name,
        "model": stats.model,
        "small_model": stats.small_model,
        "session_name": stats.session_name,
        "ralph_iteration": stats.ralph_iteration,
        "ralph_total": stats.ralph_total,
        "resumed_from": stats.resumed_from,
        "session_elapsed": max(0.0, now - stats.session_started),
        "chain_elapsed": max(0.0, now - stats.chain_started),
        "turns": stats.turns,
        "zoom_turns": stats.zoom_turns,
        "max_context": stats.max_context,
        "zoom_max_context": stats.zoom_max_context,
        "last_context_tokens": stats.last_context_tokens,
        "last_zoom_context_tokens": stats.last_zoom_context_tokens,
        "zoom_active": stats.zoom_active,
        "outer": asdict(stats.outer),
        "inner": asdict(stats.inner),
        "chain_outer": asdict(stats.chain_outer),
        "chain_inner": asdict(stats.chain_inner),
        "files": {
            path: None if counts is None else list(counts)
            for path, counts in stats.files.items()
        },
        "chain_files": {
            path: None if counts is None else list(counts)
            for path, counts in stats.chain_files.items()
        },
    }


def stats_from_wire(
    doc: dict, now: float | None = None, *, age: float = 0.0
) -> SessionStats:
    """Tolerant inverse of stats_to_wire: every field falls back to its
    default, so a document written by a different gerbil version (or torn in
    some way the atomic replace didn't catch) degrades the display rather
    than crashing the viewer."""
    now = time.monotonic() if now is None else now
    s = SessionStats()

    def _usage(key: str) -> Usage:
        val = doc.get(key)
        if isinstance(val, dict):
            try:
                return Usage(**val)
            except TypeError:
                pass
        return Usage()

    def _opt_int(key: str) -> int | None:
        val = doc.get(key)
        return int(val) if isinstance(val, (int, float)) and val else None

    s.run_name = str(doc.get("run_name") or "")
    s.model = str(doc.get("model") or "")
    s.small_model = doc.get("small_model") or None
    s.session_name = str(doc.get("session_name") or "")
    s.ralph_iteration = _opt_int("ralph_iteration")
    s.ralph_total = _opt_int("ralph_total")
    s.resumed_from = doc.get("resumed_from") or None
    s.session_started = now - float(doc.get("session_elapsed") or 0.0) - age
    s.chain_started = now - float(doc.get("chain_elapsed") or 0.0) - age
    s.turns = int(doc.get("turns") or 0)
    s.zoom_turns = int(doc.get("zoom_turns") or 0)
    s.max_context = _opt_int("max_context")
    s.zoom_max_context = _opt_int("zoom_max_context")
    s.last_context_tokens = int(doc.get("last_context_tokens") or 0)
    s.last_zoom_context_tokens = int(doc.get("last_zoom_context_tokens") or 0)
    s.zoom_active = doc.get("zoom_active") or None
    s.outer = _usage("outer")
    s.inner = _usage("inner")
    s.chain_outer = _usage("chain_outer")
    s.chain_inner = _usage("chain_inner")
    for key, dst in (("files", s.files), ("chain_files", s.chain_files)):
        files_doc = doc.get(key)
        if isinstance(files_doc, dict):
            for path, counts in files_doc.items():
                try:
                    dst[str(path)] = (
                        None if counts is None
                        else (int(counts[0]), int(counts[1]))
                    )
                except (TypeError, IndexError, ValueError):
                    continue  # one malformed entry costs one row, not the pane
    return s


class RunnerView(PrintView):
    """The detached runner's view: the classic stream plus live statistics.

    Runs with stdout/stderr redirected to the run's display.ansi (and
    GERBIL_FORCE_STYLE keeping render.style live), so the inherited PrintView
    prints are exactly what the viewer tails. On top, it maintains a
    SessionStats and rewrites the run's stats.json after every stats-relevant
    event -- a handful of writes per turn, never per streaming delta. All
    persistence is best-effort: registry I/O must never take a session down.

    The one display divergence from PrintView is turn_header: the full-width
    rule render.turn_header draws (at the 80-column non-tty fallback) is wrong
    inside a viewer pane, so the runner writes the compact divider the
    in-process TUI used to draw, and leaves the context gauge to the stats
    pane."""

    wants_wip_patch = True  # the file table is parsed from the wip patch

    def __init__(self, run_dir: Path):
        self._dir = run_dir
        self.stats = SessionStats(run_name=run_dir.name)
        self._tail: list[str] = []

    def _save(self) -> None:
        try:
            runs.write_stats_doc(self._dir, {
                "v": 1,
                "written_at": time.time(),
                "stats": stats_to_wire(self.stats),
                "tail": list(self._tail),
            })
        except Exception:
            pass  # write_stats_doc already swallows OSError; belt and braces

    def session_begin(self, *, name, model, small_model, ralph, resumed_from):
        self.stats.on_session_begin(
            name=name, model=model, small_model=small_model, ralph=ralph,
            resumed_from=resumed_from, now=time.monotonic(),
        )
        self._save()

    def turn_header(self, label, *styles, max_context=None, usage=None):
        stamp = time.strftime("%H:%M:%S")
        rule, sep = render.GLYPHS["rule"], render.GLYPHS["sep"]
        print("\n" + render.style(f"{rule * 2} {label} {sep} {stamp}", *styles),
              flush=True)
        self.stats.on_turn_header(max_context)
        self._save()

    def turn_complete(self, usage, *, zoom):
        self.stats.on_turn_complete(usage, zoom=zoom)
        self._save()

    def zoom_begin(self, pos, small_model, max_context):
        super().zoom_begin(pos, small_model, max_context)
        self.stats.on_zoom_begin(pos, small_model, max_context)
        self._save()

    def zoom_end(self, text):
        super().zoom_end(text)
        self.stats.on_zoom_end()
        self._save()

    def wip_patch(self, patch_text):
        self.stats.on_wip_patch(patch_text)
        self._save()

    def usage_summary(self, turns, usage, cost):
        super().usage_summary(turns, usage, cost)
        self._tail.append(render.usage_line(turns, usage, cost))
        self._save()

    def result_line(self, text):
        super().result_line(text)
        self._tail.append(text)
        self._save()
