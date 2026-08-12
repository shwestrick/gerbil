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
- TuiView (tui.py): the full-screen live view. It lives in its own module so
  this one never imports textual -- the classic path must keep working even if
  the textual dependency were missing or broken.

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
from dataclasses import dataclass, field
from typing import Callable, Protocol

from . import render
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


def run_under_view(use_tui: bool, body: Callable[[SessionView], None],
                   theme: str | None = None) -> None:
    """Run `body` (the whole session loop) under the chosen view.

    The plain path is deliberately nothing at all: construct a PrintView and
    call straight through on the caller's thread, so --plain and non-TTY runs
    have exactly the control flow they had before the TUI existed. The TUI
    import is deferred to here so the classic path never touches textual.

    `theme` is the TUI's color scheme ("light" or "dark", from the project
    config; None = the default dark). The plain view has no theme -- its
    colors are whatever the user's terminal shows for ANSI."""
    if not use_tui:
        body(PrintView())
        return
    from .tui import launch_tui
    launch_tui(body, theme=theme)


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

    # Identity (per session)
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

    # Set when the user asked to interrupt; render_stats shows a banner.
    interrupt_requested: bool = False

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


def _elide_path(path: str, width: int) -> str:
    """Left-elide a path to fit `width` columns: the filename end is the
    distinctive part."""
    if len(path) <= width or width < 2:
        return path
    return "…" + path[-(width - 1):] if render.UNICODE else \
        "..." + path[-(width - 3):]


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

    title = f"gerbil {sep} {stats.model}"
    lines.append(render.style(title, "bold"))
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

    lines.append("")
    header = f"{rule * 2} files {rule * 2}"
    lines.append(render.style(header, "gray"))
    if not stats.files:
        lines.append(render.style("(none yet)", "gray"))
    else:
        # Biggest churn first; binary entries (no counts) last.
        def churn(item):
            counts = item[1]
            return -1 if counts is None else counts[0] + counts[1]
        ordered = sorted(stats.files.items(), key=churn, reverse=True)
        total_added = total_removed = 0
        for path, counts in ordered:
            if counts is None:
                figures = "bin"
            else:
                figures = (render.style(f"+{counts[0]}", "green") + " "
                           + render.style(f"-{counts[1]}", "red"))
                total_added += counts[0]
                total_removed += counts[1]
            fig_width = render._visible_len(figures)
            path_width = max(2, width - fig_width - 2)
            shown = _elide_path(path, path_width)
            pad = max(1, width - len(shown) - fig_width)
            lines.append(f"{shown}{' ' * pad}{figures}")
        if len(ordered) > 1:
            total = (render.style(f"+{total_added}", "green") + " "
                     + render.style(f"-{total_removed}", "red"))
            label = f"total ({len(ordered)} files)"
            pad = max(1, width - len(label) - render._visible_len(total))
            lines.append(render.style(f"{label}{' ' * pad}", "gray") + total)

    return "\n".join(lines)
