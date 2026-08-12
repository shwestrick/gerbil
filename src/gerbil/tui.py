"""The full-screen live session view: a textual app with a stats pane on the
left (files edited with per-file +/- lines, turns, context, tokens, cost,
wall-clock) and the scrolling turn output on the right.

Division of labor: everything *stateful and pure* (SessionStats, the patch
parser, render_stats) lives in view.py, testable without a terminal. This
module is only the glue between three threads of control:

- The MAIN thread runs the textual App (it owns the terminal; textual requires
  it for signal handling).
- A WORKER thread runs the whole session loop -- the same `body(view)` closure
  cmd_run/cmd_resume would call inline under a PrintView, with every `with`
  block (sandbox, MCP) intact. It is a plain daemon Thread, deliberately NOT a
  textual worker: textual awaits worker cancellation on shutdown, and a thread
  blocked in a `docker exec` cannot be cancelled -- the app must be able to
  exit while the worker is still unwinding.
- TuiView is the bridge: its methods run on the worker thread, do any heavy
  lifting there (patch parsing, line buffering), and marshal one small
  _apply() call to the UI thread via app.call_from_thread. Stats are mutated
  and read only on the UI thread, so no locks are needed.

Interruption is cooperative. The first Ctrl-C (or q) sets a threading.Event;
the next view event raised on the worker thread becomes a KeyboardInterrupt
there, which unwinds the sandbox `with` blocks normally (the container is
stopped) and is then re-raised on the main thread -- after the terminal is
restored -- into cmd_run/cmd_resume's existing `except -> _abort` path. A
second Ctrl-C detaches the UI immediately without waiting for the worker.
There is no way to interrupt a tool call mid-flight (the same is true of the
classic view -- a KeyboardInterrupt lands between bytecode instructions, and
the worker is blocked inside a docker exec); its own `timeout N bash -c`
bounds the wait.

render.py note: everything the right pane shows is the same ANSI-styled text
the classic view prints, converted with rich's Text.from_ansi. That works
because the TUI is only ever selected when stdout is a real TTY (cli._use_tui),
so render.ENABLED was already computed True at import -- unless NO_COLOR is
set, in which case the strings are plain and the TUI is simply uncolored,
which is what NO_COLOR asks for.
"""

import sys
import threading
import time

from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal
from textual.events import Print
from textual.widgets import Footer, RichLog, Static

from . import render
from .view import SessionStats, patch_file_stats, render_stats

# Right-pane scrollback bound. Old lines beyond it fall out of the pane (the
# .jsonl session log remains the complete record); well above any session's
# useful scrollback while keeping memory flat on very long ralph chains.
LOG_MAX_LINES = 20_000

# The stats pane's fixed width (columns), including 1 column of padding each
# side. Wide enough for a 5-digit +/- file table row and the token figures.
STATS_PANE_WIDTH = 44


class TuiView:
    """SessionView implementation that feeds the textual app.

    Public methods run on the WORKER thread: they check for a pending
    interrupt (raising KeyboardInterrupt there -- the one place the
    cooperative quit lands), prepare display text, and marshal a single
    _apply() onto the UI thread. _apply and everything it touches (stats, the
    widgets) run only on the UI thread."""

    wants_wip_patch = True

    def __init__(self):
        self._app: GerbilApp | None = None
        self._interrupt = threading.Event()
        self.stats = SessionStats()
        self._pending = ""  # partial assistant line; worker thread only
        # Plain reprint of the session:/patch:/usage lines after the alt
        # screen closes, so the outcome survives in normal scrollback.
        # Appended on the worker thread, read by the main thread after join.
        self.tail_lines: list[str] = []

    def bind(self, app: "GerbilApp") -> None:
        self._app = app

    # -- interruption (UI thread sets, worker thread trips) ------------------

    @property
    def interrupt_requested(self) -> bool:
        return self._interrupt.is_set()

    def request_interrupt(self) -> None:
        self._interrupt.set()
        self.stats.interrupt_requested = True

    # -- worker-side plumbing -------------------------------------------------

    def _dispatch(self, mutate=None, log_text: str | None = None) -> None:
        """Marshal one event to the UI thread; the cooperative-interrupt
        checkpoint. After the app has exited (detach), call_from_thread fails;
        that is swallowed -- display is best-effort -- and the set interrupt
        event ends the worker at this same checkpoint on the next call."""
        if self._interrupt.is_set():
            raise KeyboardInterrupt
        try:
            self._app.call_from_thread(self._app.apply_event, mutate, log_text)
        except Exception:
            pass

    def _flush_pending(self) -> None:
        """Send any buffered partial assistant line to the log."""
        if self._pending:
            text, self._pending = self._pending, ""
            self._dispatch(None, text)

    # -- SessionView methods (worker thread) ----------------------------------

    def session_begin(self, *, name, model, small_model, ralph, resumed_from):
        self._flush_pending()
        now = time.monotonic()
        self._dispatch(lambda: self.stats.on_session_begin(
            name=name, model=model, small_model=small_model, ralph=ralph,
            resumed_from=resumed_from, now=now,
        ))

    def banner(self, text):
        self._dispatch(None, text)

    def turn_header(self, label, *styles, max_context=None, usage=None):
        # A compact divider instead of render.turn_header: no full-width rule
        # (wrong inside a pane), and no [context: ...] suffix -- the left pane
        # owns the gauge. The styles the caller chose (dark_red outer, magenta
        # zoom) still color it, so the two loops stay distinguishable.
        self._flush_pending()
        stamp = time.strftime("%H:%M:%S")
        rule, sep = render.GLYPHS["rule"], render.GLYPHS["sep"]
        header = render.style(f"{rule * 2} {label} {sep} {stamp}", *styles)
        self._dispatch(lambda: self.stats.on_turn_header(max_context), header)

    def assistant_delta(self, text):
        # Buffer to whole lines: one UI marshal per line, not per stream chunk.
        # The partial tail is held here (worker thread) until a newline, the
        # next tool call, or the end of the turn flushes it.
        if self._interrupt.is_set():
            raise KeyboardInterrupt
        self._pending += text
        if "\n" in self._pending:
            done, self._pending = self._pending.rsplit("\n", 1)
            self._dispatch(None, done)

    def tool_call(self, name, args, read_file=None):
        self._flush_pending()
        self._dispatch(None, render.format_tool_call(name, args, read_file))

    def tool_result(self, name, content, raw_content, is_error):
        self._dispatch(
            None, render.format_tool_result(name, content, raw_content, is_error)
        )

    def notice(self, text, *, newline_before=True, stderr=False):
        # newline_before is a print-stream spacing convention; the log pane is
        # already line-oriented, so spacer-only notices are dropped (part of
        # the "more compact" brief) and stderr-ness is meaningless here.
        self._flush_pending()
        if text.strip():
            self._dispatch(None, text)

    def turn_end(self):
        self._flush_pending()  # no blank line: the pane stays compact

    def zoom_begin(self, pos, small_model, max_context):
        self._flush_pending()
        rule = render.GLYPHS["rule"]
        header = render.style(
            f"{rule * 2} zoom in: {pos} ({small_model})", "bold", "magenta",
        )
        self._dispatch(
            lambda: self.stats.on_zoom_begin(pos, small_model, max_context),
            header,
        )

    def zoom_end(self, text):
        self._flush_pending()
        self._dispatch(self.stats.on_zoom_end, text)

    def turn_complete(self, usage, *, zoom):
        self._dispatch(lambda: self.stats.on_turn_complete(usage, zoom=zoom))

    def wip_patch(self, patch_text):
        # Parse on the worker thread (the patch can be large); assign on the UI
        # thread so stats never see a half-built dict.
        files = patch_file_stats(patch_text)
        def assign():
            self.stats.files = files
        self._dispatch(assign)

    def usage_summary(self, turns, usage, cost):
        self._flush_pending()
        line = render.usage_line(turns, usage, cost)
        self.tail_lines.append(line)
        self._dispatch(None, line)

    def result_line(self, text):
        self._flush_pending()
        self.tail_lines.append(text)
        self._dispatch(None, text)


class GerbilApp(App):
    """Two panes and a footer; all real state lives in view.stats."""

    TITLE = "gerbil"
    CSS = f"""
    Horizontal {{ height: 1fr; }}
    #stats {{
        width: {STATS_PANE_WIDTH};
        padding: 0 1;
        border-right: solid $accent;
    }}
    #log {{ width: 1fr; }}
    """
    BINDINGS = [
        Binding("ctrl+c", "interrupt", "interrupt", priority=True),
        Binding("q", "interrupt", "interrupt"),
        Binding("end", "follow", "follow tail"),
    ]

    def __init__(self, view: TuiView, worker: threading.Thread):
        super().__init__()
        self._view = view
        self.worker_thread = worker

    def compose(self):
        with Horizontal():
            yield Static(id="stats")
            yield RichLog(id="log", wrap=True, max_lines=LOG_MAX_LINES,
                          auto_scroll=True)
        yield Footer()

    def on_mount(self):
        self.set_interval(1.0, self.refresh_stats)  # the elapsed clock tick
        # Stray print()s from inside the bracket (pricing.py's unknown-pricing
        # warning, providers.py's Portkey fallback note) land in the log
        # instead of corrupting the alt screen.
        self.begin_capture_print(self)
        self.refresh_stats()
        self.worker_thread.start()

    def on_print(self, event: Print) -> None:
        text = event.text.rstrip("\n")
        if text:
            self._write_log(text)

    # -- events marshaled from the worker (UI thread) -------------------------

    def apply_event(self, mutate, log_text) -> None:
        if mutate is not None:
            mutate()
        if log_text is not None:
            self._write_log(log_text)
        self.refresh_stats()

    def _write_log(self, text: str) -> None:
        log = self.query_one("#log", RichLog)
        # Follow the tail only while the user is already at it: a reader who
        # scrolled back keeps their place, and returning to the bottom (or the
        # `end` binding) re-engages the auto-follow.
        log.write(Text.from_ansi(text), scroll_end=log.is_vertical_scroll_end)

    def refresh_stats(self) -> None:
        pane = self.query_one("#stats", Static)
        width = pane.content_size.width or (STATS_PANE_WIDTH - 2)
        pane.update(Text.from_ansi(render_stats(self._view.stats, width)))

    # -- user actions ----------------------------------------------------------

    def action_interrupt(self) -> None:
        if self._view.interrupt_requested:
            self.exit()  # second press: detach now, let the worker unwind alone
            return
        self._view.request_interrupt()
        self.refresh_stats()

    def action_follow(self) -> None:
        self.query_one("#log", RichLog).scroll_end(animate=False)


def launch_tui(body) -> None:
    """Run `body(view)` -- the whole session loop -- under the full-screen app.

    The worker's exception (a crash, or the KeyboardInterrupt our cooperative
    interrupt raises) is captured and re-raised HERE, on the main thread, only
    after the terminal is restored -- it then flows into cmd_run/cmd_resume's
    existing `except -> _abort` path, whose stderr message prints legibly onto
    the normal screen. The finally block also covers main()'s SIGTERM/SIGHUP
    -> SystemExit handlers firing inside app.run(): the worker is asked to
    stop and joined (giving the sandbox `with` blocks their normal teardown)
    before the exit proceeds."""
    view = TuiView()
    outcome: dict[str, BaseException] = {}

    def worker():
        try:
            body(view)
        except BaseException as exc:
            outcome["exc"] = exc
        finally:
            try:
                app.call_from_thread(app.exit)
            except Exception:
                pass  # app already gone (detached before the worker finished)

    app = GerbilApp(view, threading.Thread(
        target=worker, daemon=True, name="gerbil-session",
    ))
    view.bind(app)
    try:
        app.run()
    finally:
        view.request_interrupt()
        if app.worker_thread.is_alive():
            # Detached (or torn down by a signal) while the session was still
            # unwinding. Wait for it: the sandbox context managers are on that
            # thread. A real Ctrl-C works again now that the terminal is
            # restored -- it abandons the daemon thread, at worst leaving the
            # container running (the same exposure as kill -9 has always had).
            print(
                "waiting for the current operation to finish "
                "(Ctrl-C to force-quit; that may leave the sandbox container "
                "running)", file=sys.stderr,
            )
            app.worker_thread.join()
        # The session:/patch:/usage lines, replayed onto the normal screen so
        # the run's outcome survives in scrollback after the alt screen closed.
        for line in view.tail_lines:
            print(line)
    if "exc" in outcome:
        raise outcome["exc"]
