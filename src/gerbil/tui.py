"""The full-screen session viewer: a textual app attached to a detached
background run (see runs.py for the architecture and registry layout).

This process owns nothing but the screen. The session runs in the *runner*
child cli._spawn_and_attach started; the viewer tails the run's display.ansi
into the right pane, re-renders the runner's stats.json into the left pane,
and can be closed (detach: `d`) and reopened (`gerbil grab NAME`) freely --
which is the whole point. There is no worker thread and no shared memory: two
processes, three files, and a 4 Hz poll.

Everything the right pane shows is the ANSI-styled classic stream the runner
prints (GERBIL_FORCE_STYLE keeps render.style live despite its redirected
stdout), converted with rich's Text.from_ansi. Everything the left pane shows
is view.SessionStats, reconstructed each poll by view.stats_from_wire -- the
elapsed clocks re-anchor to this process's monotonic clock, so the 1s tick
keeps counting between the runner's writes.

Key semantics (user-facing):
- Ctrl-C / q: interrupt the run -- a real SIGINT to the runner, which is
  exactly a --plain Ctrl-C over there (the sandbox unwinds, _abort writes the
  "interrupted:" + resume lines into the display stream). A second press
  closes the viewer while the runner unwinds.
- d: detach -- the viewer exits, the run continues in the background.
- p / c: pause / continue. Pause SIGSTOPs the runner where it stands -- the
  sandbox container stays alive -- and c (from this or any later viewer;
  paused runs detach and grab like running ones) SIGCONTs the same process,
  which picks up exactly where it left off. Deliberately NOT called resume:
  `gerbil resume` is the unrelated feature that rebuilds a crashed session
  in a fresh sandbox.
- When the run ends (complete, interrupted, error, or the runner died), the
  viewer holds the finished screen until q/enter/Ctrl-C confirms; only a
  confirmed exit removes the run's registry entry and reprints the
  session:/patch:/usage tail onto the normal terminal.

The pure pieces (tailing, liveness classification, the wire format) live in
runs.py and view.py and are tested there; this module is the thinnest possible
textual shell over them, exercised by hand and by the fake-runner script.
"""

import os
import signal
import sys
import time

from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.containers import Horizontal
from textual.events import Print
from textual.widgets import Footer, RichLog, Static

from . import runs
from .view import SessionStats, render_stats, stats_from_wire

# Right-pane scrollback bound. Old lines beyond it fall out of the pane (the
# display.ansi file and the .jsonl session log remain the complete records).
LOG_MAX_LINES = 20_000

# The stats pane's fixed width (columns), including 1 column of padding each
# side. Wide enough for a 5-digit +/- file table row and the token figures.
STATS_PANE_WIDTH = 44

POLL_INTERVAL = 0.25  # display/stats/liveness poll cadence (seconds)


class ViewerApp(App):
    """Two panes and a footer over a background run's registry files."""

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
        Binding("d", "detach", "background"),
        Binding("p", "pause", "pause"),
        Binding("c", "continue_run", "continue"),
        Binding("enter", "confirm_exit", "exit", show=False),
        Binding("end", "follow", "follow tail"),
    ]

    def __init__(self, name: str, meta: dict):
        super().__init__()
        self._name = name
        self._meta = meta
        # Seeded so the pane isn't blank while the runner is still booting its
        # sandbox (the first stats.json arrives at the first session_begin).
        self.stats = SessionStats(run_name=name, model=meta.get("model") or "")
        self.stats.session_started = time.monotonic()
        self.stats.chain_started = self.stats.session_started
        self.tail: list[str] = []       # result/usage lines, from stats.json
        self._offset = 0                # read position in display.ansi
        self._pending = b""             # partial display line, bytes
        self._interrupt_sent = False    # this viewer asked the runner to stop
        # How the viewer session ended, read by attach_viewer after run():
        # "detach" | "detach-unwinding" | "complete" | "interrupted" | "error".
        self.outcome = "detach"

    @property
    def _display_path(self):
        return runs.run_dir(self._name) / "display.ansi"

    def compose(self):
        with Horizontal():
            yield Static(id="stats")
            yield RichLog(id="log", wrap=True, max_lines=LOG_MAX_LINES,
                          auto_scroll=True)
        yield Footer()

    def on_mount(self):
        theme = self._meta.get("theme")
        if theme:
            self.theme = f"textual-{theme}"
        lines, self._offset, truncated = runs.initial_display(self._display_path)
        log = self.query_one("#log", RichLog)
        if truncated:
            log.write(Text.from_ansi(
                f"[earlier output omitted; full log: {self._display_path}]"))
        for line in lines:
            log.write(Text.from_ansi(line))
        log.scroll_end(animate=False)
        # Stray prints from THIS process (e.g. pricing.py's unknown-model
        # warning, triggered by the cost lookup in render_stats) land in the
        # log instead of corrupting the alt screen.
        self.begin_capture_print(self)
        self.refresh_stats()
        self.set_interval(POLL_INTERVAL, self._poll)
        self.set_interval(1.0, self.refresh_stats)  # the elapsed clock tick

    def on_print(self, event: Print) -> None:
        text = event.text.rstrip("\n")
        if text:
            self._write_log(text)

    # -- the poll: display bytes, stats doc, liveness -------------------------

    def _poll(self) -> None:
        self._drain_display()

        # Stop consuming the stats wire once the run has finished. The doc is
        # final by then (the runner writes its last stats before the meta
        # status that marks the outcome, so the poll that detects the ending
        # has already absorbed the final numbers) -- and re-reading it would
        # actively corrupt the frozen screen: stats_from_wire re-anchors the
        # shipped elapsed seconds against the CURRENT clock, so each reload
        # pushes session_started forward while render_stats holds `now` at
        # finished_at, and the elapsed display ticks backwards one second per
        # poll.
        doc = None if self.stats.finished is not None \
            else runs.load_stats_doc(self._name)
        if doc is not None and isinstance(doc.get("stats"), dict):
            # A torn or foreign doc keeps the last-good stats instead.
            finished, finished_at = self.stats.finished, self.stats.finished_at
            self.stats = stats_from_wire(doc["stats"])
            # Viewer-owned state is deliberately not on the wire; reapply it.
            self.stats.finished, self.stats.finished_at = finished, finished_at
            self.stats.interrupt_requested = self._interrupt_sent
            if isinstance(doc.get("tail"), list):
                self.tail = [str(t) for t in doc["tail"]]

        if self.stats.finished is None:
            meta = runs.load_meta(self._name) or self._meta
            self._meta = meta
            state = runs.classify(meta)
            self.stats.paused = state == "paused"
            if state not in ("running", "paused"):
                self._drain_display()  # the runner's last words, incl. _abort's
                if state == "died":
                    self._write_log(
                        "[runner died without recording an exit status]")
                self.stats.finished = "error" if state == "died" else state
                self.stats.finished_at = time.monotonic()

        self.refresh_stats()

    def _drain_display(self) -> None:
        lines, self._pending, self._offset = runs.tail_display(
            self._display_path, self._offset, self._pending)
        for line in lines:
            self._write_log(line)

    def _write_log(self, text: str) -> None:
        log = self.query_one("#log", RichLog)
        # Follow the tail only while the user is already at it: a reader who
        # scrolled back keeps their place; the bottom (or `end`) re-engages.
        log.write(Text.from_ansi(text), scroll_end=log.is_vertical_scroll_end)

    def refresh_stats(self) -> None:
        pane = self.query_one("#stats", Static)
        width = pane.content_size.width or (STATS_PANE_WIDTH - 2)
        pane.update(Text.from_ansi(render_stats(self.stats, width)))

    # -- user actions ----------------------------------------------------------

    def action_interrupt(self) -> None:
        if self.stats.finished is not None:
            self.outcome = self.stats.finished
            self.exit()  # the run already ended; this is the exit confirmation
            return
        if self._interrupt_sent:
            self.outcome = "detach-unwinding"
            self.exit()  # second press: stop watching the unwind
            return
        # A SIGINT queued on a stopped process is not delivered until it is
        # continued -- interrupting a paused run means waking it first.
        if self.stats.paused:
            runs.continue_run(self._name)
            self.stats.paused = False
        try:
            os.kill(self._meta.get("pid") or 0, signal.SIGINT)
        except (ProcessLookupError, PermissionError):
            pass  # already gone (or not ours); the next poll classifies it
        self._interrupt_sent = True
        self.stats.interrupt_requested = True
        self.refresh_stats()

    def action_pause(self) -> None:
        """Freeze the runner in place (SIGSTOP; see runs.pause_run). The
        sandbox container stays alive, so this is nothing like `gerbil
        resume`: pressing c continues the very same process mid-thought."""
        if self.stats.finished is not None or self._interrupt_sent:
            return
        if runs.pause_run(self._name):
            self.stats.paused = True
            self.refresh_stats()

    def action_continue_run(self) -> None:
        if self.stats.finished is not None:
            return
        if runs.continue_run(self._name):
            self.stats.paused = False
            self.refresh_stats()

    def action_detach(self) -> None:
        if self.stats.finished is not None:
            self.outcome = self.stats.finished  # nothing left to background
        self.exit()

    def action_confirm_exit(self) -> None:
        # Enter exits only the finished screen; mid-run it does nothing (too
        # easy to lean on for it to mean "interrupt" or "detach").
        if self.stats.finished is not None:
            self.outcome = self.stats.finished
            self.exit()

    def action_follow(self) -> None:
        self.query_one("#log", RichLog).scroll_end(animate=False)


def attach_viewer(name: str) -> int:
    """Run a viewer over the named background run; returns the process exit
    code. Detaching leaves the run (and its registry entry) alone; a confirmed
    exit from the finished screen reprints the run's tail lines onto the
    normal terminal -- so the outcome survives in scrollback -- and removes
    the registry entry (the session log and patch live in their usual homes)."""
    meta = runs.load_meta(name)
    if meta is None:
        sys.exit(f"error: no background run named {name!r} "
                 "(see `gerbil ps`)")

    app = ViewerApp(name, meta)
    app.run()

    if app.outcome in ("detach", "detach-unwinding"):
        print(f"detached: run continues in the background "
              f"(reattach: gerbil grab {name})")
        return 0

    for line in app.tail:
        print(line)
    if app.outcome != "complete" and app.stats.session_name:
        # The run ended early; point at the resume command the same way
        # _abort does on a foreground crash (its own line is in the display
        # stream, which is gone with the alt screen).
        session_log = (runs.RUNNING_DIR.parent / "sessions"
                       / f"{app.stats.session_name}.jsonl")
        print(f"resume: gerbil resume {session_log}")
    runs.remove_run(name)
    return {"complete": 0, "interrupted": 130}.get(app.outcome, 1)
