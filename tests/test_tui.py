"""Tests for the live-view plumbing: view.py's stats model, patch parser, and
left-pane renderer; PrintView's byte-compatibility with the historical inline
prints; and RunnerView (the detached runner's PrintView-plus-stats).

No Docker, no network, no real terminal (the textual viewer app itself is
exercised by hand and by tests/fake_runner.py -- see CLAUDE.md; everything
below it is covered here headlessly). The registry itself is covered by
tests/test_runs.py. render.style() is a no-op off a TTY, so output is plain
text and substring asserts work. Run: uv run python tests/test_tui.py
"""

import contextlib
import io

from gerbil import render
from gerbil.pricing import MODEL_PRICING
from gerbil.providers import Usage
import json

from gerbil.view import (
    PrintView,
    SessionStats,
    chain_cost,
    live_cost,
    patch_file_stats,
    render_stats,
    stats_from_wire,
    stats_to_wire,
)


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


# A model with a known price, whatever the table currently holds.
PRICED_MODEL = next(iter(MODEL_PRICING))


# ---------------------------------------------------------------------------
# patch_file_stats
# ---------------------------------------------------------------------------

# A two-commit `format-patch --stdout` with the traps the parser must survive:
# commit-message bullets starting with "-", the `-- ` signature separator
# pausing (not stopping) the count, +++/--- headers, a rename, a binary file,
# a path with spaces, and the same file touched in both commits (summed).
MULTI_COMMIT_PATCH = """\
From 1111111 Mon Sep 17 00:00:00 2001
From: gerbil <gerbil@sandbox>
Date: Tue, 12 Aug 2026 10:00:00 +0000
Subject: [PATCH 1/2] first

message body
- a bullet that must not count as a removal

---
 Foo.lean | 3 ++-
 1 file changed, 2 insertions(+), 1 deletion(-)

diff --git a/Foo.lean b/Foo.lean
index 1111111..2222222 100644
--- a/Foo.lean
+++ b/Foo.lean
@@ -1,2 +1,3 @@
+added one
+added two
-removed one
 context line
diff --git a/Old.lean b/New.lean
similarity index 95%
rename from Old.lean
rename to New.lean
index 3333333..4444444 100644
--- a/Old.lean
+++ b/New.lean
@@ -1 +1 @@
+renamed add
diff --git a/img/pic.png b/img/pic.png
new file mode 100644
index 0000000..5555555
Binary files /dev/null and b/img/pic.png differ
diff --git a/My Dir/My File.lean b/My Dir/My File.lean
index 6666666..7777777 100644
--- a/My Dir/My File.lean
+++ b/My Dir/My File.lean
@@ -1 +1,2 @@
+spaced add
-- 
2.39.0

From 2222222 Mon Sep 17 00:00:00 2001
From: gerbil <gerbil@sandbox>
Date: Tue, 12 Aug 2026 10:05:00 +0000
Subject: [PATCH 2/2] second

- another bullet between commits, still uncounted

---
diff --git a/Foo.lean b/Foo.lean
index 2222222..8888888 100644
--- a/Foo.lean
+++ b/Foo.lean
@@ -1,3 +1,3 @@
+third add
-second removal
-- 
2.39.0

"""


def test_patch_parser_multi_commit() -> None:
    files = patch_file_stats(MULTI_COMMIT_PATCH)
    check("same file summed across commits", files.get("Foo.lean") == (3, 2),
          str(files))
    check("rename counts on the b/ path", files.get("New.lean") == (1, 0),
          str(files))
    check("rename leaves no a/ entry", "Old.lean" not in files, str(files))
    check("binary marked None", "img/pic.png" in files
          and files["img/pic.png"] is None, str(files))
    check("path with spaces", files.get("My Dir/My File.lean") == (1, 0),
          str(files))
    check("no phantom files from message bullets", len(files) == 4, str(files))


def test_patch_parser_edges() -> None:
    check("empty patch", patch_file_stats("") == {}, "")
    # A GIT binary patch's base85 payload lines may start with +/-; the None
    # marker must shield them.
    git_binary = (
        "diff --git a/blob.bin b/blob.bin\n"
        "GIT binary patch\n"
        "literal 10\n"
        "-c$aBC0ssIL10VntA\n"
        "+zzz\n"
    )
    check("GIT binary payload not counted",
          patch_file_stats(git_binary) == {"blob.bin": None},
          str(patch_file_stats(git_binary)))
    # +++/--- are headers, not content.
    plain = (
        "diff --git a/A.lean b/A.lean\n"
        "--- a/A.lean\n"
        "+++ b/A.lean\n"
        "+x\n"
    )
    check("file headers excluded", patch_file_stats(plain) == {"A.lean": (1, 0)},
          str(patch_file_stats(plain)))


# ---------------------------------------------------------------------------
# SessionStats event replay + costs
# ---------------------------------------------------------------------------


def test_stats_replay() -> None:
    stats = SessionStats()
    stats.on_session_begin(
        name="gerbil-1", model=PRICED_MODEL, small_model=None,
        ralph={"iteration": 1, "total": 2, "chain_base": "abc", "ancestors": []},
        resumed_from=None, now=100.0,
    )
    stats.on_turn_header(200_000)
    stats.on_turn_complete(Usage(input_tokens=1000, output_tokens=100), zoom=False)
    stats.on_turn_complete(Usage(input_tokens=2000, output_tokens=200), zoom=False)

    check("turns counted", stats.turns == 2, str(stats.turns))
    check("window adopted", stats.max_context == 200_000, str(stats.max_context))
    check("last context is the latest turn's",
          stats.last_context_tokens == 2200, str(stats.last_context_tokens))
    check("outer bucket accumulates",
          stats.outer.input_tokens == 3000 and stats.outer.output_tokens == 300,
          str(stats.outer))

    cost_before_zoom = live_cost(stats)
    check("live cost priced for a known model",
          cost_before_zoom is not None and cost_before_zoom > 0,
          str(cost_before_zoom))

    # A zoom sub-session: its turns and tokens land in the inner bucket, its
    # window must not clobber the outer gauge.
    stats.on_zoom_begin("Foo.lean:3", "small-model-x", 32_000)
    stats.on_turn_header(32_000)  # the zoom's own header
    stats.on_turn_complete(Usage(input_tokens=500, output_tokens=50), zoom=True)
    check("zoom turn counted separately",
          stats.zoom_turns == 1 and stats.turns == 2,
          f"{stats.zoom_turns}/{stats.turns}")
    check("outer window survives the zoom", stats.max_context == 200_000,
          str(stats.max_context))
    check("inner bucket separate", stats.inner.input_tokens == 500,
          str(stats.inner))
    stats.on_zoom_end()
    check("zoom cleared", stats.zoom_active is None, str(stats.zoom_active))

    # The zoom's small model is unknown to pricing, so the whole estimate goes
    # N/A rather than a made-up partial number (mirrors agent.py).
    with contextlib.redirect_stderr(io.StringIO()):
        check("unpriced small model makes cost N/A", live_cost(stats) is None, "")

    # Second ralph session: per-session state resets, chain buckets keep going.
    stats.on_session_begin(
        name="gerbil-2", model=PRICED_MODEL, small_model=None,
        ralph={"iteration": 2, "total": 2, "chain_base": "abc", "ancestors": []},
        resumed_from=None, now=400.0,
    )
    check("session counters reset",
          stats.turns == 0 and stats.zoom_turns == 0
          and stats.outer.input_tokens == 0 and stats.files == {},
          str(stats))
    check("chain buckets survive",
          stats.chain_outer.input_tokens == 3000
          and stats.chain_inner.input_tokens == 500,
          f"{stats.chain_outer}/{stats.chain_inner}")
    check("chain clock keeps its anchor", stats.chain_started == 100.0
          and stats.session_started == 400.0,
          f"{stats.chain_started}/{stats.session_started}")

    stats.on_turn_complete(Usage(input_tokens=100, output_tokens=10), zoom=False)
    check("chain accumulation continues",
          stats.chain_outer.input_tokens == 3100,
          str(stats.chain_outer.input_tokens))
    check("chain cost covers both sessions", chain_cost(stats) is None
          or chain_cost(stats) >= (live_cost(stats) or 0), "")


# ---------------------------------------------------------------------------
# render_stats
# ---------------------------------------------------------------------------


def test_render_stats() -> None:
    stats = SessionStats()
    stats.on_session_begin(
        name="gerbil-260812", model="some-model", small_model=None,
        ralph={"iteration": 2, "total": 5, "chain_base": "", "ancestors": []},
        resumed_from=None, now=0.0,
    )
    stats.chain_started = 0.0
    stats.session_started = 100.0
    stats.on_turn_header(100_000)
    stats.on_turn_complete(Usage(input_tokens=60_000, output_tokens=2_000),
                           zoom=False)
    stats.files = {
        "Small.lean": (1, 0),
        "Big/Change.lean": (120, 8),
        "img.png": None,
    }
    with contextlib.redirect_stderr(io.StringIO()):  # unknown-pricing warning
        out = render_stats(stats, 40, now=3823.0)  # 3723s session, 3823s chain

    check("model shown", "some-model" in out, out)
    check("session name shown", "gerbil-260812" in out, out)
    check("ralph i/N", "2/5" in out, out)
    check("elapsed hh:mm:ss", "01:02:03" in out, out)
    check("chain elapsed", "01:03:43" in out, out)
    check("turns", "turns 1" in out, out)
    check("context percent", "62.0%" in out and "62,000 / 100,000" in out, out)
    check("token totals", "in 60,000" in out and "out 2,000" in out, out)
    check("unknown model cost is N/A", "N/A" in out, out)
    check("file table lives in its own pane now",
          "Big/Change.lean" not in out, out)

    # Zoom indicator and interrupt banner.
    stats.on_zoom_begin("Foo.lean:42", "small-x", 10_000)
    stats.last_zoom_context_tokens = 1_800
    stats.interrupt_requested = True
    with contextlib.redirect_stderr(io.StringIO()):
        out2 = render_stats(stats, 40, now=3823.0)
    check("zoom line", "zoom: Foo.lean:42" in out2 and "18.0%" in out2, out2)
    check("interrupt banner", "interrupting" in out2, out2)

    # Unknown window: raw token count, no percentage.
    stats2 = SessionStats()
    stats2.on_session_begin(name="s", model="m", small_model=None, ralph=None,
                            resumed_from=None, now=0.0)
    stats2.on_turn_complete(Usage(input_tokens=123_456), zoom=False)
    with contextlib.redirect_stderr(io.StringIO()):
        out3 = render_stats(stats2, 40, now=1.0)
    check("unknown window shows raw tokens",
          "123,456 tokens (window unknown)" in out3, out3)
    check("no ralph rows for a single session", "ralph" not in out3, out3)
    check("running status always shown", "running" in out3, out3)
    check("gap after the banner (trailing blank line)",
          out3.endswith("\n"), repr(out3[-20:]))


def test_file_summary() -> None:
    """The scrollable file pane: tree layout, chain accumulation, wire."""
    from gerbil import render
    from gerbil.view import file_summary, merge_file_stats

    tee, corner = render.GLYPHS["tee"], render.GLYPHS["corner"]
    pipe, blank = render.GLYPHS["pipe"], render.GLYPHS["blank"]

    stats = SessionStats()
    stats.on_session_begin(name="s", model="m", small_model=None, ralph=None,
                           resumed_from=None, now=0.0)
    out = file_summary(stats, 40)
    check("placeholder before any diff", "(none yet)" in out, out)
    check("no chain section for a single session", "chain" not in out, out)

    stats.files = {
        "Toy/Basic.lean": (120, 8),
        "Toy/Sub/Deep/X.lean": (1, 0),
        "Root.lean": (2, 1),
        "img.png": None,
    }
    out = file_summary(stats, 40)
    lines = out.splitlines()
    check("directories first, tree connectors drawn",
          any(l.startswith(f"{tee}Toy/") for l in lines)
          and any(l.startswith(f"{corner}img.png") for l in lines), out)
    check("single-child directory chain compressed",
          any(l.startswith(f"{pipe}{tee}Sub/Deep/") for l in lines), out)
    check("nested leaf under the compressed chain",
          any(l.startswith(f"{pipe}{pipe}{corner}X.lean") and "+1" in l
              for l in lines), out)
    check("per-file figures right-aligned",
          any("Basic.lean" in l and l.rstrip().endswith("-8") and "+120" in l
              for l in lines), out)
    check("binary marked bin",
          any("img.png" in l and l.rstrip().endswith("bin") for l in lines),
          out)
    check("totals row", any("total (4 files)" in l and "+123" in l
                            and "-9" in l for l in lines), out)

    # A ralph chain: session 1's diff folds into the chain at the next
    # session_begin, and the chain section shows the running sum.
    chain = SessionStats()
    ralph1 = {"iteration": 1, "total": 3, "chain_base": "", "ancestors": []}
    chain.on_session_begin(name="s1", model="m", small_model=None,
                           ralph=ralph1, resumed_from=None, now=0.0)
    chain.files = {"A.lean": (10, 2), "B.lean": (1, 1), "img.png": None}
    chain.on_session_begin(name="s2", model="m", small_model=None,
                           ralph={**ralph1, "iteration": 2},
                           resumed_from=None, now=5.0)
    check("session_begin folds files into the chain",
          chain.chain_files == {"A.lean": (10, 2), "B.lean": (1, 1),
                                "img.png": None}
          and chain.files == {}, str(chain.chain_files))
    chain.files = {"A.lean": (5, 5), "C.lean": (7, 0)}
    out = file_summary(chain, 40)
    check("both sections shown mid-chain",
          "files (this session)" in out and "files (chain)" in out, out)
    session_part, chain_part = out.split("files (chain)")
    check("chain sums this session on top of finished ones",
          any("A.lean" in l and "+15" in l and "-7" in l
              for l in chain_part.splitlines())
          and any("C.lean" in l and "+7" in l
                  for l in chain_part.splitlines()), chain_part)
    check("session section shows only the live diff",
          not any("B.lean" in l for l in session_part.splitlines()),
          session_part)
    check("binary absorbs in a merge",
          merge_file_stats({"x": (1, 1)}, {"x": None}) == {"x": None}
          and merge_file_stats({"x": None}, {"x": (1, 1)}) == {"x": None})

    # chain_files crosses the wire like files.
    doc = json.loads(json.dumps(stats_to_wire(chain, now=6.0)))
    r = stats_from_wire(doc, now=6.0)
    check("chain_files round-trips the wire",
          r.chain_files == chain.chain_files, str(r.chain_files))


def test_render_stats_finished() -> None:
    """The end-of-run hold: a finished banner replaces the interrupt one and
    the clocks freeze at finished_at, not at render time."""
    stats = SessionStats()
    stats.on_session_begin(name="s", model="m", small_model=None, ralph=None,
                           resumed_from=None, now=0.0)
    stats.interrupt_requested = True  # superseded once finished
    stats.finished = "complete"
    stats.finished_at = 60.0
    with contextlib.redirect_stderr(io.StringIO()):
        out = render_stats(stats, 40, now=9_999.0)
    check("finished banner shown", "session complete" in out, out)
    check("exit hint shown", "press q or enter to exit" in out, out)
    check("interrupt banner superseded", "interrupting" not in out, out)
    check("clock frozen at finished_at", "00:01:00" in out, out)

    for kind in ("interrupted", "error"):
        stats.finished = kind
        with contextlib.redirect_stderr(io.StringIO()):
            out = render_stats(stats, 40, now=9_999.0)
        check(f"finished banner: {kind}", f"session {kind}" in out, out)

    # The paused banner shows only while nothing weightier is going on.
    stats.finished = None
    stats.finished_at = None
    stats.interrupt_requested = False
    stats.paused = True
    with contextlib.redirect_stderr(io.StringIO()):
        out = render_stats(stats, 40, now=9_999.0)
    check("paused banner", "paused: press c to continue" in out, out)
    check("paused replaces running", "running" not in out, out)
    stats.interrupt_requested = True
    with contextlib.redirect_stderr(io.StringIO()):
        out = render_stats(stats, 40, now=9_999.0)
    check("interrupt banner outranks paused", "interrupting" in out
          and "paused:" not in out, out)

    # "stopping loop..." -- the ralph_done check passed and only teardown
    # remains. Beats paused/running; outranked by interrupt and finished.
    stats.interrupt_requested = False
    stats.stopping = True
    with contextlib.redirect_stderr(io.StringIO()):
        out = render_stats(stats, 40, now=9_999.0)
    check("stopping banner outranks paused",
          "stopping loop..." in out and "paused:" not in out
          and "running" not in out, out)
    stats.interrupt_requested = True
    with contextlib.redirect_stderr(io.StringIO()):
        out = render_stats(stats, 40, now=9_999.0)
    check("interrupt outranks stopping",
          "interrupting" in out and "stopping loop" not in out, out)
    stats.finished = "complete"
    stats.finished_at = 60.0
    with contextlib.redirect_stderr(io.StringIO()):
        out = render_stats(stats, 40, now=9_999.0)
    check("finished outranks stopping",
          "session complete" in out and "stopping loop" not in out, out)


def test_resolve_theme() -> None:
    """`theme` in .gerbil/config.toml: light/dark accepted, absence is None,
    anything else is a preflight error; the user-level ~/.gerbil/config.toml
    supplies a default the project config overrides."""
    import os
    import tempfile
    from pathlib import Path

    from gerbil.cli import _resolve_theme

    tmp = Path(tempfile.mkdtemp())
    home = tmp / "home"
    (home / ".gerbil").mkdir(parents=True)
    saved_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home)  # hermetic: never read the developer's
    try:
        root = tmp / "proj"
        root.mkdir()
        check("no config file -> default", _resolve_theme(root) is None, "")

        cfg = root / ".gerbil"
        cfg.mkdir()
        (cfg / "config.toml").write_text('image = "custom:latest"\n')
        check("config without theme -> default", _resolve_theme(root) is None, "")

        for value in ("light", "dark"):
            (cfg / "config.toml").write_text(f'theme = "{value}"\n')
            check(f"theme {value} accepted", _resolve_theme(root) == value, "")

        (cfg / "config.toml").write_text('theme = "solarized"\n')
        try:
            _resolve_theme(root)
            check("invalid theme rejected", False, "no SystemExit")
        except SystemExit as exc:
            check("invalid theme rejected", "light" in str(exc), str(exc))

        user_cfg = home / ".gerbil" / "config.toml"
        user_cfg.write_text('theme = "light"\n')
        (cfg / "config.toml").write_text('image = "custom:latest"\n')
        check("user config supplies the default",
              _resolve_theme(root) == "light", "")
        (cfg / "config.toml").write_text('theme = "dark"\n')
        check("project config overrides the user's",
              _resolve_theme(root) == "dark", "")
        user_cfg.write_text('theme = "solarized"\n')
        (cfg / "config.toml").unlink()
        try:
            _resolve_theme(root)
            check("invalid user theme rejected", False, "no SystemExit")
        except SystemExit as exc:
            check("invalid user theme rejected, naming the user config",
                  str(home) in str(exc), str(exc))
    finally:
        if saved_home is not None:
            os.environ["HOME"] = saved_home


# ---------------------------------------------------------------------------
# PrintView byte-compatibility with the historical inline prints
# ---------------------------------------------------------------------------


class FakeUsage:
    context_tokens = 96_000


class FrozenDatetime:
    """Pin render.turn_header's wall-clock stamp for an exact comparison."""

    @staticmethod
    def now():
        class _Stamp:
            @staticmethod
            def strftime(fmt):
                return "12:34:56"
        return _Stamp()


def _capture(fn) -> tuple[str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        fn()
    return out.getvalue(), err.getvalue()


def test_printview_byte_compat() -> None:
    pv = PrintView()
    check("PrintView never asks for the wip patch",
          pv.wants_wip_patch is False, "")

    real_dt = render.datetime
    render.datetime = FrozenDatetime
    try:
        got, _ = _capture(lambda: pv.turn_header(
            "turn 3", "bold", "dark_red", max_context=100_000, usage=FakeUsage()))
        want = "\n" + render.turn_header(
            "turn 3", "bold", "dark_red", max_context=100_000, usage=FakeUsage()
        ) + "\n"
        check("turn_header == historical print", got == want,
              f"{got!r} != {want!r}")
    finally:
        render.datetime = real_dt

    got, _ = _capture(lambda: pv.tool_call("bash", {"command": "ls"}))
    want = "\n" + render.format_tool_call("bash", {"command": "ls"}, None) + "\n"
    check("tool_call == historical print", got == want, f"{got!r} != {want!r}")

    got, _ = _capture(lambda: pv.tool_result("bash", "ok", "ok", False))
    want = render.format_tool_result("bash", "ok", "ok", False) + "\n"
    check("tool_result == historical print", got == want, f"{got!r} != {want!r}")

    got, _ = _capture(lambda: pv.banner("[context window: 1 tokens]"))
    check("banner: no leading newline", got == "[context window: 1 tokens]\n",
          repr(got))

    got, _ = _capture(lambda: pv.notice("[note]"))
    check("notice: leading newline by default", got == "\n[note]\n", repr(got))
    got, _ = _capture(lambda: pv.notice("[note]", newline_before=False))
    check("notice: newline_before=False", got == "[note]\n", repr(got))
    got, _ = _capture(lambda: pv.notice("", newline_before=False))
    check("empty notice is the bare print()", got == "\n", repr(got))
    got, err = _capture(lambda: pv.notice("warn", newline_before=False,
                                          stderr=True))
    check("stderr notice routes to stderr", got == "" and err == "warn\n",
          f"{got!r}/{err!r}")

    got, _ = _capture(pv.turn_end)
    check("turn_end is a blank line", got == "\n", repr(got))

    got, _ = _capture(lambda: pv.zoom_begin("Foo.lean:3", "small-x", None))
    check("zoom_begin banner",
          got == "\n===== zoom in: Foo.lean:3 (small-x) =====\n", repr(got))
    got, _ = _capture(lambda: pv.zoom_end("===== zoom out (2 turns) ====="))
    check("zoom_end", got == "\n===== zoom out (2 turns) =====\n", repr(got))

    usage = Usage(input_tokens=1000, output_tokens=50, thinking_tokens=10,
                  cache_read_tokens=200, cache_write_tokens=30)
    got, _ = _capture(lambda: pv.usage_summary(4, usage, 1.25))
    want, _ = _capture(lambda: render.print_usage(4, usage, 1.25))
    check("usage_summary == print_usage", got == want, f"{got!r} != {want!r}")

    got, _ = _capture(lambda: pv.result_line("session: /tmp/x"))
    check("result_line", got == "session: /tmp/x\n", repr(got))

    # The data feeds are silent no-ops.
    got, err = _capture(lambda: (
        pv.session_begin(name="n", model="m", small_model=None, ralph=None,
                         resumed_from=None),
        pv.turn_complete(usage, zoom=False),
        pv.wip_patch("diff --git a/X b/X"),
    ))
    check("data feeds print nothing", got == "" and err == "",
          f"{got!r}/{err!r}")


# ---------------------------------------------------------------------------
# RunnerView: the detached runner's PrintView-plus-stats (stub registry dir)
# ---------------------------------------------------------------------------


def test_runner_view() -> None:
    import json
    import tempfile
    from pathlib import Path

    from gerbil import view as view_mod
    from gerbil.view import RunnerView, stats_from_wire

    d = Path(tempfile.mkdtemp()) / "brave-otter"
    d.mkdir()
    rv = RunnerView(d)
    pv = PrintView()
    check("RunnerView wants the wip patch", rv.wants_wip_patch is True, "")
    check("run name seeded from the dir", rv.stats.run_name == "brave-otter",
          rv.stats.run_name)

    # Every non-overridden method is byte-identical to PrintView -- the
    # display stream the viewer tails IS the classic stream.
    cases = [
        ("banner", lambda v: v.banner("[context window: 1 tokens]")),
        ("assistant_delta", lambda v: v.assistant_delta("stream chunk")),
        ("tool_call", lambda v: v.tool_call("bash", {"command": "ls"})),
        ("tool_result", lambda v: v.tool_result("bash", "ok", "ok", False)),
        ("notice", lambda v: v.notice("[note]")),
        ("turn_end", lambda v: v.turn_end()),
        ("zoom_begin", lambda v: v.zoom_begin("Foo.lean:3", "small-x", None)),
        ("zoom_end", lambda v: v.zoom_end("===== zoom out (2 turns) =====")),
        ("result_line", lambda v: v.result_line("session: /tmp/x")),
        ("usage_summary", lambda v: v.usage_summary(
            1, Usage(input_tokens=10, output_tokens=1), None)),
        ("loop_stopping", lambda v: v.loop_stopping()),
    ]
    for label, call in cases:
        got, _ = _capture(lambda: call(rv))
        want, _ = _capture(lambda: call(pv))
        check(f"runner {label} == PrintView", got == want,
              f"{got!r} != {want!r}")

    # turn_header is the one display divergence: a compact divider, not the
    # full-width rule (wrong inside a viewer pane).
    got, _ = _capture(lambda: rv.turn_header(
        "turn 1", "bold", max_context=100_000, usage=None))
    rule, sep = render.GLYPHS["rule"], render.GLYPHS["sep"]
    check("compact turn_header",
          got.startswith(f"\n{rule * 2} turn 1 {sep} ") and got.endswith("\n")
          and rule * 10 not in got, repr(got))
    check("turn_header feeds the window", rv.stats.max_context == 100_000,
          str(rv.stats.max_context))

    # Stats persistence: the sidecar parses back through the wire format.
    with contextlib.redirect_stdout(io.StringIO()):
        rv.session_begin(name="gerbil-x-01", model=PRICED_MODEL,
                         small_model=None, ralph=None, resumed_from=None)
        rv.turn_complete(Usage(input_tokens=10, output_tokens=2), zoom=False)
        rv.wip_patch("diff --git a/A.lean b/A.lean\n+x\n")
        rv.result_line("session: /tmp/s.jsonl")
        rv.usage_summary(1, Usage(input_tokens=10, output_tokens=2), None)
    doc = json.loads((d / "stats.json").read_text())
    s = stats_from_wire(doc["stats"])
    check("stats.json round-trips",
          s.turns == 1 and s.files == {"A.lean": (1, 0)}
          and s.session_name == "gerbil-x-01" and s.run_name == "brave-otter",
          str(doc["stats"]))
    check("session_begin cleared an earlier stopping flag",
          s.stopping is False, str(doc["stats"]))
    rv.loop_stopping()
    doc = json.loads((d / "stats.json").read_text())
    check("loop_stopping persists immediately",
          stats_from_wire(doc["stats"]).stopping is True, str(doc["stats"]))
    # (the byte-compat sweep above also exercised result_line/usage_summary,
    # so the tail holds those entries too -- assert on the latest pair)
    check("tail carried in the doc", len(doc["tail"]) >= 2
          and doc["tail"][-2] == "session: /tmp/s.jsonl", str(doc["tail"]))

    # Registry I/O failure must never escape a view method.
    real_write = view_mod.runs.write_stats_doc
    view_mod.runs.write_stats_doc = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("disk gone"))
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rv.turn_complete(Usage(input_tokens=1), zoom=False)
        check("failing stats write never raises", True, "")
    finally:
        view_mod.runs.write_stats_doc = real_write

def main() -> None:
    test_patch_parser_multi_commit()
    test_patch_parser_edges()
    test_stats_replay()
    test_render_stats()
    test_file_summary()
    test_render_stats_finished()
    test_resolve_theme()
    test_printview_byte_compat()
    test_runner_view()
    print("\nAll tui tests passed.")


if __name__ == "__main__":
    main()
