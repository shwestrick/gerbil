"""Tests for the background-run registry (runs.py) and the running/grab
commands: naming, meta round-trips, liveness classification, display tailing,
the SessionStats wire format, the runner exit-status wrapper, and the
cmd_running/cmd_grab behavior against fabricated run dirs.

No Docker, no network, no terminal. The textual viewer that consumes all of
this is exercised by hand (see tests/fake_runner.py).
Run: uv run python tests/test_runs.py
"""

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from pathlib import Path

from gerbil import runs
from gerbil.providers import Usage
from gerbil.view import SessionStats, stats_from_wire, stats_to_wire


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed: {label}\n{detail}")


@contextlib.contextmanager
def tmp_running_dir():
    """Point runs.RUNNING_DIR at a scratch dir for the duration."""
    real = runs.RUNNING_DIR
    runs.RUNNING_DIR = Path(tempfile.mkdtemp()) / "running"
    try:
        yield runs.RUNNING_DIR
    finally:
        runs.RUNNING_DIR = real


def test_names() -> None:
    with tmp_running_dir():
        name = runs.new_run_name()
        adj, _, animal = name.partition("-")
        check("name is adjective-animal",
              adj in runs._ADJECTIVES and animal in runs._ANIMALS, name)

        # Shrink the wordlists to force collisions -> numeric suffix.
        real = runs._ADJECTIVES, runs._ANIMALS
        runs._ADJECTIVES, runs._ANIMALS = ("only",), ("name",)
        try:
            check("unique against taken",
                  runs.new_run_name(taken={"only-name"}) == "only-name-2", "")
            check("suffix increments",
                  runs.new_run_name(taken={"only-name", "only-name-2"})
                  == "only-name-3", "")
        finally:
            runs._ADJECTIVES, runs._ANIMALS = real

        # Existing run dirs count as taken.
        runs.create_run("only-taken", project_dir=Path("/p"), model="m",
                        theme=None, command=[])
        check("existing dirs are taken",
              "only-taken" not in {runs.new_run_name() for _ in range(50)}
              or len(runs._ADJECTIVES) > 1, "")


def test_meta_roundtrip() -> None:
    with tmp_running_dir() as rd:
        d = runs.create_run("calm-vole", project_dir=Path("/proj"), model="m1",
                            theme="dark", command=["run", "--prompt", "p.md"])
        check("create_run makes the dir", d.is_dir() and
              (d / "display.ansi").exists(), str(d))
        meta = runs.load_meta("calm-vole")
        check("meta round-trip", meta is not None
              and meta["status"] == "starting" and meta["pid"] is None
              and meta["model"] == "m1" and meta["theme"] == "dark", str(meta))

        runs.save_meta("calm-vole", pid=1234, status="running")
        meta = runs.load_meta("calm-vole")
        check("save_meta merges", meta["pid"] == 1234
              and meta["status"] == "running" and meta["model"] == "m1",
              str(meta))
        check("atomic write leaves no tmp residue",
              not list(d.glob("*.tmp")), str(list(d.iterdir())))

        (d / "meta.json").write_text("{not json")
        check("corrupt meta -> None", runs.load_meta("calm-vole") is None, "")
        check("missing run -> None", runs.load_meta("no-such") is None, "")
        check("stats doc of a fresh run -> None",
              runs.load_stats_doc("calm-vole") is None, "")


def test_liveness() -> None:
    check("own pid alive", runs.pid_alive(os.getpid()), "")
    check("pid None dead", not runs.pid_alive(None), "")
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    check("reaped child dead", not runs.pid_alive(proc.pid), str(proc.pid))

    live, dead = os.getpid(), proc.pid
    check("terminal status wins over a live pid",
          runs.classify({"status": "complete", "pid": live}) == "complete", "")
    check("running + live pid", runs.classify(
        {"status": "running", "pid": live}) == "running", "")
    check("running + dead pid -> died", runs.classify(
        {"status": "running", "pid": dead}) == "died", "")
    check("starting + no pid -> died", runs.classify(
        {"status": "starting", "pid": None}) == "died", "")


def test_tail_display() -> None:
    d = Path(tempfile.mkdtemp())
    f = d / "display.ansi"
    f.write_bytes(b"hello\nwor")
    lines, pending, offset = runs.tail_display(f, 0, b"")
    check("complete line out, partial held",
          lines == ["hello"] and pending == b"wor" and offset == 9,
          f"{lines}/{pending!r}/{offset}")
    with f.open("ab") as fh:
        fh.write(b"ld\n")
    lines, pending, offset = runs.tail_display(f, offset, pending)
    check("held partial completes", lines == ["world"] and pending == b"",
          f"{lines}/{pending!r}")
    lines, pending, offset2 = runs.tail_display(f, offset, pending)
    check("no growth -> nothing", lines == [] and offset2 == offset, "")

    # A UTF-8 multibyte character torn across two reads must survive.
    f2 = d / "multi.ansi"
    f2.write_bytes("é".encode()[:1])
    lines, pending, offset = runs.tail_display(f2, 0, b"")
    check("torn multibyte held as bytes", lines == [] and len(pending) == 1, "")
    with f2.open("ab") as fh:
        fh.write("é".encode()[1:] + b"\n")
    lines, pending, offset = runs.tail_display(f2, offset, pending)
    check("torn multibyte decodes whole", lines == ["é"], str(lines))

    # initial_display: bounded, line-aligned, leaves the partial unread.
    f3 = d / "init.ansi"
    f3.write_bytes(b"a\nb\npartial")
    lines, offset, truncated = runs.initial_display(f3)
    check("initial load: complete lines only",
          lines == ["a", "b"] and offset == 4 and not truncated,
          f"{lines}/{offset}")
    lines, pending, _ = runs.tail_display(f3, offset, b"")
    check("partial picked up by the tail",
          lines == [] and pending == b"partial", f"{lines}/{pending!r}")

    f4 = d / "big.ansi"
    f4.write_bytes(b"x" * 100 + b"\n" + b"tail-line\n")
    lines, offset, truncated = runs.initial_display(f4, limit=50)
    check("oversize file truncates to whole lines",
          truncated and lines == ["tail-line"], f"{truncated}/{lines}")


def test_wire_roundtrip() -> None:
    s = SessionStats(run_name="calm-vole", model="m", small_model="sm",
                     session_name="gerbil-x-02", ralph_iteration=2,
                     ralph_total=5, turns=7, zoom_turns=3,
                     max_context=200_000, last_context_tokens=42_000,
                     zoom_active="Foo.lean:9", zoom_max_context=32_000)
    s.session_started = 40.0   # at now=100: 60s elapsed
    s.chain_started = 10.0     # at now=100: 90s elapsed
    s.outer = Usage(input_tokens=100, output_tokens=10, thinking_tokens=2,
                    cache_read_tokens=50, cache_write_tokens=5)
    s.chain_outer = Usage(input_tokens=300)
    s.files = {"A.lean": (3, 1), "img.png": None}
    s.interrupt_requested = True   # viewer-owned: must NOT cross the wire
    s.finished = "complete"

    doc = json.loads(json.dumps(stats_to_wire(s, now=100.0)))
    r = stats_from_wire(doc, now=250.0)
    check("anchors shift by exactly the now delta",
          r.session_started == 190.0 and r.chain_started == 160.0,
          f"{r.session_started}/{r.chain_started}")
    check("counters and identity survive",
          (r.turns, r.zoom_turns, r.run_name, r.session_name,
           r.ralph_iteration, r.ralph_total, r.zoom_active)
          == (7, 3, "calm-vole", "gerbil-x-02", 2, 5, "Foo.lean:9"), str(r))
    check("usage buckets survive", r.outer == s.outer
          and r.chain_outer == s.chain_outer and r.inner == Usage(), str(r))
    check("files tuples and None survive",
          r.files == {"A.lean": (3, 1), "img.png": None}, str(r.files))
    check("viewer-owned state excluded from the wire",
          "finished" not in doc and "interrupt_requested" not in doc
          and r.finished is None and not r.interrupt_requested, str(doc))

    degraded = stats_from_wire({"turns": 1, "unknown_future_key": [1, 2]},
                               now=5.0)
    check("foreign doc degrades, never crashes",
          degraded.turns == 1 and degraded.model == "", str(degraded))
    check("malformed files entry skipped", stats_from_wire(
        {"files": {"ok.lean": [1, 2], "bad": "nope"}}).files
        == {"ok.lean": (1, 2)}, "")


def test_run_runner() -> None:
    def status_after(body) -> tuple[str, BaseException | None]:
        with tmp_running_dir():
            runs.create_run("t-run", project_dir=Path("/p"), model="m",
                            theme=None, command=[])
            caught = None
            try:
                runs.run_runner("t-run", body)
            except BaseException as exc:
                caught = exc
            return runs.load_meta("t-run")["status"], caught

    status, exc = status_after(lambda: None)
    check("clean return -> complete", status == "complete" and exc is None, status)
    status, exc = status_after(lambda: sys.exit(0))
    check("SystemExit(0) -> complete",
          status == "complete" and isinstance(exc, SystemExit), status)
    status, exc = status_after(lambda: sys.exit(130))
    check("SystemExit(130) -> interrupted (the _abort convention)",
          status == "interrupted" and isinstance(exc, SystemExit), status)
    status, exc = status_after(lambda: sys.exit(1))
    check("SystemExit(1) -> error", status == "error", status)

    def raise_kbd():
        raise KeyboardInterrupt

    status, exc = status_after(raise_kbd)
    check("KeyboardInterrupt -> interrupted, re-raised",
          status == "interrupted" and isinstance(exc, KeyboardInterrupt), status)

    def raise_err():
        raise RuntimeError("boom")

    status, exc = status_after(raise_err)
    check("crash -> error, re-raised",
          status == "error" and isinstance(exc, RuntimeError), status)

    with tmp_running_dir():
        runs.create_run("t-pid", project_dir=Path("/p"), model="m",
                        theme=None, command=[])
        seen = {}
        runs.run_runner("t-pid", lambda: seen.update(
            pid=runs.load_meta("t-pid")["pid"],
            status=runs.load_meta("t-pid")["status"]))
        check("runner records its pid before the body",
              seen == {"pid": os.getpid(), "status": "running"}, str(seen))


def _fab_run(name: str, *, status: str, pid, turns: int = 0,
             session: str = "") -> None:
    """Fabricate a registered run for the command tests."""
    runs.create_run(name, project_dir=Path(f"/projects/{name}"), model="m-x",
                    theme=None, command=[])
    runs.save_meta(name, pid=pid, status=status,
                   started_at=time.time() - 90)
    if turns or session:
        runs.write_stats_doc(runs.run_dir(name), {
            "v": 1, "written_at": time.time(),
            "stats": {"turns": turns, "session_name": session,
                      "last_context_tokens": 50_000, "max_context": 100_000},
            "tail": ["session: /tmp/x.jsonl"],
        })


def test_cmd_running_and_grab() -> None:
    from gerbil.cli import cmd_grab, cmd_running

    with tmp_running_dir():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cmd_running(Namespace())
        check("empty registry message", "no background runs." in out.getvalue(),
              out.getvalue())

        _fab_run("live-run", status="running", pid=os.getpid(), turns=4,
                 session="gerbil-260812-01")
        _fab_run("done-run", status="complete", pid=None, turns=9)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            cmd_running(Namespace())
        text = out.getvalue()
        check("live run listed", "live-run" in text and "running" in text
              and "gerbil-260812-01" in text and "50%" in text, text)
        check("finished run listed once", "done-run" in text
              and "complete" in text, text)
        check("grab hint names a live run", "gerbil grab live-run" in text, text)
        check("finished run pruned after listing",
              not runs.run_dir("done-run").exists()
              and runs.run_dir("live-run").exists(), "")

        # grab: bare with one live run resolves it -- but stdout is not a tty
        # here, so it must exit with the tail -f pointer instead of attaching.
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                cmd_grab(Namespace(name=None))
            check("grab without a tty exits", False, "no SystemExit")
        except SystemExit as exc:
            check("grab without a tty points at the display file",
                  "display.ansi" in str(exc), str(exc))

        try:
            cmd_grab(Namespace(name="no-such-run"))
            check("grab unknown name exits", False, "no SystemExit")
        except SystemExit as exc:
            check("grab unknown name lists live runs",
                  "no-such-run" in str(exc) and "live-run" in str(exc),
                  str(exc))

        _fab_run("other-run", status="running", pid=os.getpid())
        try:
            cmd_grab(Namespace(name=None))
            check("ambiguous grab exits", False, "no SystemExit")
        except SystemExit as exc:
            check("ambiguous grab lists candidates",
                  "live-run" in str(exc) and "other-run" in str(exc), str(exc))

    with tmp_running_dir():
        try:
            cmd_grab(Namespace(name=None))
            check("grab with no runs exits", False, "no SystemExit")
        except SystemExit as exc:
            check("grab with no runs says so", "no background runs" in str(exc),
                  str(exc))


def main() -> None:
    test_names()
    test_meta_roundtrip()
    test_liveness()
    test_tail_display()
    test_wire_roundtrip()
    test_run_runner()
    test_cmd_running_and_grab()
    print("\nAll runs tests passed.")


if __name__ == "__main__":
    main()
