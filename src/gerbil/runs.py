"""The background-run registry -- what makes gerbil runs detachable.

In TUI mode `gerbil run`/`gerbil resume` never run the session in the process
attached to the terminal: cli._spawn_and_attach starts a detached child (the
*runner*, re-invoked with the hidden --_runner NAME flag) and the terminal
process becomes a disposable *viewer* over it (tui.attach_viewer). Everything
the two processes share lives in one directory per run:

    ~/.gerbil/running/<name>/
        meta.json      identity + liveness: pid, status, project, model, theme
        stats.json     the live left-pane state (view.stats_to_wire) + the
                       result/usage tail lines, rewritten a few times per turn
        display.ansi   the runner's stdout+stderr -- the styled classic stream
                       (GERBIL_FORCE_STYLE keeps the ANSI on), which the viewer
                       tails into its log pane

Names are adjective-animal ("brave-otter"): short, pronounceable, unique among
the runs that currently exist.

Failure discipline: writers go through a tmp file + os.replace (the viewer
polls several times a second, so torn reads would be routine, not rare -- this
is the one place gerbil needs atomic replace), never raise on I/O (a registry
hiccup must never cost a session), and every reader tolerates a missing or
corrupt file by returning None (the context_windows.py convention). The
registry is bookkeeping; the real outputs (.jsonl log, .patch) live in
~/.gerbil/sessions/ and the project's .gerbil/ exactly as always.
"""

import errno
import json
import os
import random
import shutil
import signal
import time
from pathlib import Path
from typing import Callable

RUNNING_DIR = Path.home() / ".gerbil" / "running"

# Bounded initial read of a display file on attach: enough scrollback to be
# useful, small enough to load instantly. The full file stays on disk.
INITIAL_DISPLAY_LIMIT = 2_000_000

_ADJECTIVES = (
    "able", "amber", "bold", "brave", "brisk", "calm", "cedar", "civic",
    "clear", "coral", "cosy", "crisp", "daring", "deft", "dusty", "eager",
    "early", "fabled", "fair", "fleet", "fond", "free", "gentle", "glad",
    "golden", "grand", "happy", "hardy", "humble", "jolly", "keen", "kind",
    "lively", "lucid", "lunar", "mellow", "merry", "noble", "polar", "proud",
    "quiet", "rapid", "rosy", "spry", "stout", "sunny", "swift", "witty",
)
_ANIMALS = (
    "badger", "bat", "beaver", "bison", "crane", "crow", "deer", "dove",
    "egret", "elk", "ermine", "ferret", "finch", "fox", "gecko", "hare",
    "heron", "ibex", "jay", "koala", "lemur", "lark", "lynx", "marmot",
    "marten", "mole", "moose", "newt", "otter", "owl", "panda", "pika",
    "quail", "raven", "robin", "seal", "shrew", "skink", "stoat", "swan",
    "tapir", "tern", "toad", "vole", "walrus", "weasel", "wren", "yak",
)


def new_run_name(taken: set[str] | None = None) -> str:
    """A fresh adjective-animal name, unique among the run dirs that exist
    (and any extra `taken` names). After enough collisions -- a very busy
    machine -- fall back to a numeric suffix, which cannot collide."""
    if taken is None:
        taken = set()
    try:
        taken = taken | {p.name for p in RUNNING_DIR.iterdir() if p.is_dir()}
    except OSError:
        pass
    for _ in range(64):
        name = f"{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}"
        if name not in taken:
            return name
    base = f"{random.choice(_ADJECTIVES)}-{random.choice(_ANIMALS)}"
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def run_dir(name: str) -> Path:
    return RUNNING_DIR / name


def create_run(name: str, *, project_dir: Path, model: str,
               theme: str | None, command: list[str]) -> Path:
    """Register a run about to be spawned. The pid is filled in by save_meta
    once the child exists; status "starting" covers the gap (a run that dies
    between create and spawn classifies as died, which is the truth)."""
    d = run_dir(name)
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "meta.json", {
        "name": name,
        "pid": None,
        "status": "starting",
        "project_dir": str(project_dir),
        "model": model,
        "theme": theme,
        "command": command,
        "started_at": time.time(),
    })
    (d / "display.ansi").touch()
    return d


def _write_json(path: Path, doc: dict) -> None:
    """Atomic-replace write; swallows I/O errors (registry bookkeeping must
    never take a session down). The tmp file sits in the same directory so the
    rename never crosses filesystems."""
    try:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc))
        os.replace(tmp, path)
    except OSError:
        pass


def _read_json(path: Path) -> dict | None:
    """Tolerant read: None on missing, unreadable, or corrupt -- the caller
    keeps its last-good value or treats the run as unknown."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def load_meta(name: str) -> dict | None:
    return _read_json(run_dir(name) / "meta.json")


def save_meta(name: str, **updates) -> None:
    """Read-modify-write of meta.json. Only the runner and its spawner write
    meta, and never concurrently for the same keys, so last-write-wins on the
    whole document is safe."""
    meta = load_meta(name) or {"name": name}
    meta.update(updates)
    _write_json(run_dir(name) / "meta.json", meta)


def load_stats_doc(name: str) -> dict | None:
    return _read_json(run_dir(name) / "stats.json")


def write_stats_doc(d: Path, doc: dict) -> None:
    _write_json(d / "stats.json", doc)


def pid_alive(pid: int | None) -> bool:
    """Whether a process with this pid exists. EPERM means it exists but is
    someone else's -- still 'alive' for liveness purposes (classify() lets a
    recorded terminal status override, which also covers pid reuse)."""
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


# The statuses a runner records on exit (via run_runner). Anything else in
# meta.json means the run never got to record its ending.
TERMINAL_STATUSES = ("complete", "interrupted", "error")


def classify(meta: dict) -> str:
    """The one liveness rule, shared by the viewer's poll and `gerbil ps`:
    a recorded terminal status is the truth (it survives pid reuse); otherwise
    a live pid is running -- or paused, when a viewer has SIGSTOPped it -- and
    a dead one means the runner died without getting to record how (SIGKILL,
    OOM, power loss)."""
    status = meta.get("status")
    if status in TERMINAL_STATUSES:
        return status
    if not pid_alive(meta.get("pid")):
        return "died"
    return "paused" if status == "paused" else "running"


def pause_run(name: str) -> bool:
    """Freeze a running runner with SIGSTOP. The whole process stops exactly
    where it is -- mid-turn, mid-tool-call -- while its sandbox container (a
    separate process tree under the container daemon) stays alive, so a later
    continue_run picks the session back up precisely where it left off. Nothing like
    `gerbil resume` happens here: no replay, no new process, no new log.

    The status write comes AFTER the stop lands: a stopped process cannot
    write, so there is no concurrent writer to race with. The one hole -- the
    runner recorded a terminal status in the instant before the stop landed --
    is checked for explicitly, and the near-corpse is released to finish
    exiting rather than being recorded as paused forever.

    Returns False (having done nothing lasting) unless the run was running."""
    meta = load_meta(name) or {}
    if classify(meta) != "running":
        return False
    try:
        os.kill(meta["pid"], signal.SIGSTOP)
    except (OSError, TypeError):
        return False
    if (load_meta(name) or {}).get("status") in TERMINAL_STATUSES:
        try:
            os.kill(meta["pid"], signal.SIGCONT)
        except OSError:
            pass
        return False
    save_meta(name, status="paused")
    return True


def continue_run(name: str) -> bool:
    """Continue a paused runner with SIGCONT. The status write comes BEFORE
    the wake-up -- while the runner is still frozen and cannot write meta
    concurrently (its exit path would otherwise race a terminal status
    against this "running"). Returns False unless the run was paused."""
    meta = load_meta(name) or {}
    if classify(meta) != "paused":
        return False
    save_meta(name, status="running")
    try:
        os.kill(meta["pid"], signal.SIGCONT)
    except (OSError, TypeError):
        return False
    return True


def list_runs() -> list[dict]:
    """Every registered run, oldest first: its meta plus `state` (classify)
    and `stats` (the stats.json document, None before the runner's first
    write). Unreadable entries are skipped, not fatal."""
    out = []
    try:
        entries = sorted(RUNNING_DIR.iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.is_dir():
            continue
        meta = load_meta(d.name)
        if meta is None:
            continue
        out.append({**meta, "state": classify(meta),
                    "stats": load_stats_doc(d.name)})
    out.sort(key=lambda m: m.get("started_at") or 0)
    return out


def remove_run(name: str) -> None:
    shutil.rmtree(run_dir(name), ignore_errors=True)


def tail_display(path: Path, offset: int, pending: bytes,
                 ) -> tuple[list[str], bytes, int]:
    """Incrementally read the display file: everything new past `offset`,
    returned as complete decoded lines. A trailing partial line is carried in
    `pending` as *bytes* until its newline arrives -- splitting before
    decoding is what makes a UTF-8 sequence torn across two reads safe
    (0x0A never occurs inside a multibyte sequence)."""
    try:
        with path.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
    except OSError:
        return [], pending, offset
    if not chunk:
        return [], pending, offset
    offset += len(chunk)
    buf = pending + chunk
    head, sep, pending = buf.rpartition(b"\n")
    if not sep:
        return [], pending, offset  # no complete line yet
    lines = head.decode("utf-8", errors="replace").split("\n")
    return lines, pending, offset


def initial_display(path: Path, limit: int = INITIAL_DISPLAY_LIMIT,
                    ) -> tuple[list[str], int, bool]:
    """The bounded first load on attach: at most the final `limit` bytes,
    aligned to line boundaries on both ends. Returns (complete lines, offset,
    truncated?). A trailing partial line is deliberately left *unread* --
    the offset stops after the last newline -- so the caller's next
    tail_display() picks it up whole, instead of it appearing twice."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            truncated = size > limit
            if truncated:
                f.seek(size - limit)
                f.readline()  # discard the partial line the cut landed in
            start = f.tell()
            data = f.read()
    except OSError:
        return [], 0, False
    head, sep, _partial = data.rpartition(b"\n")
    if not sep:
        return [], start, truncated  # no complete line yet
    offset = start + len(head) + 1
    return head.decode("utf-8", errors="replace").split("\n"), offset, truncated


def run_runner(name: str, body: Callable[[], None]) -> None:
    """The child-side wrapper around the whole command: record the pid, run,
    and -- however it ends -- record how, so the viewer and `gerbil ps`
    can tell a finished run from a killed one. sys.exit unwinds through here
    (SystemExit is a BaseException), so _abort's exit codes are readable:
    130 is its KeyboardInterrupt convention."""
    save_meta(name, pid=os.getpid(), status="running")
    status = "error"
    try:
        body()
        status = "complete"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    except SystemExit as exc:
        code = exc.code
        status = ("complete" if code in (0, None)
                  else "interrupted" if code == 130 else "error")
        raise
    finally:
        save_meta(name, status=status, finished_at=time.time())
