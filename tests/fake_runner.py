"""A fake background run for exercising the viewer without Docker or a model.

Registers a run named `fake-run` (override with argv[1]), then plays the part
of a runner: appends styled lines to display.ansi and rewrites stats.json a
few times a second for ~30 seconds (argv[2] overrides), then records status
"complete" and exits. While it runs (from the repo root):

    uv run python tests/fake_runner.py &
    uv run python -m gerbil running
    uv run python -m gerbil grab fake-run

Try: scrollback while lines stream, `d` (detach + reattach), Ctrl-C (it
really does SIGINT this process -- the wrapper records "interrupted"), and
the held finished screen at the end. The viewer's confirm-exit removes the
run dir; if it's left behind, `gerbil running` prunes it.
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.environ["GERBIL_FORCE_STYLE"] = "1"  # before render's import-time check

from gerbil import render, runs  # noqa: E402
from gerbil.providers import Usage  # noqa: E402
from gerbil.view import SessionStats, stats_to_wire  # noqa: E402


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "fake-run"
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

    rd = runs.create_run(name, project_dir=Path.cwd(), model="fake-model-1",
                         theme=None, command=["fake_runner"])
    display = (rd / "display.ansi").open("a", encoding="utf-8")

    def emit(text: str) -> None:
        display.write(text + "\n")
        display.flush()

    stats = SessionStats(run_name=name, model="fake-model-1")
    tail: list[str] = []

    def body() -> None:
        stats.on_session_begin(name="fake-session", model="fake-model-1",
                               small_model=None, ralph=None, resumed_from=None,
                               now=time.monotonic())
        emit(render.style("[fake runner: booting an imaginary sandbox]", "gray"))
        deadline = time.monotonic() + seconds
        turn = 0
        while time.monotonic() < deadline:
            turn += 1
            emit(render.style(f"── turn {turn} · {time.strftime('%H:%M:%S')}",
                              "bold", "dark_red"))
            emit(f"thinking very hard about lemma {turn}...")
            emit(render.style("  -> ", "cyan") + render.style("bash", "bold", "cyan")
                 + f"(sleep + echo {turn})")
            stats.on_turn_header(100_000)
            stats.on_turn_complete(
                Usage(input_tokens=1000 * turn, output_tokens=50 * turn),
                zoom=False)
            stats.files = {"Fake/Lemmas.lean": (3 * turn, turn)}
            runs.write_stats_doc(rd, {"v": 1, "written_at": time.time(),
                                      "stats": stats_to_wire(stats),
                                      "tail": tail})
            time.sleep(0.7)
        line = f"session: (fake) {name} ran {turn} turns"
        tail.append(line)
        emit(line)
        runs.write_stats_doc(rd, {"v": 1, "written_at": time.time(),
                                  "stats": stats_to_wire(stats), "tail": tail})

    try:
        runs.run_runner(name, body)
    finally:
        display.close()


if __name__ == "__main__":
    main()
