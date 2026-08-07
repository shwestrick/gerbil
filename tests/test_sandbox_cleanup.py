"""Tests for sandbox container cleanup on interrupt.

The container must never outlive gerbil. The leak windows: a Ctrl-C during
__enter__ (the slow startup cache fetch) lands before the `with` body is
entered, so __exit__ would never run; and a second Ctrl-C during __exit__'s
graceful 5s stop would abort teardown. Both are driven here with a fake
container client -- no Docker/podman needed.
Run with: uv run python tests/test_sandbox_cleanup.py
"""

from types import SimpleNamespace

import gerbil.sandbox as sandbox_mod
from gerbil.sandbox import LeanSandbox


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


class FakeContainer:
    def __init__(self, stop_raises=None):
        self.stop_calls = 0
        self.killed = False
        self.status = "running"
        self.id = "fake-container"
        self._stop_raises = stop_raises

    def reload(self):
        pass

    def exec_run(self, cmd, demux=False, **kwargs):
        """Enough of exec_run for the boot-time image check: report a conforming
        image, so __enter__ reaches the part these tests are actually about."""
        if kwargs.get("user") == "root":
            return 0, b"0\n"
        out = f"uid:{sandbox_mod.SANDBOX_UID}\n".encode()
        return (0, (out, b"")) if demux else (0, out)

    def stop(self, timeout=None):
        self.stop_calls += 1
        if self._stop_raises:
            raise self._stop_raises

    def kill(self):
        self.killed = True


def make_sandbox(container):
    """A LeanSandbox wired to a fake container client (restores runtime.client)."""
    orig = sandbox_mod.runtime.client
    sandbox_mod.runtime.client = lambda: SimpleNamespace(
        containers=SimpleNamespace(run=lambda *a, **k: container)
    )
    try:
        return LeanSandbox(project_dir=".", fetch_cache=False)
    finally:
        sandbox_mod.runtime.client = orig


def test_interrupt_during_enter_stops_container() -> None:
    """Ctrl-C during sandbox boot (before the `with` body is entered) must
    still stop the container -- __exit__ would otherwise never run."""
    fake = FakeContainer()
    sb = make_sandbox(fake)
    sb._upload_project = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
    try:
        with sb:
            check("enter interrupt: with-body must not run", False)
    except KeyboardInterrupt:
        pass
    check("enter interrupt: container stopped", fake.stop_calls == 1,
          str(fake.stop_calls))
    check("enter interrupt: sandbox forgets the container",
          sb._container is None)

    # A plain crash during boot cleans up the same way.
    fake2 = FakeContainer()
    sb2 = make_sandbox(fake2)
    sb2._upload_project = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    try:
        with sb2:
            pass
    except RuntimeError:
        pass
    check("enter crash: container stopped", fake2.stop_calls == 1)


def test_second_interrupt_during_exit_kills() -> None:
    """A second Ctrl-C during the graceful 5s stop falls back to an immediate
    kill (auto_remove then deletes the container) and re-raises."""
    fake = FakeContainer(stop_raises=KeyboardInterrupt())
    sb = make_sandbox(fake)
    sb._container = fake
    try:
        sb.__exit__()
        check("exit interrupt: re-raises", False, "no exception")
    except KeyboardInterrupt:
        check("exit interrupt: re-raises", True)
    check("exit interrupt: container killed", fake.killed is True)

    # An ordinary stop() failure stays swallowed, as before.
    fake2 = FakeContainer(stop_raises=RuntimeError("already gone"))
    sb2 = make_sandbox(fake2)
    sb2._container = fake2
    sb2.__exit__()
    check("exit error: swallowed", fake2.killed is False)


def test_exit_idempotent() -> None:
    """__exit__ runs from both __enter__'s failure path and the with-statement;
    the second call must be a no-op."""
    fake = FakeContainer()
    sb = make_sandbox(fake)
    sb._container = fake
    sb.__exit__()
    sb.__exit__()
    check("exit idempotent: stopped exactly once", fake.stop_calls == 1,
          str(fake.stop_calls))


def main() -> None:
    test_interrupt_during_enter_stops_container()
    test_second_interrupt_during_exit_kills()
    test_exit_idempotent()
    print("\nAll sandbox cleanup tests passed.")


if __name__ == "__main__":
    main()
