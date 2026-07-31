"""Tests for container-runtime selection (GERBIL_SANDBOX=docker|podman).

The podman client is a hand-written stand-in for the sliver of the Docker SDK
that LeanSandbox uses, so what matters is that it builds the right `podman`
command lines and reshapes the results the way the SDK does. Those subprocess
calls are intercepted here -- no Docker, no podman, no containers needed. If a
real podman IS installed, a final live phase boots an actual container and runs
the same operations against it end to end.

Run with: uv run python tests/test_runtime.py
"""

import io
import os
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

from gerbil import runtime


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"test failed at: {label}\n{detail}")


class FakeRun:
    """Stands in for subprocess.run inside runtime.py: records every argv it is
    handed and replies with a canned result."""

    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.calls: list[list[str]] = []
        self.inputs: list[bytes | None] = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, argv, **kwargs):
        self.calls.append(list(argv))
        self.inputs.append(kwargs.get("input"))
        text = kwargs.get("text", False)
        out, err = self.stdout, self.stderr
        if text:
            out, err = out.decode(), err.decode()
        return subprocess.CompletedProcess(argv, self.returncode, out, err)

    @property
    def argv(self) -> list[str]:
        return self.calls[-1]


def with_env(value):
    """Set (or clear) GERBIL_SANDBOX; returns the previous value."""
    prev = os.environ.get(runtime.ENV_VAR)
    if value is None:
        os.environ.pop(runtime.ENV_VAR, None)
    else:
        os.environ[runtime.ENV_VAR] = value
    return prev


def with_fake_run(fake):
    """Swap runtime.subprocess.run for `fake`; returns the original."""
    orig = runtime.subprocess.run
    runtime.subprocess.run = fake
    return orig


def test_selection() -> None:
    """GERBIL_SANDBOX picks the runtime; unset means docker; junk is an error."""
    prev = with_env(None)
    try:
        check("unset -> docker", runtime.runtime_name() == runtime.DOCKER)
        check("unset -> docker argv", runtime.cli_argv() == ["docker"])

        with_env("podman")
        check("podman selected", runtime.runtime_name() == runtime.PODMAN)
        # The podman CLI prefix must silence warnings: podman's own stderr is
        # indistinguishable from the container command's, so a chatty warning
        # would land in every tool result the model sees.
        check("podman argv is muzzled",
              runtime.cli_argv() == ["podman", "--log-level=error"],
              str(runtime.cli_argv()))

        with_env("docker")
        check("docker selected explicitly", runtime.runtime_name() == runtime.DOCKER)
        with_env("  podman  ")  # tolerate stray whitespace
        check("whitespace tolerated", runtime.runtime_name() == runtime.PODMAN)

        with_env("containerd")
        try:
            runtime.runtime_name()
            check("unknown runtime rejected", False, "no exception")
        except ValueError as exc:
            check("unknown runtime rejected", "containerd" in str(exc))
        problem = runtime.check_available()
        check("unknown runtime -> preflight message",
              problem is not None and "containerd" in problem, str(problem))

        with_env("")  # empty is treated as unset
        check("empty -> docker", runtime.runtime_name() == runtime.DOCKER)
    finally:
        with_env(prev)


def test_podman_run_argv() -> None:
    """containers.run must reproduce the SDK call LeanSandbox makes."""
    prev, fake = with_env("podman"), FakeRun(stdout=b"deadbeefcafe\n")
    orig = with_fake_run(fake)
    try:
        client = runtime.client()
        container = client.containers.run(
            "gerbil-lean-sandbox:latest",
            command="sleep infinity",
            detach=True,
            auto_remove=True,
            working_dir="/workspace/project",
            environment={"TZ": "America/New_York"},
        )
        check("run argv", fake.argv == [
            "podman", "--log-level=error", "run", "--detach", "--rm",
            "--workdir", "/workspace/project",
            "--env", "TZ=America/New_York",
            "gerbil-lean-sandbox:latest", "sleep", "infinity",
        ], str(fake.argv))
        check("run returns the container id", container.id == "deadbeefcafe")

        # environment=None (no host timezone) must not produce a stray --env.
        client.containers.run("img", command="sleep infinity", detach=True,
                              auto_remove=True, working_dir="/w", environment=None)
        check("no env -> no --env flag", "--env" not in fake.argv, str(fake.argv))

        # A failed `podman run` must raise, not hand back a bogus container.
        boom = FakeRun(returncode=125, stderr=b"Error: no such image")
        with_fake_run(boom)
        try:
            client.containers.run("img", command="sleep infinity", detach=True)
            check("failed run raises", False, "no exception")
        except RuntimeError as exc:
            check("failed run raises", "no such image" in str(exc), str(exc))
    finally:
        with_fake_run(orig)
        with_env(prev)


def test_podman_container_ops() -> None:
    """exec_run/put_archive/stop/kill/reload: argv + SDK-shaped results."""
    prev = with_env("podman")
    fake = FakeRun(stdout=b"out", stderr=b"err")
    orig = with_fake_run(fake)
    try:
        container = runtime._PodmanContainer("abc123")

        # The workhorse: sandbox.run()'s demuxed exec.
        code, (out, err) = container.exec_run(
            ["timeout", "60", "bash", "-c", "echo hi"],
            workdir="/workspace/project", demux=True,
        )
        check("exec argv", fake.argv == [
            "podman", "--log-level=error", "exec",
            "--workdir", "/workspace/project", "abc123",
            "timeout", "60", "bash", "-c", "echo hi",
        ], str(fake.argv))
        check("exec demux splits the streams", (out, err) == (b"out", b"err"))
        check("exec returns the exit code", code == 0)

        # The root chown in _upload_project (undemuxed, like the SDK default).
        code, output = container.exec_run(["chown", "-R", "1000:1000", "/w"], user="root")
        check("exec --user", "--user" in fake.argv and "root" in fake.argv, str(fake.argv))
        check("undemuxed output is combined", output == b"outerr", str(output))

        # A nonzero command exit is a result, not an exception (sandbox.run
        # reports exit codes to the model; 124 is its timeout signal).
        rc = FakeRun(returncode=124)
        with_fake_run(rc)
        code, _ = container.exec_run(["true"], demux=True)
        check("nonzero exit passed through", code == 124, str(code))

        with_fake_run(fake)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo("f.txt")
            info.size = 2
            tar.addfile(info, io.BytesIO(b"hi"))
        payload = buf.getvalue()
        container.put_archive("/tmp", payload)
        # Deliberately NOT `podman cp -`: podman's stdin copier dies with
        # "io: read/write on closed pipe" at certain payload sizes (see
        # put_archive), so uploads go through the container's own tar.
        check("put_archive unpacks with the container's tar", fake.argv == [
            "podman", "--log-level=error", "exec", "--interactive", "abc123",
            "tar", "-x", "-p", "--no-same-owner", "-f", "-", "-C", "/tmp",
        ], str(fake.argv))
        check("put_archive streams the tar on stdin", fake.inputs[-1] == payload)

        container.stop(timeout=5)
        check("stop argv", fake.argv == [
            "podman", "--log-level=error", "stop", "--time", "5", "abc123",
        ], str(fake.argv))
        container.kill()
        check("kill argv", fake.argv[-2:] == ["kill", "abc123"], str(fake.argv))

        running = FakeRun(stdout=b"running\n")
        with_fake_run(running)
        container.reload()
        check("reload argv", running.argv == [
            "podman", "--log-level=error", "inspect",
            "--format", "{{.State.Status}}", "abc123",
        ], str(running.argv))
        check("reload sets status", container.status == "running", container.status)

        # A vanished container must surface as an error, not a silent status.
        gone = FakeRun(returncode=125, stderr=b"Error: no such object")
        with_fake_run(gone)
        try:
            container.reload()
            check("reload raises when the container is gone", False, "no exception")
        except RuntimeError as exc:
            check("reload raises when the container is gone",
                  "no such object" in str(exc), str(exc))
    finally:
        with_fake_run(orig)
        with_env(prev)


def test_unmappable_user_hint() -> None:
    """When a container cannot start because rootless podman has no uid to map
    the image's non-root user onto, the failure must say how to fix it -- that
    is exactly the host where gerbil needs the root-inside-the-container image,
    and podman's own message ("crun: setresgid to `1000`: Invalid argument")
    explains nothing."""
    prev = with_env("podman")
    orig = runtime.subprocess.run

    def fake_info(uid_map: str):
        """subprocess.run that answers `podman info` per its --format."""
        def run(argv, **kwargs):
            fmt = argv[-1] if "--format" in argv else ""
            out = ""
            if "Rootless" in fmt:
                out = "true\n"
            elif "IDMappings" in fmt:
                out = uid_map
            elif "run" in argv:
                return subprocess.CompletedProcess(
                    argv, 126, "", "Error: crun: setresgid to `1000`: Invalid argument"
                )
            if not kwargs.get("text", False):
                out = out.encode()
            return subprocess.CompletedProcess(argv, 0, out, "" if kwargs.get("text") else b"")
        return run

    try:
        # Single mapped uid (no /etc/subuid range): the hint must appear.
        runtime.subprocess.run = fake_info("0:1 ")
        try:
            runtime.client().containers.run("img", command="sleep infinity", detach=True)
            check("unmappable user: raises", False, "no exception")
        except RuntimeError as exc:
            check("unmappable user: raises", True)
            check("unmappable user: keeps podman's own message",
                  "setresgid" in str(exc), str(exc))
            check("unmappable user: points at the root-mode image",
                  "SANDBOX_UID=0" in str(exc), str(exc))

        # A normal subuid range covers uid 1000 -- some other failure, no hint.
        runtime.subprocess.run = fake_info("0:65536 ")
        try:
            runtime.client().containers.run("img", command="sleep infinity", detach=True)
        except RuntimeError as exc:
            check("mappable user: no misleading hint",
                  "SANDBOX_UID=0" not in str(exc), str(exc))
    finally:
        runtime.subprocess.run = orig
        with_env(prev)


def test_live_podman() -> None:
    """If podman is actually installed, drive a real container through the same
    operations LeanSandbox performs. Skipped when podman is absent or unusable
    (e.g. no rootless user-namespace setup)."""
    prev = with_env("podman")
    try:
        if shutil.which("podman") is None:
            print("[SKIP] live podman: not installed")
            return
        problem = runtime.check_available()
        if problem:
            print("[SKIP] live podman: not usable --\n" + problem)
            return
        image = os.environ.get("GERBIL_TEST_IMAGE", "docker.io/library/alpine:latest")
        if subprocess.run([*runtime.cli_argv(), "image", "inspect", image],
                          capture_output=True).returncode != 0:
            print(f"[SKIP] live podman: image {image} not present locally")
            return

        container = runtime.client().containers.run(
            image, command="sleep infinity", detach=True, auto_remove=True,
            working_dir="/tmp", environment={"TZ": "America/New_York"},
        )
        try:
            container.reload()
            check("live: container is running", container.status == "running",
                  container.status)

            code, (out, err) = container.exec_run(
                ["sh", "-c", "echo out; echo err >&2; exit 3"], demux=True
            )
            check("live: exit code", code == 3, str(code))
            check("live: stdout captured", out == b"out\n", str(out))
            check("live: stderr captured -- and no podman warnings in it",
                  err == b"err\n", str(err))

            code, (out, _) = container.exec_run(["sh", "-c", "echo $TZ"], demux=True)
            check("live: environment reached the container",
                  out == b"America/New_York\n", str(out))

            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tar:
                data = b"hello\n"
                info = tarfile.TarInfo("gerbil-probe.txt")
                info.size = len(data)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(data))
            container.put_archive("/tmp", buf.getvalue())
            code, (out, _) = container.exec_run(
                ["cat", "/tmp/gerbil-probe.txt"], demux=True
            )
            check("live: put_archive landed the file", out == b"hello\n", str(out))

            # Regression: `podman cp -` rejects payloads of certain exact sizes
            # ("io: read/write on closed pipe" -- 40, 70, 100, 130 KiB ... on
            # podman 5.8), and a real project upload lands right in that range.
            # The tar-in-the-container path must not care.
            for kib in (40, 70, 100, 130):
                body = kib * 1024 - 10240 - 512  # -> a tar of exactly kib KiB
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w") as tar:
                    info = tarfile.TarInfo(f"gerbil-probe-{kib}.bin")
                    info.size = body
                    info.mode = 0o644
                    tar.addfile(info, io.BytesIO(b"x" * body))
                payload = buf.getvalue()
                check(f"live: {kib} KiB upload is exactly that size",
                      len(payload) == kib * 1024, str(len(payload)))
                container.put_archive("/tmp", payload)
                code, (out, _) = container.exec_run(
                    ["stat", "-c", "%s", f"/tmp/gerbil-probe-{kib}.bin"], demux=True
                )
                check(f"live: {kib} KiB upload landed intact",
                      out.strip() == str(body).encode(), str(out))

            code, (out, _) = container.exec_run(["pwd"], workdir="/", demux=True)
            check("live: workdir honored", out == b"/\n", str(out))
        finally:
            container.stop(timeout=1)
        # --rm means stopping also removes it, so inspect must now fail.
        gone = subprocess.run(
            [*runtime.cli_argv(), "inspect", container.id], capture_output=True
        )
        check("live: stop removed the container", gone.returncode != 0)
    finally:
        with_env(prev)


def main() -> None:
    test_selection()
    test_podman_run_argv()
    test_podman_container_ops()
    test_unmappable_user_hint()
    test_live_podman()
    print("\nAll runtime tests passed.")


if __name__ == "__main__":
    main()
