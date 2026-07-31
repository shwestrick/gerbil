"""Container-runtime selection: Docker (the default) or Podman.

gerbil runs every session inside a container. Which engine provides that
container is chosen by the GERBIL_SANDBOX environment variable:

    GERBIL_SANDBOX=docker   (or unset)  -- the Docker daemon, via the Docker SDK
    GERBIL_SANDBOX=podman               -- podman, via its CLI

Docker keeps its original path: the `docker` Python SDK talking to the daemon
socket. Podman is driven through the `podman` command instead of the SDK, even
though podman ships a Docker-compatible REST API: that API only exists while
`podman system service` (the podman.socket unit) is running, and it is disabled
by default on most installs. Podman's whole selling point here is that it needs
no daemon -- requiring the user to start one to use it would defeat the purpose.

To keep sandbox.py free of engine branching, PodmanClient below reimplements
exactly the sliver of the Docker SDK surface that LeanSandbox uses -- and
nothing more -- over `podman` subprocesses. See _PodmanContainer for the
container-side methods and the (few) places where podman's behavior differs.
"""

import os
import shlex
import shutil
import subprocess

ENV_VAR = "GERBIL_SANDBOX"
DOCKER = "docker"
PODMAN = "podman"
RUNTIMES = (DOCKER, PODMAN)


def runtime_name() -> str:
    """The selected container runtime: "docker" (default) or "podman".

    Raises ValueError on an unrecognized GERBIL_SANDBOX value rather than
    silently falling back to Docker -- a typo'd runtime name should be loud."""
    name = os.environ.get(ENV_VAR, "").strip() or DOCKER
    if name not in RUNTIMES:
        raise ValueError(
            f"{ENV_VAR}={name!r} is not a supported container runtime "
            f"(expected one of: {', '.join(RUNTIMES)})"
        )
    return name


def cli_argv() -> list[str]:
    """The command prefix for talking to the selected runtime on the command
    line: e.g. ["docker"] or ["podman", "--log-level=error"].

    Used by mcp_client to `<runtime> exec` into the sandbox, and internally by
    the podman client below.

    Why podman gets --log-level=error: podman writes its own diagnostics to
    stderr, and `podman exec` also forwards the container process's stderr
    there, so the two are indistinguishable to the caller. Podman is chatty at
    the default warning level (e.g. "Network file system detected as backing
    store" on every single invocation, when the graph root lives on NFS), and
    that noise would otherwise be spliced into the stderr of every tool result
    the model sees -- and into the MCP server's stdio stream. Errors still come
    through."""
    name = runtime_name()
    return [PODMAN, "--log-level=error"] if name == PODMAN else [DOCKER]


def client():
    """A container client for the selected runtime, exposing the subset of the
    Docker SDK API that LeanSandbox uses (`.containers.run(...)` returning a
    container object). Docker gets the real SDK; podman gets PodmanClient."""
    if runtime_name() == PODMAN:
        return PodmanClient()
    import docker

    return docker.from_env()


def check_available() -> str | None:
    """Probe the selected runtime and return an error message describing why it
    is unusable, or None if it is fine. Used by the CLI preflight (which turns
    the message into a clean exit) instead of letting the first container
    operation fail with an SDK traceback."""
    try:
        name = runtime_name()
    except ValueError as exc:
        return f"error: {exc}"
    return _check_podman() if name == PODMAN else _check_docker()


_DOCKER_PERMISSION_HELP = """\
error: cannot connect to Docker -- permission denied.

Docker must be usable without sudo (gerbil talks to the daemon via the Docker
SDK, which cannot use sudo). To fix:
  - add yourself to the docker group:  sudo usermod -aG docker $USER
    then log out and back in (or run: newgrp docker), and verify with:
    docker run hello-world
  - or set up rootless Docker: https://docs.docker.com/engine/security/rootless/
  - or, if Docker is not available at all, use podman:  export GERBIL_SANDBOX=podman"""

_DOCKER_DAEMON_HELP = """\
error: cannot connect to the Docker daemon -- is it running?
Start it (e.g. `sudo systemctl start docker`) or open Docker Desktop, then \
retry. If Docker is not available at all, use podman instead:
  export GERBIL_SANDBOX=podman"""


def _check_docker() -> str | None:
    import docker

    try:
        docker.from_env().ping()
        return None
    except Exception as exc:
        detail = str(exc)
        msg = detail.lower()

    if "permission denied" in msg:
        return _DOCKER_PERMISSION_HELP
    if "connect" in msg or "daemon" in msg:
        return _DOCKER_DAEMON_HELP
    return f"error: Docker is not usable: {detail}"


_ROOTUSER_IMAGE_HINT = """
The sandbox image runs as a non-root user, but rootless podman here has only
uid 0 mapped -- this account has no /etc/subuid range -- so that user cannot
exist inside the container. Rebuild the image to run as root *inside* the
container instead; rootless podman maps that back to your own unprivileged user
on the host, so nothing gains privilege:

  podman build --build-arg SANDBOX_UID=0 -t <image> src/lean-sandbox

(The `gerbil` launcher detects this and builds the right image by itself; this
note is for images built by hand.)"""


def _unmappable_sandbox_user_hint() -> str:
    """Extra guidance to append when a container fails to start because the
    image's non-root user has no uid mapping. Returns "" whenever the runtime
    could map it, so it can be appended unconditionally on the failure path."""
    if runtime_name() != PODMAN:
        return ""
    rootless = subprocess.run(
        [*cli_argv(), "info", "--format", "{{.Host.Security.Rootless}}"],
        capture_output=True, text=True,
    )
    if rootless.stdout.strip() != "true":
        return ""  # rootful podman maps every uid
    proc = subprocess.run(
        [*cli_argv(), "info", "--format",
         "{{range .Host.IDMappings.UIDMap}}{{.ContainerID}}:{{.Size}} {{end}}"],
        capture_output=True, text=True,
    )
    for entry in proc.stdout.split():
        first, _, size = entry.partition(":")
        try:
            if int(first) <= 1000 < int(first) + int(size):
                return ""  # the usual sandbox uid is mappable; not this problem
        except ValueError:
            return ""  # unexpected format -- don't guess
    return _ROOTUSER_IMAGE_HINT


def _check_podman() -> str | None:
    """Podman has no daemon to ping, so `podman info` is the equivalent check:
    it exercises the local storage configuration and the user namespace setup,
    which is where rootless podman actually fails."""
    if shutil.which(PODMAN) is None:
        return (
            f"error: podman is not installed, but {ENV_VAR}=podman is set.\n"
            "Install it (https://podman.io/docs/installation), or unset "
            f"{ENV_VAR} to use Docker."
        )
    proc = subprocess.run(
        [*cli_argv(), "info"], capture_output=True, text=True
    )
    if proc.returncode == 0:
        return None
    return "error: podman is not usable:\n" + (proc.stderr.strip() or proc.stdout.strip())


# ----------------------------------------------------------------------
# Podman: the Docker-SDK-shaped façade over the podman CLI
# ----------------------------------------------------------------------


class PodmanClient:
    """Stand-in for `docker.from_env()`, implementing only what LeanSandbox
    calls: `client.containers.run(...)`."""

    def __init__(self) -> None:
        self.containers = _PodmanContainers()


class _PodmanContainers:
    """Stand-in for the SDK's `client.containers` collection."""

    def run(
        self,
        image: str,
        command: str | list[str] | None = None,
        detach: bool = False,
        auto_remove: bool = False,
        working_dir: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> "_PodmanContainer":
        """Start a container, mirroring `client.containers.run(...)`.

        Only the detached form is supported -- that is all gerbil uses (the
        sandbox runs `sleep infinity` and is driven by exec) and it is the only
        form where returning a container handle, rather than the container's
        output, is the right result."""
        if not detach:
            raise NotImplementedError(
                "the podman runtime only supports detached containers"
            )
        argv = [*cli_argv(), "run", "--detach"]
        if auto_remove:
            argv.append("--rm")
        if working_dir:
            argv += ["--workdir", working_dir]
        for key, value in (environment or {}).items():
            argv += ["--env", f"{key}={value}"]
        argv.append(image)
        # The SDK accepts a command string and splits it; do the same so callers
        # can pass either form.
        argv += shlex.split(command) if isinstance(command, str) else list(command or [])

        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"podman run failed:\n"
                f"{proc.stderr.strip() or proc.stdout.strip()}"
                f"{_unmappable_sandbox_user_hint()}"
            )
        return _PodmanContainer(proc.stdout.strip())


class _PodmanContainer:
    """Stand-in for the SDK's Container object, over `podman` subprocesses.

    Implements exactly the methods LeanSandbox uses: id, status/reload,
    put_archive, exec_run, stop, kill."""

    def __init__(self, container_id: str) -> None:
        self.id = container_id
        self.status = "created"

    def _podman(self, *args: str, stdin: bytes | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*cli_argv(), *args], input=stdin, capture_output=True
        )

    def reload(self) -> None:
        """Refresh `status` from the engine (SDK parity: the SDK's reload()
        re-reads the container's state into the object)."""
        proc = self._podman("inspect", "--format", "{{.State.Status}}", self.id)
        if proc.returncode != 0:
            raise RuntimeError(
                f"podman inspect failed for container {self.id[:12]}:\n"
                f"{proc.stderr.decode(errors='replace').strip()}"
            )
        self.status = proc.stdout.decode(errors="replace").strip()

    # Extraction flags for the in-container tar (see put_archive):
    #   -p              apply the archived modes exactly (no umask), as the
    #                   Docker SDK's put_archive does
    #   --no-same-owner create everything owned by the container user we exec
    #                   as, instead of restoring the archived uid/gid
    # Ownership deserves a word. The SDK restores the tar entries' uid/gid;
    # gerbil stamps SANDBOX_UID/SANDBOX_GID on them, which is exactly the
    # sandbox image's USER -- so extracting as that user gives the same result
    # without a chown. Restoring them literally instead would need the uid
    # mapped into the (rootless) user namespace, which fails outright on hosts
    # with no /etc/subuid range -- a very common way to run podman.
    _TAR_FLAGS = ("-x", "-p", "--no-same-owner", "-f", "-", "-C")

    def put_archive(self, path: str, data: bytes) -> bool:
        """Extract the tar stream `data` into `path` inside the container: the
        SDK's put_archive.

        The obvious spelling, `podman cp - <id>:<path>`, is NOT used: podman's
        stdin copier is unreliable, failing with "io: read/write on closed
        pipe" for particular payload sizes (deterministically at 40, 70, 100,
        130 KiB ... with podman 5.8) -- and gerbil's uploads land squarely in
        that range. Piping the archive into the container's own `tar` avoids
        podman's copier entirely, and streams through `podman exec` the same
        way every other command does."""
        proc = self._podman(
            "exec", "--interactive", self.id, "tar", *self._TAR_FLAGS, path,
            stdin=data,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"unpacking an upload into {path} failed:\n"
                f"{proc.stderr.decode(errors='replace').strip()}"
            )
        return True

    def exec_run(
        self,
        cmd: str | list[str],
        workdir: str | None = None,
        user: str | None = None,
        environment: dict[str, str] | None = None,
        demux: bool = False,
    ) -> tuple[int, tuple[bytes | None, bytes | None] | bytes]:
        """Run a command in the container, mirroring the SDK's exec_run: returns
        (exit_code, output), where output is a (stdout, stderr) pair when
        demux=True and the two streams combined otherwise.

        The exit code is the command's own, except when podman itself fails to
        start it -- then it is podman's (125 could-not-exec / 126 not
        executable / 127 not found, the same convention as `docker exec`) and
        podman's explanation lands on stderr."""
        argv = ["exec"]
        if workdir:
            argv += ["--workdir", workdir]
        if user:
            argv += ["--user", user]
        for key, value in (environment or {}).items():
            argv += ["--env", f"{key}={value}"]
        argv.append(self.id)
        argv += shlex.split(cmd) if isinstance(cmd, str) else list(cmd)

        proc = self._podman(*argv)
        if demux:
            return proc.returncode, (proc.stdout or None, proc.stderr or None)
        return proc.returncode, (proc.stdout or b"") + (proc.stderr or b"")

    def stop(self, timeout: int | None = None) -> None:
        """Stop the container, giving it `timeout` seconds to exit before it is
        killed (SDK parity). With --rm at run time, stopping also removes it."""
        args = ["stop"]
        if timeout is not None:
            args += ["--time", str(int(timeout))]
        proc = self._podman(*args, self.id)
        if proc.returncode != 0:
            raise RuntimeError(
                f"podman stop failed for container {self.id[:12]}:\n"
                f"{proc.stderr.decode(errors='replace').strip()}"
            )

    def kill(self) -> None:
        """SIGKILL the container immediately (the impatient-second-Ctrl-C path
        in LeanSandbox.__exit__)."""
        proc = self._podman("kill", self.id)
        if proc.returncode != 0:
            raise RuntimeError(
                f"podman kill failed for container {self.id[:12]}:\n"
                f"{proc.stderr.decode(errors='replace').strip()}"
            )
