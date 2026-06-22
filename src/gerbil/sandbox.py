import io
import posixpath
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path

import docker

WORKSPACE_DIR = "/workspace/project"

# Must match the uid/gid of the user created in the Dockerfile, so files we
# upload land owned by that user and git operations don't hit ownership errors.
SANDBOX_UID = 1000
SANDBOX_GID = 1000


@dataclass
class CommandResult:
    """Result of running a command in the sandbox."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    timeout_occurred: bool


class LeanSandbox:
    """Sandboxed Lean environment running inside a Docker container.

    Isolation is provided entirely by Docker. We talk to the container directly
    via the Docker SDK: exec_run for commands, put_archive/cat for file I/O.

    At startup:
      - Uploads all git-tracked files from project_dir into the container.
      - Makes an initial git commit so get_diff() can produce a clean patch.
      - Runs lake exe cache get to fetch precompiled mathlib oleans.

    Usage:
        with LeanSandbox(project_dir="/path/to/lean-project") as sandbox:
            sandbox.write_file("MyProof.lean", content)
            result = sandbox.lake_build()
            diff = sandbox.get_diff()
    """

    def __init__(
        self,
        project_dir: str | Path,
        image: str = "lean-sandbox:latest",
        fetch_cache: bool = True,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.image = image
        self.fetch_cache = fetch_cache
        self._docker = docker.from_env()
        self._container = None

    def __enter__(self) -> "LeanSandbox":
        self._container = self._docker.containers.run(
            self.image,
            command="sleep infinity",
            detach=True,
            auto_remove=True,
            working_dir=WORKSPACE_DIR,
        )
        self._wait_running()
        self._upload_project()
        self._init_git()
        if self.fetch_cache:
            self._fetch_mathlib_cache()
        return self

    @property
    def container_id(self) -> str:
        """The running container's id (used to `docker exec` into the sandbox)."""
        if self._container is None:
            raise RuntimeError("sandbox is not running")
        return self._container.id

    def __exit__(self, *_) -> None:
        if self._container:
            try:
                self._container.stop(timeout=5)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Startup helpers
    # ------------------------------------------------------------------

    def _wait_running(self, retries: int = 30, delay: float = 0.5) -> None:
        for _ in range(retries):
            self._container.reload()
            if self._container.status == "running":
                return
            time.sleep(delay)
        raise TimeoutError("sandbox container did not reach running state")

    def _upload_project(self) -> None:
        """Tar all git-tracked files from the host project and extract them into
        the container workspace in one shot."""
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.project_dir,
            capture_output=True,
            check=True,
        )
        rel_paths = [p for p in result.stdout.decode().split("\0") if p]

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel in rel_paths:
                local = self.project_dir / rel
                if local.is_file():
                    tar.add(local, arcname=rel, filter=_own_by_sandbox)
        buf.seek(0)
        self._container.put_archive(WORKSPACE_DIR, buf.getvalue())

    def _init_git(self) -> None:
        """Initialize a git repo in the container and commit the uploaded files,
        establishing a clean baseline for get_diff()."""
        self.run("git init -b main")
        self.run("git config user.email gerbil@local")
        self.run("git config user.name gerbil")
        self.run("git add -A")
        self.run("git commit -m initial")

    def _fetch_mathlib_cache(self) -> None:
        """Download precompiled mathlib oleans. Runs once per session."""
        print("Fetching mathlib cache...")
        result = self.run("lake exe cache get", timeout=600.0)
        if result.exit_code != 0:
            raise RuntimeError(f"lake exe cache get failed:\n{result.stderr}")

    # ------------------------------------------------------------------
    # Agent-facing API
    # ------------------------------------------------------------------

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox. Path is relative to the project dir."""
        result = self.run(f"cat {_quote(path)}")
        if result.exit_code != 0:
            raise FileNotFoundError(path)
        return result.stdout

    def write_file(self, path: str, content: str) -> None:
        """Write (or overwrite) a file in the sandbox. Path is relative to the
        project dir; parent directories are created as needed."""
        abs_path = posixpath.join(WORKSPACE_DIR, path)
        parent = posixpath.dirname(abs_path)
        self.run(f"mkdir -p {_quote(parent)}")

        data = content.encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=posixpath.basename(path))
            info.size = len(data)
            info.mode = 0o644
            info.uid = SANDBOX_UID
            info.gid = SANDBOX_GID
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        self._container.put_archive(parent, buf.getvalue())

    def lake_build(self, timeout: float = 120.0) -> CommandResult:
        """Run lake build and return stdout, stderr, and exit_code."""
        return self.run("lake build", timeout=timeout)

    def run(self, command: str, timeout: float = 60.0) -> CommandResult:
        """Run a shell command in the sandbox workspace directory."""
        wrapped = ["timeout", str(int(timeout)), "bash", "-c", command]
        exit_code, (stdout, stderr) = self._container.exec_run(
            wrapped, workdir=WORKSPACE_DIR, demux=True
        )
        return CommandResult(
            command=command,
            exit_code=exit_code,
            stdout=(stdout or b"").decode(errors="replace"),
            stderr=(stderr or b"").decode(errors="replace"),
            timeout_occurred=exit_code == 124,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_diff(self) -> str:
        """Return a git patch of changes since the last commit (the session baseline)."""
        self.run("git add -A")
        return self.run("git diff --cached").stdout

    def commit(self, message: str) -> bool:
        """Commit all current changes inside the container, advancing the baseline
        for the next get_diff(). Returns False if there was nothing to commit.

        Used by --ralph to chain sessions as a series of commits.
        """
        self.run("git add -A")
        if self.run("git diff --cached --quiet").exit_code == 0:
            return False
        # Pass the message on stdin via a quoted heredoc so its content is taken
        # literally (no shell expansion). `command` reaches bash -c verbatim.
        script = f"git commit -F - <<'GERBIL_MSG'\n{message}\nGERBIL_MSG\n"
        result = self.run(script)
        if result.exit_code != 0:
            raise RuntimeError(f"git commit failed:\n{result.stderr}")
        return True


def _own_by_sandbox(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = SANDBOX_UID
    info.gid = SANDBOX_GID
    info.uname = ""
    info.gname = ""
    return info


def _quote(s: str) -> str:
    """Single-quote a string for safe use in a bash command."""
    return "'" + s.replace("'", "'\\''") + "'"
