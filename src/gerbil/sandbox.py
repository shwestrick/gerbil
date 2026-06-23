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

    The Lake project need not be the git repo root: we upload the whole repo
    (rooted at repo_root) into WORKSPACE_DIR, and operate on the Lake project at
    WORKSPACE_DIR/<subdir>, where subdir is project_dir relative to repo_root.

    At startup:
      - Uploads all git-tracked files from repo_root into the container, plus the
        .git directory (full history, for format-patch and past-commit lookups).
      - Configures a committer identity so gerbil can commit the agent's work.
      - Runs lake exe cache get to fetch precompiled mathlib oleans.

    Usage:
        with LeanSandbox(project_dir="/repo/sub/lean-project") as sandbox:
            sandbox.write_file("MyProof.lean", content)
            result = sandbox.lake_build()
            diff = sandbox.get_diff()
    """

    def __init__(
        self,
        project_dir: str | Path,
        image: str = "lean-sandbox:latest",
        fetch_cache: bool = True,
        repo_root: str | Path | None = None,
    ):
        self.project_dir = Path(project_dir).resolve()
        self.repo_root = Path(repo_root).resolve() if repo_root else self.project_dir
        # The Lake project's path relative to the repo root ("" when they coincide).
        rel = self.project_dir.relative_to(self.repo_root).as_posix()
        self._subdir = "" if rel == "." else rel
        self.image = image
        self.fetch_cache = fetch_cache
        self._docker = docker.from_env()
        self._container = None

    @property
    def project_path(self) -> str:
        """Container path of the Lake project root: WORKSPACE_DIR (the repo root)
        or a subdirectory of it. All Lake/agent/MCP operations run here; git
        commands work too, since git resolves .git up the tree."""
        return posixpath.join(WORKSPACE_DIR, self._subdir) if self._subdir else WORKSPACE_DIR

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
        self._configure_git()
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
        """Upload the real repository into the container: the .git directory (full
        history, so the agent can refer to past commits) plus the tracked files.
        Rooted at repo_root, which may be an ancestor of the Lake project. The
        working tree is required to be clean (see the CLI preflight), so the
        tracked files match HEAD and no untracked files are uploaded -- the agent
        commits on top of a clean, known baseline."""
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.repo_root,
            capture_output=True,
            check=True,
        ).stdout
        rels = [p for p in out.decode().split("\0") if p]

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel in rels:
                local = self.repo_root / rel
                if local.is_file():
                    tar.add(local, arcname=rel, filter=_own_by_sandbox)
            gitdir = self.repo_root / ".git"
            if gitdir.is_dir():
                tar.add(gitdir, arcname=".git", filter=_own_by_sandbox)
        buf.seek(0)
        self._container.put_archive(WORKSPACE_DIR, buf.getvalue())

        # Leading directories that put_archive creates for nested entries (e.g.
        # the Lake project's subdir, when it isn't the repo root) land owned by
        # root, so the sandbox user can't write into them (lake creates .lake/
        # there). Reassert ownership over the whole workspace, as root.
        self._container.exec_run(
            ["chown", "-R", f"{SANDBOX_UID}:{SANDBOX_GID}", WORKSPACE_DIR],
            user="root",
        )

    def _configure_git(self) -> None:
        """Set a local committer identity so gerbil can commit the agent's work.
        The uploaded repo keeps its real history; we do not re-init."""
        self.run("git config user.email gerbil@local")
        self.run("git config user.name gerbil")
        # Uploaded .git is owned by the sandbox user (uid 1000 == container user),
        # but add safe.directory defensively in case of ownership quirks.
        self.run(f"git config --global --add safe.directory {WORKSPACE_DIR}")

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
        abs_path = posixpath.join(self.project_path, path)
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
            wrapped, workdir=self.project_path, demux=True
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

    def head(self) -> str:
        """The current HEAD commit hash. Raises if the repo has no commits
        (rev-parse otherwise echoes a bogus 'HEAD' on an unborn branch)."""
        result = self.run("git rev-parse --verify HEAD")
        if result.exit_code != 0:
            raise RuntimeError("repository has no commits (no HEAD)")
        return result.stdout.strip()

    def get_diff(self) -> str:
        """Return a git diff of the uncommitted working-tree changes (vs HEAD)."""
        self.run("git add -A")
        return self.run("git diff --cached").stdout

    def commit(self, message: str) -> bool:
        """Commit all current changes inside the container on top of real HEAD.
        Returns False if there was nothing to commit. Skips hooks (--no-verify),
        since host hooks may assume tools that aren't in the sandbox.
        """
        self.run("git add -A")
        if self.run("git diff --cached --quiet").exit_code == 0:
            return False
        # Pass the message on stdin via a quoted heredoc so its content is taken
        # literally (no shell expansion). `command` reaches bash -c verbatim.
        script = f"git commit --no-verify -F - <<'GERBIL_MSG'\n{message}\nGERBIL_MSG\n"
        result = self.run(script)
        if result.exit_code != 0:
            raise RuntimeError(f"git commit failed:\n{result.stderr}")
        return True

    def format_patch(self, base: str) -> str:
        """Return an mbox patch (title + message + diff) for every commit in
        base..HEAD, as produced by `git format-patch`. Apply on the host with
        `git am`."""
        return self.run(f"git format-patch {base}..HEAD --stdout", timeout=60.0).stdout

    def amend_with_file(self, repo_path: str, content: str) -> None:
        """Fold an extra file into the HEAD commit: write it into the repo, stage
        it (force, in case its directory is gitignored), and `commit --amend`
        without changing the message. Used by --include-session to embed the
        session log in the commit before format_patch()."""
        self.write_file(repo_path, content)
        self.run(f"git add -f {_quote(repo_path)}")
        result = self.run("git commit --amend --no-edit --no-verify")
        if result.exit_code != 0:
            raise RuntimeError(f"git commit --amend failed:\n{result.stderr}")


def _own_by_sandbox(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = SANDBOX_UID
    info.gid = SANDBOX_GID
    info.uname = ""
    info.gname = ""
    return info


def _quote(s: str) -> str:
    """Single-quote a string for safe use in a bash command."""
    return "'" + s.replace("'", "'\\''") + "'"
