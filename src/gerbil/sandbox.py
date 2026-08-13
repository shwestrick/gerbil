import io
import json
import os
import posixpath
import re
import subprocess
import tarfile
import tempfile
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import runtime

WORKSPACE_DIR = "/workspace/project"


def _host_timezone() -> str | None:
    """The host's IANA timezone name (e.g. "America/New_York"), so the container
    can be pinned to it. Without this the container runs in UTC, and every git
    commit the agent (or gerbil's squash) makes inside it is timestamped in UTC
    with a +0000 offset -- hours off from the user's wall clock.

    Prefers an explicit TZ env var, then the target of the /etc/localtime symlink
    (the everywhere-on-Unix convention, including macOS). Returns None if neither
    is available, leaving the container at its default (UTC)."""
    tz = os.environ.get("TZ")
    if tz:
        return tz
    try:
        target = os.readlink("/etc/localtime")
    except OSError:
        return None
    # .../zoneinfo/America/New_York -> America/New_York
    marker = "zoneinfo/"
    idx = target.rfind(marker)
    return target[idx + len(marker):] if idx != -1 else None


def _sanitized_git_dir(repo_root: Path, scratch: Path) -> Path:
    """Build a stripped .git for the sandbox: the current branch's history and
    nothing else. Returns the path of the new .git directory (inside `scratch`).

    The agent needs the history behind the commit it starts from (past-commit
    lookups, format-patch), but must not see anything beyond that: no other
    branches or tags, no remotes/upstream config, no reflogs, no stash -- any
    of which could leak information into the sandbox (in-progress work, private
    branch names, remote URLs and credentials-bearing remote helpers).

    Implemented as a `git fetch` of HEAD into a fresh repo rather than a copy
    of the host .git: the fetch transport transfers only objects *reachable*
    from HEAD, so loose/packed objects belonging to other branches stay behind
    too (a raw copy -- or a local `git clone`, which hardlinks the entire
    object store -- would carry them along, recoverable via `git cat-file` /
    `git fsck --lost-found`). Fetching HEAD also works from a detached HEAD
    and from a linked worktree, where .git is a file rather than a directory."""

    def git(*args: str) -> None:
        r = subprocess.run(
            ["git", "-C", str(clone), *args], capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed while building the sanitized "
                f"repo for upload:\n{r.stderr}"
            )

    # Keep the real branch name -- it's part of the current branch, and the
    # prompt may refer to it. A detached HEAD gets a neutral stand-in.
    named = subprocess.run(
        ["git", "symbolic-ref", "--short", "-q", "HEAD"],
        cwd=repo_root, capture_output=True, text=True,
    )
    branch = named.stdout.strip() or "gerbil-work"

    # Init on a throwaway branch name that cannot equal `branch`: git refuses
    # to fetch into the checked-out branch (even an unborn one), so HEAD must
    # point elsewhere until the fetch lands and we flip it.
    clone = scratch / "sanitized"
    placeholder = "gerbil-init" if branch != "gerbil-init" else "gerbil-init-2"
    subprocess.run(["git", "init", "-q", "-b", placeholder, str(clone)], check=True)
    # No reflogs: each entry would record the fetch command line -- host path
    # included -- and old ref positions.
    git("config", "core.logAllRefUpdates", "false")
    # HEAD is always advertised by upload-pack, so no ref name is needed;
    # --no-tags stops tag auto-following from dragging tag refs back in.
    git("fetch", "-q", "--no-tags", str(repo_root), f"+HEAD:refs/heads/{branch}")
    git("symbolic-ref", "HEAD", f"refs/heads/{branch}")
    # FETCH_HEAD records the host path the objects came from -- drop it.
    (clone / ".git" / "FETCH_HEAD").unlink(missing_ok=True)
    return clone / ".git"


def submodule_entries(repo_root: Path, prefix: str = "") -> list[tuple[str, str]]:
    """Every submodule in the repo, recursively: (path relative to repo_root,
    the commit sha recorded for it in its parent's tree). Parents come before
    their children -- the order the upload needs.

    Read straight out of the index (mode 160000 entries) rather than parsed from
    `git submodule status`: that keeps it NUL-safe for paths with spaces, and it
    sees gitlinks that have no .gitmodules entry at all.

    Recursion stops at an uninitialized submodule -- there is no repo there to
    read an index from. cli._require_clean_submodules rejects those before any
    run, so in practice callers always get the whole tree.

    Returns [] for the overwhelmingly common no-submodule repo, which leaves the
    entire submodule code path inert."""
    out = subprocess.run(
        ["git", "ls-files", "-s", "-z"],
        cwd=repo_root, capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    entries: list[tuple[str, str]] = []
    for record in out.stdout.split("\0"):
        if not record.startswith("160000 "):
            continue
        # "<mode> <sha> <stage>\t<path>"
        meta, _, path = record.partition("\t")
        sub = prefix + path
        entries.append((sub, meta.split(" ")[1]))
        if (repo_root / path / ".git").exists():
            entries.extend(submodule_entries(repo_root / path, sub + "/"))
    return entries

# Must match the uid/gid of the user created in the Dockerfile, so files we
# upload land owned by that user and git operations don't hit ownership errors.
# (Under podman it is also the uid/gid uploads end up owned by: they are
# unpacked by the container's own tar, running as that same user -- see
# runtime._PodmanContainer.put_archive.)
SANDBOX_UID = 1000
SANDBOX_GID = 1000

# Programs gerbil itself runs inside the container. bash + timeout wrap every
# LeanSandbox.run; git is all of the bookkeeping; tar is what podman's
# put_archive unpacks uploads with; chown reasserts ownership after an upload;
# mktemp backs --ralph_done; lake fetches the mathlib cache and builds. Anything
# else the agent needs is the image's business, not gerbil's.
IMAGE_REQUIRED_PROGRAMS = ("bash", "timeout", "git", "tar", "chown", "id", "mktemp", "lake")

# Probed under plain `sh`, so it must stay POSIX. Emits one token per problem
# plus the container user's uid; _image_problems turns that into English.
_IMAGE_PROBE = "\n".join(
    [f'for b in {" ".join(IMAGE_REQUIRED_PROGRAMS)}; do',
     '  command -v "$b" >/dev/null 2>&1 || echo "missing:$b"',
     'done',
     'command -v lean-lsp-mcp >/dev/null 2>&1 || echo "optional-missing:lean-lsp-mcp"',
     f'if [ ! -d {WORKSPACE_DIR} ]; then echo workspace-missing',
     f'elif [ ! -w {WORKSPACE_DIR} ]; then echo workspace-unwritable',
     'fi',
     'echo "uid:$(id -u 2>/dev/null)"']
)


def uses_mathlib(project_dir: Path) -> bool:
    """Whether the Lake project at project_dir depends on mathlib, directly or
    transitively (or simply is mathlib).

    `cache` is an executable *mathlib* provides, so `lake exe cache get` is not
    just pointless in a project without mathlib -- it fails. Answering this up
    front is what lets the startup fetch be skipped instead of erroring out.

    Three sources, and any one of them saying yes is enough:

      - lake-manifest.json, Lake's *resolved* dependency graph. The only source
        that sees mathlib arriving transitively, via a package that requires it.
      - lakefile.toml's [[require]] names.
      - lakefile.lean's `require` lines.

    The manifest alone is not enough: it is generated by `lake update` and can
    lag a lakefile that has just gained mathlib. Reading all three and taking
    the union means a stale manifest costs a needless fetch attempt rather than
    a silent from-source rebuild of mathlib.

    When nothing at all can be read -- no manifest, no parseable lakefile -- the
    answer is True, which is gerbil's original unconditional-fetch behavior. The
    skip is only ever taken on positive evidence that mathlib is absent."""
    read_any = False

    manifest = project_dir / "lake-manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            data = None
        if isinstance(data, dict):
            read_any = True
            names = {data.get("name")}
            packages = data.get("packages")
            if isinstance(packages, list):
                names |= {p.get("name") for p in packages if isinstance(p, dict)}
            if any(str(n).lower() == "mathlib" for n in names if n):
                return True

    toml_file = project_dir / "lakefile.toml"
    if toml_file.is_file():
        try:
            settings = tomllib.loads(toml_file.read_text())
        except (tomllib.TOMLDecodeError, OSError, UnicodeDecodeError):
            settings = None
        if isinstance(settings, dict):
            read_any = True
            names = {settings.get("name")}
            requires = settings.get("require")
            if isinstance(requires, list):
                names |= {r.get("name") for r in requires if isinstance(r, dict)}
            if any(str(n).lower() == "mathlib" for n in names if n):
                return True

    lean_file = project_dir / "lakefile.lean"
    if lean_file.is_file():
        try:
            text = lean_file.read_text()
        except (OSError, UnicodeDecodeError):
            text = None
        if text is not None:
            read_any = True
            # `require mathlib from git ...`, `require «mathlib» ...`, and the
            # package declaration of mathlib itself.
            if re.search(r"^\s*(require|package)\s+«?mathlib»?\b", text, re.M):
                return True

    return not read_any


def _image_problems(probe: str, root_uid: str) -> tuple[list[str], list[str]]:
    """Turn _IMAGE_PROBE's output (and the uid a root exec reported) into
    (fatal problems, warnings), both as human-readable sentences.

    Split out as a pure function so the whole compatibility matrix is testable
    without building deliberately-broken container images."""
    tokens = probe.split()
    problems, warnings = [], []

    missing = [t.split(":", 1)[1] for t in tokens if t.startswith("missing:")]
    if missing:
        problems.append(f"missing required program(s): {', '.join(missing)}")
    if "workspace-missing" in tokens:
        problems.append(
            f"{WORKSPACE_DIR} does not exist in the image -- it must be created "
            "there (podman does not materialize a missing --workdir)"
        )
    if "workspace-unwritable" in tokens:
        problems.append(f"{WORKSPACE_DIR} is not writable by the container user")

    uid = next((t.split(":", 1)[1] for t in tokens if t.startswith("uid:")), "")
    if not uid:
        problems.append("could not determine the container user's uid")
    elif uid not in ("0", str(SANDBOX_UID)):
        # Uploads are tarred as SANDBOX_UID and chowned to it afterwards, so any
        # other uid gets a workspace it cannot write to.
        problems.append(
            f"the image runs as uid {uid}; gerbil uploads files owned by uid "
            f"{SANDBOX_UID}, so the image must run as uid {SANDBOX_UID} (or 0)"
        )
    if root_uid.strip() != "0":
        problems.append(
            "cannot exec as root in the container (needed once after upload, to "
            "reassert ownership of the workspace)"
        )

    if "optional-missing:lean-lsp-mcp" in tokens:
        warnings.append(
            "lean-lsp-mcp is not on PATH in this image; the session will run "
            "with gerbil's built-in tools only (no Lean LSP tools)"
        )
    return problems, warnings


@dataclass
class CommandResult:
    """Result of running a command in the sandbox."""

    command: str
    exit_code: int
    stdout: str
    stderr: str
    timeout_occurred: bool


class LeanSandbox:
    """Sandboxed Lean environment running inside a container.

    Isolation is provided entirely by the container runtime -- Docker by
    default, or podman when GERBIL_SANDBOX=podman (see runtime.py). We talk to
    the container directly through a Docker-SDK-shaped client: exec_run for
    commands, put_archive/cat for file I/O. Podman is driven through its CLI
    behind that same interface, so nothing below branches on the runtime.

    The Lake project need not be the git repo root: we upload the whole repo
    (rooted at repo_root) into WORKSPACE_DIR, and operate on the Lake project at
    WORKSPACE_DIR/<subdir>, where subdir is project_dir relative to repo_root.

    At startup:
      - Uploads all git-tracked files from repo_root into the container, plus a
        sanitized .git holding only the current branch's history (full history
        of HEAD for format-patch and past-commit lookups, but no other branches,
        tags, remotes, or reflogs -- the agent sees nothing beyond the branch).
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
        image: str = "gerbil-lean-sandbox:latest",
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
        self._client = runtime.client()
        self._container = None
        # Submodule paths (relative to repo_root), filled in by _upload_project.
        # Empty for the common no-submodule repo.
        self.submodule_paths: list[str] = []
        # Repo-relative files force-added (git add -f) into every squash and
        # wip snapshot, past any .gitignore -- today the --fill-sorry plan
        # file at .gerbil/plans/, which must ship in the patch even though
        # .gerbil/ is conventionally gitignored, exactly like the folded
        # session log that amend_with_file force-adds. Set by cli.py.
        self.force_include: list[str] = []

    @property
    def project_path(self) -> str:
        """Container path of the Lake project root: WORKSPACE_DIR (the repo root)
        or a subdirectory of it. All Lake/agent/MCP operations run here; git
        commands work too, since git resolves .git up the tree."""
        return posixpath.join(WORKSPACE_DIR, self._subdir) if self._subdir else WORKSPACE_DIR

    def __enter__(self) -> "LeanSandbox":
        # Pin the container to the host's timezone so git commit dates (and any
        # `date` the agent runs) match the user's wall clock instead of UTC. The
        # image ships full tzdata, so the IANA name resolves inside the container.
        tz = _host_timezone()
        self._container = self._client.containers.run(
            self.image,
            command="sleep infinity",
            detach=True,
            auto_remove=True,
            working_dir=WORKSPACE_DIR,
            environment={"TZ": tz} if tz else None,
        )
        try:
            self._wait_running()
            self._check_image()
            self._upload_project()
            self._configure_git()
            if self.fetch_cache:
                self._fetch_mathlib_cache()
        except BaseException:
            # A crash -- or, far more likely, a Ctrl-C during the slow startup
            # cache fetch -- lands before the `with` body is entered, so
            # __exit__ will never run. Without this the container outlives
            # gerbil, running `sleep infinity` forever. BaseException on
            # purpose: KeyboardInterrupt/SystemExit are exactly the cases.
            self.__exit__()
            raise
        return self

    @property
    def container_id(self) -> str:
        """The running container's id (used to `docker`/`podman exec` into the
        sandbox)."""
        if self._container is None:
            raise RuntimeError("sandbox is not running")
        return self._container.id

    def __exit__(self, *_) -> None:
        """Stop the container (auto_remove then deletes it). Idempotent -- also
        called from __enter__'s failure path, where the `with` body was never
        entered. The graceful stop takes its full 5s timeout (`sleep` as PID 1
        ignores SIGTERM), so an impatient second Ctrl-C during it is likely;
        that falls back to an immediate kill rather than leaking the container,
        then re-raises so the interrupt still wins."""
        container, self._container = self._container, None
        if container is None:
            return
        try:
            container.stop(timeout=5)
        except KeyboardInterrupt:
            try:
                container.kill()
            except Exception:
                pass
            raise
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
            # An image with an ENTRYPOINT is the usual cause: the container is
            # started with `sleep infinity` as its *command*, which an ENTRYPOINT
            # swallows as arguments instead of running. Say so rather than
            # sitting out the full retry budget to report a bare timeout.
            if self._container.status == "exited":
                raise RuntimeError(
                    f"sandbox container from image {self.image} exited immediately.\n"
                    "gerbil runs the container as `sleep infinity` and drives it "
                    "with exec, so the image must not define an ENTRYPOINT that "
                    "swallows that command."
                )
            time.sleep(delay)
        raise TimeoutError("sandbox container did not reach running state")

    def _check_image(self) -> None:
        """Verify the image can host a gerbil session, before any upload happens.

        Custom images (--image / .gerbil/config.toml) are the reason this exists:
        an image that is subtly wrong otherwise fails deep into a session, as an
        unwritable upload or a missing `lake`. Every problem is collected and
        reported at once, since a hand-rolled image is usually wrong in more than
        one way and one round trip should say so.

        Deliberately does NOT go through self.run(): that wraps everything in
        `timeout ... bash -c`, two of the very things being checked. The probe
        runs under plain `sh` instead, and an image without even that fails here
        with its own message."""
        try:
            code, streams = self._container.exec_run(
                ["sh", "-c", _IMAGE_PROBE], demux=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"image {self.image} is not usable as a gerbil sandbox: could not "
                f"run `sh` in it ({exc})."
            ) from exc
        probe = (streams[0] or b"").decode(errors="replace")
        if code != 0 and not probe.strip():
            stderr = (streams[1] or b"").decode(errors="replace").strip()
            raise RuntimeError(
                f"image {self.image} is not usable as a gerbil sandbox: its `sh` "
                f"could not run the compatibility probe.\n{stderr}"
            )
        # The post-upload chown runs as root; confirm that exec works at all.
        root_code, root_out = self._container.exec_run(["id", "-u"], user="root")
        root_uid = (root_out or b"").decode(errors="replace") if root_code == 0 else ""

        problems, warnings = _image_problems(probe, root_uid)
        if problems:
            listed = "\n".join(f"  - {p}" for p in problems)
            raise RuntimeError(
                f"image {self.image} is not compatible with gerbil:\n{listed}\n\n"
                "A sandbox image must provide bash/timeout/git/tar/chown/mktemp "
                f"and lake, must own a writable {WORKSPACE_DIR}, and must run as "
                f"uid {SANDBOX_UID} or 0. If this is gerbil's own image, rebuild "
                "it (see src/lean-sandbox/Dockerfile)."
            )
        # Only once the image is known good -- a warning alongside a refusal is
        # noise about a session that is not going to run.
        for warning in warnings:
            print(f"warning: {warning}")

    def _upload_project(self) -> None:
        """Upload the repository into the container: the tracked files plus a
        *sanitized* .git holding only the current branch (see _sanitized_git_dir
        -- the agent must not see other branches, remotes, tags, or reflogs).
        Rooted at repo_root, which may be an ancestor of the Lake project. The
        working tree is required to be clean (see the CLI preflight), so the
        tracked files match HEAD and no untracked files are uploaded -- the agent
        commits on top of a clean, known baseline.

        Submodules are uploaded separately (_upload_submodule): git reports each
        one here as a single gitlink entry -- a directory on disk, not a file --
        and its contents live in a repo of its own."""
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=self.repo_root,
            capture_output=True,
            check=True,
        ).stdout
        rels = [p for p in out.decode().split("\0") if p]

        self.submodule_paths = [p for p, _ in submodule_entries(self.repo_root)]
        subs = set(self.submodule_paths)

        buf = io.BytesIO()
        with tempfile.TemporaryDirectory() as scratch:
            gitdir = _sanitized_git_dir(self.repo_root, Path(scratch))
            with tarfile.open(fileobj=buf, mode="w") as tar:
                for rel in rels:
                    if rel in subs:
                        continue
                    local = self.repo_root / rel
                    if local.is_file():
                        tar.add(local, arcname=rel, filter=_own_by_sandbox)
                tar.add(gitdir, arcname=".git", filter=_own_by_sandbox)
                for i, sub in enumerate(self.submodule_paths):
                    self._upload_submodule(tar, sub, Path(scratch) / f"sub{i}")
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

    def _upload_submodule(self, tar: tarfile.TarFile, sub: str, scratch: Path) -> None:
        """Add one submodule to the upload tar: its tracked files plus a .git of
        its own, so it lands in the container fully populated -- exactly as if
        `git submodule update --init --recursive` had been run there, but sourced
        from the host's already-initialized working tree, so no network is needed
        (the sandbox has none) and nothing on the host is touched.

        The .git goes in as a real *directory* at <sub>/.git, not the modern
        gitdir-file-plus-.git/modules layout. Git still supports that (pre-1.7.8)
        arrangement, and it keeps this simple: nested submodules need no
        module-path juggling -- each just gets its own .git inside its own path --
        and _sanitized_git_dir is reused verbatim, so a submodule's history is
        stripped exactly like the superproject's (only the current commit's
        history; no other branches, tags, remotes, or reflogs).

        The submodule's own gitlink entries, if it has any, are skipped here for
        the same reason as in _upload_project: each is uploaded by its own pass."""
        root = self.repo_root / sub
        out = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        ).stdout
        for rel in (p for p in out.decode().split("\0") if p):
            local = root / rel
            if local.is_file():
                tar.add(
                    local, arcname=posixpath.join(sub, rel), filter=_own_by_sandbox
                )
        tar.add(
            _sanitized_git_dir(root, scratch),
            arcname=posixpath.join(sub, ".git"),
            filter=_own_by_sandbox,
        )

    def _configure_git(self) -> None:
        """Set a local committer identity so gerbil can commit the agent's work.
        The uploaded repo keeps its real history; we do not re-init."""
        self._git("config user.email gerbil@local")
        self._git("config user.name gerbil")
        # Uploaded .git is owned by the sandbox user (uid 1000 == container user),
        # but add safe.directory defensively in case of ownership quirks.
        self.run(f"git config --global --add safe.directory {WORKSPACE_DIR}")
        # The sanitized .git ships without an index (it was built by fetch, not
        # checkout); build one matching HEAD so the baseline reads as clean --
        # otherwise the agent's first `git status` would show every file as a
        # staged deletion plus an untracked file.
        self._git("read-tree HEAD")

        for sub in self.submodule_paths:
            # Same story one level down: each uploaded submodule .git was built
            # by fetch, so it has no index either.
            self._git_at(sub, "read-tree HEAD")
            self.run(
                "git config --global --add safe.directory "
                f"{posixpath.join(WORKSPACE_DIR, sub)}"
            )
        if self.submodule_paths:
            # Present submodules to the agent as pinned, fixed dependencies: this
            # keeps `git status` clean and stops `git add -A` from staging a moved
            # gitlink if the agent commits inside one. It is only the polite half
            # of the rule the system prompt states -- _reset_submodule_state is
            # what actually guarantees no submodule change ever reaches a patch.
            # Repo-wide, so it needs no submodule names and covers nested ones.
            self._git("config diff.ignoreSubmodules all")
            for sub in self.submodule_paths:
                self._git_at(sub, "config diff.ignoreSubmodules all")

    def _fetch_mathlib_cache(self) -> None:
        """Download precompiled mathlib oleans. Runs once per session.

        Skipped for a project that does not depend on mathlib: `cache` is an
        executable mathlib itself provides, so `lake exe cache get` does not
        merely waste time there, it fails -- and used to take the whole session
        down with it unless the user knew to pass --skip-cache."""
        if not uses_mathlib(self.project_dir):
            print("No mathlib dependency found; skipping the mathlib cache fetch.")
            return
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

    def run_script(self, script: str, timeout: float = 300.0) -> CommandResult:
        """Upload `script` INTO the Lake project directory and run it there, so it
        behaves exactly as if the user had invoked it from the project root: the
        CWD is the project, and the script file itself lives in the project, so
        `$0`, `dirname "$0"`, and any sibling-relative paths resolve into the
        project. Running it from /tmp instead silently breaks the common
        `cd "$(dirname "$0")"` idiom -- the script lands in /tmp, where (among
        other things) elan can't find the project's lean-toolchain and reports
        "no default toolchain configured".

        Invoked by path so a `#!` shebang is honored (and the shell falls back to
        a shell script when there is none). Used for the --ralph_done termination
        check: its exit code is the signal, and stdout/stderr are returned for
        display. The uploaded file is removed afterward -- and the check only runs
        after the session's patch is produced -- so it never shows up in a diff."""
        # Pick a unique filename via mktemp so we never collide with (or clobber)
        # an existing file in the project. run()'s workdir is the project dir, and
        # a bare (slashless) template makes mktemp create the file there and print
        # its project-relative name -- exactly where we want the script to live.
        mk = self.run("mktemp .gerbil-ralph-done-check.XXXXXX")
        name = mk.stdout.strip()
        if mk.exit_code != 0 or not name:
            raise RuntimeError(
                "could not create a temp file for the --ralph_done script:\n"
                f"{mk.stderr or mk.stdout}"
            )
        # mktemp made an empty file; overwrite it with the script contents.
        self.write_file(name, script)
        try:
            return self.run(
                f"chmod +x {_quote(name)} && {_quote('./' + name)}", timeout=timeout
            )
        finally:
            # Best-effort cleanup so the check script never lingers in the tree
            # (e.g. to be swept into the next ralph session's commit).
            self.run(f"rm -f {_quote(name)}")

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

    @property
    def _git_env(self) -> str:
        """Environment that pins git to gerbil's real repository, bypassing all
        repository discovery. Without this, an agent that runs `git init` in the
        Lake project dir creates a nested .git that shadows the real repo for
        every later command run from there -- so gerbil's diff/commit/format-patch
        would silently operate on the wrong repository. Setting GIT_DIR and
        GIT_WORK_TREE explicitly makes gerbil's own git immune to that."""
        return self._git_env_at("")

    def _git_env_at(self, sub: str) -> str:
        """_git_env, scoped to a submodule (sub == "" is the repo itself). Every
        repo gerbil touches in the container gets the same pinned-env treatment,
        submodules included."""
        root = posixpath.join(WORKSPACE_DIR, sub) if sub else WORKSPACE_DIR
        return f"GIT_DIR={posixpath.join(root, '.git')} GIT_WORK_TREE={root}"

    def _git(self, args: str, timeout: float = 60.0) -> CommandResult:
        """Run a git command against gerbil's real repository (see _git_env). All
        of gerbil's internal git -- never the agent's bash tool -- goes through
        here, so the agent cannot redirect gerbil's bookkeeping to a stray repo."""
        return self.run(f"{self._git_env} git {args}", timeout=timeout)

    def _git_at(self, sub: str, args: str, timeout: float = 60.0) -> CommandResult:
        """_git, against one of the repo's submodules."""
        return self.run(f"{self._git_env_at(sub)} git {args}", timeout=timeout)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def head(self) -> str:
        """The current HEAD commit hash. Raises if the repo has no commits
        (rev-parse otherwise echoes a bogus 'HEAD' on an unborn branch)."""
        result = self._git("rev-parse --verify HEAD")
        if result.exit_code != 0:
            raise RuntimeError("repository has no commits (no HEAD)")
        return result.stdout.strip()

    def get_diff(self) -> str:
        """Return a git diff of the uncommitted working-tree changes (vs HEAD)."""
        self._git("add -A")
        return self._git("diff --cached").stdout

    def diff_since(self, base: str) -> str:
        """A git diff of everything that changed since `base` -- including changes
        the agent committed in intermediate commits, not just the uncommitted
        ones. Used to describe (and squash) the whole session's work."""
        self._git("add -A")
        return self._git(f"diff --cached {_quote(base)}").stdout

    def _listing(self, args: str) -> list[tuple[str, str]]:
        """(mode, path) for every entry of a `ls-files -s -z` / `ls-tree -r -z`
        listing -- both put the mode first and the path after a tab. -z keeps
        paths literal (git would otherwise quote unusual ones)."""
        entries = []
        for record in self._git(args).stdout.split("\0"):
            meta, tab, path = record.partition("\t")
            if tab:
                entries.append((meta.split(" ", 1)[0], path))
        return entries

    def _reset_submodule_state(self, base: str) -> None:
        """Undo, in the index only, anything the agent did to a submodule:
        restore every gitlink to the commit `base` records, drop any gitlink base
        did not have, and restore .gitmodules (editing it is submodule
        manipulation too).

        This is what makes "the agent does no submodule manipulation" a guarantee
        rather than a request, and it has to be enforced rather than asked for,
        because a patch cannot carry submodule work in the first place:
        format-patch renders a gitlink as a single `Subproject commit <sha>` line,
        and commits the agent makes inside a submodule live only in this container
        and die with it. Exporting a moved gitlink would yield a patch that
        `git am` applies happily and that leaves the user's repo pointing at a
        commit existing nowhere -- breaking only much later, at someone else's
        `git submodule update`.

        Index-only on purpose: `git reset <tree-ish> -- <path>` does not touch the
        working tree, so whatever the agent wrote inside a submodule stays on disk
        for the rest of the session. Paths absent from base are dropped from the
        index outright, which is exactly right for a submodule the agent added.

        A no-op on a repo with no gitlinks in either the index or base."""
        index = self._listing("ls-files -s -z")
        tree = self._listing(f"ls-tree -r -z {_quote(base)}")
        paths = sorted({p for mode, p in index + tree if mode == "160000"})
        if not paths:
            return
        if any(p == ".gitmodules" for _, p in index + tree):
            paths.append(".gitmodules")
        spec = " ".join(_quote(p) for p in paths)
        self._git(f"reset -q {_quote(base)} -- {spec}")

    def _add_force_included(self) -> None:
        """Stage the force_include files past any .gitignore, right after the
        squash/wip `git add -A` (which skips ignored paths). Only files that
        actually exist: `git add -f` on a missing pathspec is an error, and a
        force-include the agent never wrote (or deleted) is simply absent
        from the snapshot rather than fatal to it."""
        for path in self.force_include:
            if self.run(f"test -f {_quote(path)}").exit_code == 0:
                self._git(f"add -f -- {_quote(path)}")

    def changed_paths(self, base: str) -> list[str]:
        """Repo-root-relative paths differing between `base` and HEAD. Called
        after squash_commit, when HEAD is the squashed session commit, this is
        exactly the file list the emitted patch carries -- what the
        --fill-sorry patch gate checks against the spec's off_limits. -z keeps
        paths literal (git would otherwise quote unusual ones)."""
        result = self._git(f"diff --name-only --no-renames -z {_quote(base)}..HEAD")
        return [p for p in result.stdout.split("\0") if p]

    def squash_commit(self, base: str, message: str) -> bool:
        """Collapse everything from `base` to the current working tree -- the
        agent's intermediate commits AND its uncommitted changes -- into a SINGLE
        commit on top of base, so that format_patch(base) yields exactly one
        patch. Returns False (committing nothing) when nothing changed from base.

        Stages the full working tree, soft-resets HEAD back to base (which keeps
        that staged state), and commits once."""
        self._git("add -A")
        self._add_force_included()
        # Before the emptiness check, not after: a session whose only change was
        # to a submodule has, after this, changed nothing at all, and must be
        # reported as such rather than producing an empty patch.
        self._reset_submodule_state(base)
        if self._git(f"diff --cached --quiet {_quote(base)}").exit_code == 0:
            return False  # working tree identical to base -> nothing to commit
        reset = self._git(f"reset --soft {_quote(base)}")
        if reset.exit_code != 0:
            raise RuntimeError(f"git reset --soft {base} failed:\n{reset.stderr}")
        # Heredoc so the message is taken literally (no shell expansion).
        script = (
            f"{self._git_env} git commit --no-verify -F - "
            f"<<'GERBIL_MSG'\n{message}\nGERBIL_MSG\n"
        )
        result = self.run(script)
        if result.exit_code != 0:
            raise RuntimeError(f"git commit (squash) failed:\n{result.stderr}")
        return True

    def wip_patch(self, base: str) -> str:
        """A live `git format-patch` from `base` to the current state -- every
        commit the agent made itself, plus a snapshot of the uncommitted working
        tree -- produced WITHOUT moving HEAD or touching the working tree.

        This is the resume snapshot, refreshed each turn. Unlike `git diff HEAD`
        it does not lose changes the agent committed internally: the index is
        captured as a throwaway commit (via write-tree + commit-tree) parented on
        HEAD, so format-patch sees base..HEAD plus the uncommitted delta. Applying
        the result onto a clean `base` (git apply / git am) reproduces the full
        tree. Returns "" when nothing differs from base."""
        self._git("add -A")
        self._add_force_included()
        self._reset_submodule_state(base)
        tree = self._git("write-tree").stdout.strip()
        if not tree:
            return ""
        wip = self._git(f"commit-tree {tree} -p HEAD -m gerbil-wip").stdout.strip()
        if not wip or self._git(f"diff --quiet {base} {wip}").exit_code == 0:
            return ""  # identical to base -> nothing to snapshot
        result = self._git(f"format-patch {base}..{wip} --stdout", timeout=60.0)
        return result.stdout if result.exit_code == 0 else ""

    def checkout_force(self, ref: str) -> None:
        """Hard-reset the working tree to `ref` (detaching HEAD), discarding any
        tracked-file changes. Untracked files -- crucially the .lake build cache
        and fetched mathlib oleans -- are left in place, so the agent does not pay
        to refetch them. Used by --resume to recreate a session's starting commit
        before the saved working-tree patch is reapplied on top."""
        result = self._git(f"checkout -f {_quote(ref)}")
        if result.exit_code != 0:
            raise RuntimeError(f"git checkout {ref} failed:\n{result.stderr}")

    def _stage_patch(self, text: str) -> str:
        """Write patch text to a temp file inside the container and return its
        path. Patches are uploaded as a tar stream (put_archive), not passed on
        the command line -- a single exec argument is capped at ~128 KiB, which
        real session patches blow past. Lives in /tmp, outside the work tree, so
        it never shows up in a diff."""
        path = "/tmp/gerbil-patch.mbox"
        self.write_file(path, text)
        return path

    def apply_diff(self, diff_text: str) -> None:
        """Apply a working-tree patch (as produced by get_diff) to the current
        tree. No-op for an empty patch. Used by --resume to restore the
        uncommitted edits a crashed session had made on top of its base commit."""
        if not diff_text.strip():
            return
        path = self._stage_patch(diff_text)
        result = self._git(
            f"apply --whitespace=nowarn {_quote(path)}", timeout=120.0
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"git apply failed:\n{result.stderr or result.stdout}"
            )

    def commit(self, message: str) -> bool:
        """Commit all current changes inside the container on top of real HEAD.
        Returns False if there was nothing to commit. Skips hooks (--no-verify),
        since host hooks may assume tools that aren't in the sandbox.
        """
        self._git("add -A")
        if self._git("diff --cached --quiet").exit_code == 0:
            return False
        # Pass the message on stdin via a quoted heredoc so its content is taken
        # literally (no shell expansion). `command` reaches bash -c verbatim.
        script = (
            f"{self._git_env} git commit --no-verify -F - "
            f"<<'GERBIL_MSG'\n{message}\nGERBIL_MSG\n"
        )
        result = self.run(script)
        if result.exit_code != 0:
            raise RuntimeError(f"git commit failed:\n{result.stderr}")
        return True

    def git_am(self, patch_text: str) -> None:
        """Apply a format-patch (mbox) as a commit via `git am` -- the same way
        the host `gerbil commit` does. Used by --resume to replay a ralph chain's
        prior-session patches in order, rebuilding the committed history a
        mid-chain session started from. Aborts and raises on failure."""
        if not patch_text.strip():
            return
        path = self._stage_patch(patch_text)
        result = self._git(f"am {_quote(path)}", timeout=180.0)
        if result.exit_code != 0:
            self._git("am --abort")
            raise RuntimeError(f"git am failed:\n{result.stderr or result.stdout}")

    def format_patch(self, base: str) -> str:
        """Return an mbox patch (title + message + diff) for every commit in
        base..HEAD, as produced by `git format-patch`. Apply on the host with
        `git am`. Raises if git itself fails (e.g. `base` is unknown -- which is
        what a tampered/re-initialized repo looks like), so the caller never
        mistakes an error for 'no changes'."""
        result = self._git(f"format-patch {base}..HEAD --stdout", timeout=60.0)
        if result.exit_code != 0:
            raise RuntimeError(
                f"git format-patch {base[:12]}..HEAD failed:\n"
                f"{result.stderr or result.stdout}"
            )
        return result.stdout

    def amend_with_file(self, repo_path: str, content: str) -> None:
        """Fold an extra file into the HEAD commit: write it into the repo, stage
        it (force, in case its directory is gitignored), and `commit --amend`
        without changing the message. Used to embed the session log in the
        commit before format_patch() (unless --omit-session-log)."""
        self.write_file(repo_path, content)
        self._git(f"add -f {_quote(repo_path)}")
        result = self._git("commit --amend --no-edit --no-verify")
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
