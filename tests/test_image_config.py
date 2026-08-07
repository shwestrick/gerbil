"""Alternative sandbox images: how one is chosen, and how one is vetted.

Phase 1 (no Docker) covers _resolve_image's precedence and sandbox._image_problems
across the whole compatibility matrix -- the latter is a pure function precisely so
the matrix can be tested without building deliberately-broken images.

Phase 2 (Docker) checks the one thing a pure function cannot: that gerbil's own
image actually satisfies the contract gerbil enforces.
"""

import os
import tempfile
from argparse import Namespace
from pathlib import Path

from gerbil.cli import DEFAULT_SANDBOX_IMAGE, _resolve_image
from gerbil.sandbox import (
    IMAGE_REQUIRED_PROGRAMS,
    SANDBOX_UID,
    WORKSPACE_DIR,
    LeanSandbox,
    _image_problems,
)


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"image config test failed at: {label}\n{detail}")


def probe(*, missing=(), workspace="ok", uid=str(SANDBOX_UID), mcp=True) -> str:
    """Synthesize _IMAGE_PROBE output for a hypothetical image."""
    lines = [f"missing:{m}" for m in missing]
    if not mcp:
        lines.append("optional-missing:lean-lsp-mcp")
    if workspace != "ok":
        lines.append(f"workspace-{workspace}")
    lines.append(f"uid:{uid}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Phase 1a -- image resolution
# ----------------------------------------------------------------------

def phase_resolution() -> None:
    print("\n=== phase 1a: image resolution (no Docker) ===")
    saved = os.environ.pop("GERBIL_SANDBOX_IMAGE", None)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp)
            none = Namespace(image=None)

            check("falls back to gerbil's default image",
                  _resolve_image(none, proj) == DEFAULT_SANDBOX_IMAGE)

            os.environ["GERBIL_SANDBOX_IMAGE"] = "launcher-built:abc123"
            check("env (the launcher's build) beats the fallback",
                  _resolve_image(none, proj) == "launcher-built:abc123")

            (proj / ".gerbil").mkdir()
            config = proj / ".gerbil" / "config.toml"
            config.write_text('image = "project-image:v1"\n')
            check("project config beats the launcher's env",
                  _resolve_image(none, proj) == "project-image:v1")

            check("--image beats the project config",
                  _resolve_image(Namespace(image="flag-image:v2"), proj)
                  == "flag-image:v2")

            # Unrelated keys must not disturb resolution, so the file can grow.
            config.write_text('image = "project-image:v1"\nfuture_setting = 3\n')
            check("unknown config keys are ignored",
                  _resolve_image(none, proj) == "project-image:v1")

            # A config with no image key falls through to the env.
            config.write_text("future_setting = 3\n")
            check("config without an image key falls through",
                  _resolve_image(none, proj) == "launcher-built:abc123")

            config.write_text("image = [not valid toml\n")
            try:
                _resolve_image(none, proj)
                check("malformed TOML exits", False, "no SystemExit")
            except SystemExit as e:
                check("malformed TOML exits with the file named",
                      "config.toml" in str(e), str(e))

            config.write_text("image = 42\n")
            try:
                _resolve_image(none, proj)
                check("non-string image exits", False, "no SystemExit")
            except SystemExit as e:
                check("non-string image exits with a clear message",
                      "must be a non-empty string" in str(e), str(e))

            config.write_text('image = "   "\n')
            try:
                _resolve_image(none, proj)
                check("blank image exits", False, "no SystemExit")
            except SystemExit as e:
                check("blank image exits", "non-empty" in str(e), str(e))
    finally:
        os.environ.pop("GERBIL_SANDBOX_IMAGE", None)
        if saved is not None:
            os.environ["GERBIL_SANDBOX_IMAGE"] = saved


# ----------------------------------------------------------------------
# Phase 1b -- the compatibility matrix
# ----------------------------------------------------------------------

def phase_problems() -> None:
    print("\n=== phase 1b: image compatibility rules (no Docker) ===")

    problems, warnings = _image_problems(probe(), "0")
    check("a conforming image has no problems", problems == [], repr(problems))
    check("a conforming image has no warnings", warnings == [], repr(warnings))

    problems, _ = _image_problems(probe(uid="0"), "0")
    check("a root image is accepted", problems == [], repr(problems))

    # Every required program is genuinely required.
    for program in IMAGE_REQUIRED_PROGRAMS:
        problems, _ = _image_problems(probe(missing=[program]), "0")
        check(f"missing {program} is fatal",
              len(problems) == 1 and program in problems[0], repr(problems))

    problems, _ = _image_problems(probe(workspace="missing"), "0")
    check("a missing workspace is fatal",
          len(problems) == 1 and WORKSPACE_DIR in problems[0], repr(problems))

    problems, _ = _image_problems(probe(workspace="unwritable"), "0")
    check("an unwritable workspace is fatal",
          len(problems) == 1 and "not writable" in problems[0], repr(problems))

    problems, _ = _image_problems(probe(uid="1500"), "0")
    check("a foreign uid is fatal",
          len(problems) == 1 and "1500" in problems[0], repr(problems))

    problems, _ = _image_problems(probe(uid=""), "0")
    check("an unreadable uid is fatal",
          len(problems) == 1 and "uid" in problems[0], repr(problems))

    problems, _ = _image_problems(probe(), "")
    check("no root exec is fatal",
          len(problems) == 1 and "root" in problems[0], repr(problems))

    # lean-lsp-mcp is optional: a warning, never a refusal.
    problems, warnings = _image_problems(probe(mcp=False), "0")
    check("missing lean-lsp-mcp is only a warning",
          problems == [] and len(warnings) == 1, f"{problems} {warnings}")
    check("the mcp warning says what is lost",
          "built-in tools only" in warnings[0], repr(warnings))

    # A hand-rolled image is usually wrong in several ways; report them together.
    problems, warnings = _image_problems(
        probe(missing=["lake", "git"], workspace="missing", uid="1500", mcp=False), ""
    )
    check("all problems are reported at once", len(problems) == 4, repr(problems))
    check("...alongside the warning", len(warnings) == 1, repr(warnings))


# ----------------------------------------------------------------------
# Phase 2 -- gerbil's own image satisfies the contract (Docker)
# ----------------------------------------------------------------------

def phase_real_image() -> None:
    print("\n=== phase 2: gerbil's own image passes its own check (Docker) ===")
    LeanSandbox._fetch_mathlib_cache = lambda self: None

    with tempfile.TemporaryDirectory() as tmp:
        proj = Path(tmp)
        (proj / "Hello.lean").write_text("def hello := 1\n")
        import subprocess
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                     "commit", "-qm", "init"]):
            subprocess.run(cmd, cwd=proj, check=True, capture_output=True)

        image = os.environ.get("GERBIL_SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)
        # __enter__ runs _check_image; reaching the body means the image passed.
        with LeanSandbox(project_dir=proj, image=image) as sb:
            check("the shipped image boots and passes _check_image", True)
            code, out = sb._container.exec_run(["sh", "-c", "id -u"])
            uid = (out or b"").decode().strip()
            check("the shipped image runs as an accepted uid",
                  uid in ("0", str(SANDBOX_UID)), uid)
            sb._check_image()  # idempotent, and asserts no exception second time
            check("the check is repeatable", True)


def main() -> None:
    phase_resolution()
    phase_problems()
    phase_real_image()
    print("\nAll image config tests passed.")


if __name__ == "__main__":
    main()
