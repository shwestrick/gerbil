"""Mathlib detection: whether `lake exe cache get` should run at all.

`cache` is an executable mathlib provides, so the fetch fails outright in a
project without mathlib. sandbox.uses_mathlib decides, and is a pure function
over the project's Lake files -- no Docker, no Lake, no network needed here.
"""

import json
import tempfile
from pathlib import Path

from gerbil.sandbox import uses_mathlib


def check(label: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {label}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        raise SystemExit(f"mathlib detection test failed at: {label}\n{detail}")


def project(files: dict[str, str] | None = None) -> Path:
    """A throwaway project directory containing exactly the given files."""
    tmp = Path(tempfile.mkdtemp())
    for name, content in (files or {}).items():
        (tmp / name).write_text(content)
    return tmp


def manifest(*names: str, name: str = "myproject") -> str:
    """A lake-manifest.json naming `names` as resolved packages."""
    return json.dumps({
        "version": "1.1.0",
        "packagesDir": ".lake/packages",
        "name": name,
        "packages": [
            {"type": "git", "name": n, "scope": "leanprover-community",
             "rev": "deadbeef", "url": f"https://example.com/{n}"}
            for n in names
        ],
    })


TOML_MATHLIB = 'name = "myproj"\n\n[[require]]\nname = "mathlib"\nscope = "leanprover-community"\n'
TOML_OTHER = 'name = "myproj"\n\n[[require]]\nname = "batteries"\n'
LEAN_MATHLIB = (
    'import Lake\nopen Lake DSL\n\n'
    'require mathlib from git "https://github.com/leanprover-community/mathlib4"\n\n'
    'package myproj\n'
)
LEAN_OTHER = (
    'import Lake\nopen Lake DSL\n\nrequire batteries from git "x"\n\npackage myproj\n'
)


def main() -> None:
    print("=== detects mathlib ===")

    check("manifest listing mathlib directly",
          uses_mathlib(project({"lake-manifest.json": manifest("mathlib", "batteries")})))

    check("manifest where mathlib arrives transitively",
          uses_mathlib(project({"lake-manifest.json": manifest("someLib", "mathlib")})))

    check("the project that IS mathlib (manifest)",
          uses_mathlib(project({"lake-manifest.json": manifest("batteries", name="mathlib")})))

    check("lakefile.toml [[require]]",
          uses_mathlib(project({"lakefile.toml": TOML_MATHLIB})))

    check("the project that IS mathlib (lakefile.toml)",
          uses_mathlib(project({"lakefile.toml": 'name = "mathlib"\n'})))

    check("lakefile.lean require from git",
          uses_mathlib(project({"lakefile.lean": LEAN_MATHLIB})))

    check("lakefile.lean with guillemets",
          uses_mathlib(project({"lakefile.lean": 'require «mathlib» from git "x"\n'})))

    check("lakefile.lean package mathlib",
          uses_mathlib(project({"lakefile.lean": "package mathlib\n"})))

    # A manifest that has not caught up with a lakefile that just gained mathlib
    # must not produce a silent from-source rebuild of mathlib.
    check("stale manifest, lakefile says mathlib",
          uses_mathlib(project({"lake-manifest.json": manifest("batteries"),
                                "lakefile.toml": TOML_MATHLIB})))

    print("\n=== correctly reports no mathlib ===")

    check("manifest with unrelated packages only",
          not uses_mathlib(project({"lake-manifest.json": manifest("batteries", "aesop")})))

    check("manifest with no packages at all",
          not uses_mathlib(project({"lake-manifest.json": manifest()})))

    check("lakefile.toml with no requires",
          not uses_mathlib(project({"lakefile.toml": 'name = "myproj"\n'})))

    check("lakefile.toml requiring something else",
          not uses_mathlib(project({"lakefile.toml": TOML_OTHER})))

    check("lakefile.lean requiring something else",
          not uses_mathlib(project({"lakefile.lean": LEAN_OTHER})))

    check("manifest and lakefile agreeing on no mathlib",
          not uses_mathlib(project({"lake-manifest.json": manifest("batteries"),
                                    "lakefile.toml": TOML_OTHER})))

    # Only a require/package line counts -- a passing mention is not a dependency.
    check("a stray mathlib mention outside a require does not count",
          not uses_mathlib(project({
              "lakefile.lean": "-- someday we might require mathlib here\npackage myproj\n"})))

    print("\n=== falls back to fetching when it cannot tell ===")

    check("no lake files at all", uses_mathlib(project()))

    check("unparseable manifest, no lakefile",
          uses_mathlib(project({"lake-manifest.json": "{not json"})))

    check("unparseable lakefile.toml, no manifest",
          uses_mathlib(project({"lakefile.toml": "name = [unterminated\n"})))

    # One unreadable source must not veto a readable one that says no.
    check("unparseable manifest but a clean lakefile.toml",
          not uses_mathlib(project({"lake-manifest.json": "{not json",
                                    "lakefile.toml": TOML_OTHER})))

    print("\nAll mathlib detection tests passed.")


if __name__ == "__main__":
    main()
