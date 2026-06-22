"""Tiny ANSI color helper.

Colors are emitted only when stdout is a TTY and NO_COLOR is unset, so piping
output to a file or pager stays clean. style() is a no-op when disabled.
"""

import os
import sys

# https://no-color.org/ -- any non-empty NO_COLOR disables color.
ENABLED = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

_CODES = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "dark_red": "\033[38;5;88m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "gray": "\033[90m",
}


def style(text: str, *names: str) -> str:
    """Wrap text in the given styles (e.g. style("hi", "bold", "cyan"))."""
    if not ENABLED or not names:
        return text
    prefix = "".join(_CODES[n] for n in names)
    return f"{prefix}{text}{_CODES['reset']}"
