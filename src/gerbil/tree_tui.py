"""`gerbil tree`: a full-screen browser over the session forest.

Left: every archived session, grouped by project, linked into trees
(sessions_tree.build_forest -- agent parent/child, resume continuations,
ralph chains). Right: the selected session's details. Keys:

  r  resume the selected session (exits the viewer and runs `gerbil resume`)
  g  grab it, when it is a live run (exits and runs `gerbil grab`)
  n  name / rename it (empty clears the name)
  d  delete it and everything below it (confirmed first)
  R  refresh,  q  quit

A thin shell: all logic is in sessions_tree.py. Resume and grab are handed
back to the caller (cli.cmd_tree) as the app's return value, because both
take over the terminal with their own viewer.
"""

from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Input, Label, Static, Tree

from . import sessions_tree as st


# How deep the viewer nests (see TreeApp.reload). Far beyond any real tree,
# far below where textual's recursive line layout would hit Python's limit.
MAX_VIEW_DEPTH = 64


class NameScreen(ModalScreen[str | None]):
    """Ask for a name. Dismisses with the text, or None on escape."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]
    CSS = """
    NameScreen { align: center middle; }
    #box { width: 64; height: auto; border: round $accent; padding: 1 2; background: $surface; }
    """

    def __init__(self, stem: str, current: str):
        super().__init__()
        self.stem, self.current = stem, current

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Label(f"Name for {self.stem} (empty clears it):")
            yield Input(value=self.current, id="name")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    """Yes/no. Dismisses True only on an explicit yes."""

    BINDINGS = [Binding("escape", "no", "cancel"), Binding("y", "yes", "yes"),
                Binding("n", "no", "no")]
    CSS = """
    ConfirmScreen { align: center middle; }
    #box { width: 72; height: auto; border: round $error; padding: 1 2; background: $surface; }
    #buttons { height: auto; margin-top: 1; }
    """

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="box"):
            yield Static(Text(self.text))
            with Horizontal(id="buttons"):
                yield Button("Delete (y)", variant="error", id="yes")
                yield Button("Cancel (n)", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class TreeApp(App[tuple[str, str] | None]):
    """Returns ("resume", log path) / ("grab", run name) / None."""

    TITLE = "gerbil sessions"
    CSS = """
    #forest { width: 1fr; min-width: 40; border-right: solid $accent; }
    #details-scroll { width: 1fr; }
    #details { padding: 0 1; }
    #status { height: 1; padding: 0 1; color: $warning; }
    """
    BINDINGS = [
        Binding("r", "resume", "resume"),
        Binding("g", "grab", "grab"),
        Binding("n", "rename", "name"),
        Binding("d", "delete", "delete"),
        Binding("R", "refresh", "refresh"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self, base: Path | None = None, project: str | None = None,
                 live: dict[str, str] | None = None):
        super().__init__()
        self.base, self.project, self.live = base, project, live
        self.roots: list[st.Node] = []

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Tree("sessions", id="forest")
            with VerticalScroll(id="details-scroll"):
                yield Static("", id="details")
        yield Static("", id="status")
        yield Footer()

    def on_mount(self) -> None:
        self.reload()
        self.query_one("#forest", Tree).focus()
        if self.skipped:
            self.say(f"{len(self.skipped)} file(s) in the archive are not readable "
                     "session logs and were skipped: "
                     + ", ".join(p.name for p in self.skipped[:5])
                     + (" ..." if len(self.skipped) > 5 else ""))

    # -- model -> widgets -----------------------------------------------------

    def reload(self, select: str | None = None) -> None:
        skipped: list[Path] = []
        self.roots = st.build_forest(self.base, project=self.project,
                                     live=self.live, skipped=skipped)
        self.skipped = skipped
        tree = self.query_one("#forest", Tree)
        tree.clear()
        tree.root.expand()
        tree.show_root = False
        by_project: dict[str, list[st.Node]] = {}
        for r in self.roots:
            by_project.setdefault(r.project_dir or "?", []).append(r)
        target = None

        def add(parent_item, root: st.Node) -> None:
            # Iterative: session chains can be arbitrarily deep. And textual's
            # Tree lays its lines out recursively, so the DISPLAY nests at
            # most MAX_VIEW_DEPTH levels: deeper sessions are listed flat at
            # that level, tagged with their true depth. (The model -- and so
            # delete -- keeps the real structure.)
            nonlocal target
            stack = [(parent_item, root, 0)]
            while stack:
                parent, node, depth = stack.pop()
                label = escape(st.node_line(node))
                if depth >= MAX_VIEW_DEPTH:
                    label = f"[dim]⋯ depth {depth}[/dim] " + label
                nest = depth < MAX_VIEW_DEPTH
                if node.children and nest:
                    item = parent.add(label, data=node.stem, expand=True)
                else:
                    item = parent.add_leaf(label, data=node.stem)
                holder = item if nest else parent
                stack.extend((holder, c, depth + 1) for c in reversed(node.children))
                if node.stem == select:
                    target = item

        for project, rs in by_project.items():
            group = tree.root.add(f"[b]{escape(Path(project).name or project)}[/b]  "
                                  f"[dim]{escape(project)}[/dim]", data=None, expand=True)
            for r in rs:
                add(group, r)
        if not self.roots:
            tree.root.add_leaf("no sessions found", data=None)
        if target is not None:
            tree.move_cursor(target)
        self.show_details()

    def selected(self) -> st.Node | None:
        node = self.query_one("#forest", Tree).cursor_node
        stem = node.data if node is not None else None
        return st.find(self.roots, stem) if stem else None

    def show_details(self) -> None:
        n = self.selected()
        # Text, not a str: prompts and names are shown literally, never
        # parsed as markup.
        self.query_one("#details", Static).update(
            Text(st.describe(n) if n else "Select a session."))

    def say(self, text: str) -> None:
        self.query_one("#status", Static).update(Text(text))

    def on_tree_node_highlighted(self, _event) -> None:
        self.show_details()
        self.say("")

    # -- actions --------------------------------------------------------------

    def action_refresh(self) -> None:
        n = self.selected()
        self.reload(select=n.stem if n else None)
        self.say("refreshed")

    def action_resume(self) -> None:
        n = self.selected()
        if n is None:
            return
        ok, why = st.resumability(n)
        if not ok:
            self.say(f"can't resume: {why}")
            return
        self.exit(("resume", str(n.path)))

    def action_grab(self) -> None:
        n = self.selected()
        if n is None:
            return
        if not n.live_run:
            self.say("not a live run")
            return
        self.exit(("grab", n.live_run))

    def action_rename(self) -> None:
        n = self.selected()
        if n is None:
            return

        def done(value: str | None) -> None:
            if value is None:
                return
            st.set_label(n.stem, value, self.base)
            self.reload(select=n.stem)
            self.say(f"named {n.stem}" if value.strip() else f"cleared name of {n.stem}")

        self.push_screen(NameScreen(n.stem, n.label), done)

    def action_delete(self) -> None:
        n = self.selected()
        if n is None:
            return
        nodes = list(st.walk(n))
        files = st.subtree_files(n)
        text = (f"Delete {n.stem}"
                + (f" and the {len(nodes) - 1} session(s) below it" if len(nodes) > 1 else "")
                + f"?\n\n{len(files)} file(s) in {n.path.parent} will be removed "
                "permanently. Patches already in your project's .gerbil/patches/ "
                "and commits are not touched.")

        def done(yes: bool | None) -> None:
            if not yes:
                self.say("delete cancelled")
                return
            try:
                removed = st.delete_subtree(n, self.base)
            except ValueError as e:
                self.say(f"can't delete: {e}")
                return
            self.reload()
            self.say(f"deleted {len(nodes)} session(s), {len(removed)} file(s)")

        self.push_screen(ConfirmScreen(text), done)
