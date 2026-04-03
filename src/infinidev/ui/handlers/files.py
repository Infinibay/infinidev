"""File tab and explorer management for the TUI.

Owns all state related to open file editors, the directory tree explorer,
tab switching, and the in-file search bar. Uses the Facade pattern to
present a clean interface to the app while encapsulating prompt_toolkit
Window lifecycle internals.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


class FileManager:
    """Manages file editor tabs, the directory tree, and in-file search.

    Owns stable Window references for each editor tab and the tree explorer.
    The app delegates all file operations here and reads tab state for rendering.
    """

    def __init__(self, app: InfinidevApp) -> None:
        self._app = app

        # File tab state
        self.open_files: dict[str, str] = {}       # tab_id → file_path
        self.tab_names: dict[str, str] = {}         # tab_id → display name
        self.dirty_files: set[str] = set()          # dirty tab_ids
        self.editors: dict[str, Any] = {}            # tab_id → FileEditor
        self.editor_windows: dict[str, Any] = {}    # tab_id → stable Window

        # Explorer state
        self._tree_control: Any = None
        self._tree_window: Any = None

        # Placeholder until tree initializes
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.controls import FormattedTextControl
        from infinidev.ui.theme import TEXT_MUTED
        self._explorer_placeholder = Window(
            content=FormattedTextControl(lambda: [(f"{TEXT_MUTED}", " Press Ctrl+E to load explorer")]),
        )

        # Search bar
        from infinidev.ui.controls.search_bar import SearchBarState
        self.search_bar = SearchBarState()

    # ── Explorer ────────────────────────────────────────────────────

    def toggle_explorer(self) -> None:
        """Toggle the file explorer sidebar."""
        app = self._app
        app.explorer_visible = not app.explorer_visible
        if app.explorer_visible:
            if self._tree_control is None:
                self._init_tree()
            if self._tree_window is not None:
                try:
                    app.app.layout.focus(self._tree_window)
                except Exception:
                    pass
        else:
            app.focus_chat()
        app.invalidate()

    def _init_tree(self) -> None:
        """Lazy-initialize the directory tree on first explorer open."""
        from prompt_toolkit.layout.containers import Window
        from infinidev.ui.controls.directory_tree import DirectoryTreeControl
        self._tree_control = DirectoryTreeControl(
            root_path=os.getcwd(),
            on_file_selected=self._on_file_selected,
        )
        self._tree_window = Window(content=self._tree_control)

    def _on_file_selected(self, file_path: str) -> None:
        self.open_file(file_path)

    def get_explorer_content(self) -> Any:
        """Return the stable explorer Window (same instance every call)."""
        if self._tree_window is not None:
            return self._tree_window
        return self._explorer_placeholder

    # ── File tabs ───────────────────────────────────────────────────

    def open_file(self, file_path: str) -> None:
        """Open a file in a new editor tab (or switch to existing)."""
        from prompt_toolkit.layout.containers import (
            Window as _W, HSplit as _H, ConditionalContainer as _CC,
        )
        from prompt_toolkit.layout.controls import FormattedTextControl as _FTC
        from prompt_toolkit.filters import Condition as _Cond
        from infinidev.ui.controls.file_editor import FileEditor

        app = self._app
        tab_id = file_path
        name = os.path.basename(file_path)

        if tab_id in self.editors:
            app.active_tab = tab_id
            try:
                app.app.layout.focus(self.editor_windows[tab_id])
            except Exception:
                pass
            app.invalidate()
            return

        editor = FileEditor(file_path, on_dirty_change=self._on_dirty_change)
        self.editors[tab_id] = editor
        self.open_files[tab_id] = file_path
        self.tab_names[tab_id] = name

        search = self.search_bar
        editor_window = _H([
            _CC(
                content=_H([
                    _W(content=search.search_control, height=1),
                    _W(content=_FTC(lambda: search.get_status_fragments()), height=1),
                ]),
                filter=_Cond(lambda tid=tab_id: search.visible and app.active_tab == tid),
            ),
            _W(content=editor.control),
        ])
        self.editor_windows[tab_id] = editor_window

        app.active_tab = tab_id
        try:
            app.app.layout.focus(editor_window)
        except Exception:
            pass
        app.invalidate()

    def close_active_tab(self) -> None:
        """Close the active file tab."""
        app = self._app
        tab_id = app.active_tab
        if tab_id == "chat":
            return

        editor = self.editors.get(tab_id)
        if editor and editor.is_dirty:
            app.add_message("System", f"Unsaved changes in {editor.name}. Save first (Ctrl+S).", "system")
            return

        self.editors.pop(tab_id, None)
        self.editor_windows.pop(tab_id, None)
        self.open_files.pop(tab_id, None)
        self.tab_names.pop(tab_id, None)
        self.dirty_files.discard(tab_id)

        remaining = list(self.tab_names.keys())
        app.active_tab = remaining[-1] if remaining else "chat"
        if app.active_tab == "chat":
            app.focus_chat()
        app.invalidate()

    def save_active_file(self) -> None:
        """Save the active file editor."""
        app = self._app
        tab_id = app.active_tab
        editor = self.editors.get(tab_id)
        if not editor:
            return
        if editor.save():
            app.add_message("System", f"Saved: {editor.name}", "system")
        else:
            app.add_message("System", f"Failed to save: {editor.name}", "system")
        app.invalidate()

    def toggle_search_bar(self) -> None:
        """Toggle the in-file search bar for the active editor."""
        app = self._app
        tab_id = app.active_tab
        editor = self.editors.get(tab_id)
        if not editor:
            return
        self.search_bar.set_target(editor.buffer)
        self.search_bar.toggle()
        if self.search_bar.visible:
            try:
                app.app.layout.focus(self.search_bar.search_control)
            except Exception:
                pass
        app.invalidate()

    def switch_tab(self, tab_id: str) -> None:
        """Switch to a tab and focus its content."""
        app = self._app
        app.active_tab = tab_id
        if tab_id == "chat":
            app.focus_chat()
        else:
            editor_window = self.editor_windows.get(tab_id)
            if editor_window:
                try:
                    app.app.layout.focus(editor_window)
                except Exception:
                    pass
        app.invalidate()

    def _on_dirty_change(self, tab_id: str, dirty: bool) -> None:
        """Track dirty state for tab indicators."""
        if dirty:
            self.dirty_files.add(tab_id)
        else:
            self.dirty_files.discard(tab_id)
        self._app.invalidate()
