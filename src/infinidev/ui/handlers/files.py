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

        # Line numbers toggle
        self.line_numbers_visible: bool = False

        # Quick file picker (Ctrl+P) — lazy-initialized
        self._file_picker: Any = None
        self._file_picker_window: Any = None
        self._file_picker_input_window: Any = None

        # File watcher for live explorer updates
        self._file_watcher: Any = None

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
        from prompt_toolkit.layout.margins import ScrollbarMargin
        self._tree_window = Window(
            content=self._tree_control,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        # Start file watcher for live updates
        self._start_file_watcher()

    def _start_file_watcher(self) -> None:
        """Start background file watcher to refresh the tree on changes."""
        if self._file_watcher is not None:
            return
        try:
            from infinidev.cli.file_watcher import FileWatcher, WATCHFILES_AVAILABLE
            if not WATCHFILES_AVAILABLE:
                return

            def _on_change(path: str) -> None:
                if self._tree_control is not None:
                    self._tree_control.refresh()
                # Also invalidate the file picker cache
                if self._file_picker is not None:
                    self._file_picker.refresh()
                self._app.invalidate()

            cwd = os.getcwd()
            self._file_watcher = FileWatcher(
                workspace=cwd,
                callback=_on_change,
                # Return workspace root as always-visible so all changes trigger refresh
                visible_paths_callback=lambda: {cwd},
            )
            self._file_watcher.start()
        except Exception:
            pass  # Non-critical — explorer works without live updates

    def stop_file_watcher(self) -> None:
        """Stop the file watcher (call on app exit)."""
        if self._file_watcher is not None:
            self._file_watcher.stop()
            self._file_watcher = None

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
        from prompt_toolkit.layout.margins import NumberedMargin

        editor_content_window = _W(
            content=editor.control,
            left_margins=[NumberedMargin()] if self.line_numbers_visible else [],
        )
        # Store ref so toggle_line_numbers can update margins later
        editor._content_window = editor_content_window

        editor_window = _H([
            _CC(
                content=_H([
                    _W(content=search.search_control, height=1),
                    _W(content=_FTC(lambda: search.get_status_fragments()), height=1),
                ]),
                filter=_Cond(lambda tid=tab_id: search.visible and app.active_tab == tid),
            ),
            editor_content_window,
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

    # ── Line numbers (Ctrl+L) ──────────────────────────────────────

    def toggle_line_numbers(self) -> None:
        """Toggle line number margin on all open editor windows."""
        from prompt_toolkit.layout.margins import NumberedMargin

        self.line_numbers_visible = not self.line_numbers_visible

        for editor in self.editors.values():
            window = getattr(editor, '_content_window', None)
            if window is None:
                continue
            if self.line_numbers_visible:
                window.left_margins = [NumberedMargin()]
            else:
                window.left_margins = []

        self._app.invalidate()

    # ── Quick file picker (Ctrl+P) ─────────────────────────────────

    def _ensure_file_picker(self) -> None:
        """Lazy-init the file picker and register its Float."""
        if self._file_picker is not None:
            return

        from prompt_toolkit.layout.containers import (
            Float, ConditionalContainer, HSplit, Window,
        )
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.layout.dimension import Dimension as D
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from infinidev.ui.controls.file_picker import FilePickerState
        from infinidev.ui.dialogs.base import dialog_frame
        from infinidev.ui.theme import PRIMARY, SURFACE, TEXT_MUTED

        app = self._app
        picker = FilePickerState(
            root=os.getcwd(),
            on_select=self.open_file,
        )
        self._file_picker = picker

        # Input keybindings (up/down/enter/escape within the search field)
        picker_kb = KeyBindings()

        @picker_kb.add("up")
        def _up(event):
            picker.results_control.move_cursor(-1)
            app.invalidate()

        @picker_kb.add("down")
        def _down(event):
            picker.results_control.move_cursor(1)
            app.invalidate()

        @picker_kb.add("enter")
        def _select(event):
            picker.select_current()
            app.active_dialog = None
            app.invalidate()

        @picker_kb.add("escape")
        def _close(event):
            picker.close()
            app.active_dialog = None
            app.focus_chat()
            app.invalidate()

        # Stable windows
        self._file_picker_input_window = Window(
            content=BufferControl(
                buffer=picker.search_buffer,
                key_bindings=picker_kb,
                focusable=True,
            ),
            height=1,
            style=f"bg:{SURFACE} #ffffff bold",
        )

        self._file_picker_window = Window(
            content=picker.results_control,
        )

        hint = Window(
            content=__import__('prompt_toolkit').layout.controls.FormattedTextControl(
                lambda: [(f"{TEXT_MUTED}", " Type to filter  |  ↑↓ Navigate  |  Enter Open  |  Esc Close")]
            ),
            height=1,
        )

        body = HSplit([
            self._file_picker_input_window,
            self._file_picker_window,
            hint,
        ])

        frame = dialog_frame("Open File", body, width=70, height=22, border_color=PRIMARY)

        dialog_float = Float(
            content=ConditionalContainer(
                content=frame,
                filter=Condition(lambda: app.active_dialog == "file_picker"),
            ),
            transparent=False,
        )
        if app._float_container:
            app._float_container.floats.append(dialog_float)

    def open_file_picker(self) -> None:
        """Show the quick file picker dialog (Ctrl+P)."""
        self._ensure_file_picker()
        self._file_picker.open()
        self._app.active_dialog = "file_picker"
        try:
            self._app.app.layout.focus(self._file_picker_input_window)
        except Exception:
            pass
        self._app.invalidate()
