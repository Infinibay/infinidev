"""Dialog lifecycle management for the TUI.

Owns lazy initialization, state, and stable Window references for modal
dialogs (settings editor, findings browser). Uses the Lazy Initialization
pattern — dialogs are created on first open and reused thereafter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


class DialogManager:
    """Manages modal dialogs: settings editor, findings browser.

    Each dialog is lazy-initialized on first open. Stable Window references
    are created once so prompt_toolkit focus management works correctly.
    """

    def __init__(self, app: InfinidevApp) -> None:
        self._app = app

        # Settings dialog state
        self._settings_state: Any = None
        self._settings_sections_ctrl_window: Any = None
        self._settings_settings_ctrl_window: Any = None
        self._settings_dropdown_window: Any = None
        self._settings_edit_buffer_window: Any = None

        # Findings dialog state
        self._findings_list_ctrl: Any = None
        self._findings_list_window: Any = None
        self._findings_detail_window: Any = None

    # ── Settings dialog ─────────────────────────────────────────────

    def open_settings(self) -> None:
        """Open the settings editor modal (lazy-inits on first call)."""
        if self._settings_state is None:
            self._init_settings_dialog()
        self._app.active_dialog = "settings_editor"
        self._settings_state.focus_panel = "sections"
        self._settings_state.section_cursor = 0
        self._settings_state.setting_cursor = 0
        self._settings_state.editing = False
        try:
            self._app.app.layout.focus(self._settings_sections_ctrl_window)
        except Exception:
            pass
        self._app.invalidate()

    def _init_settings_dialog(self) -> None:
        """Create the settings dialog and register as a Float."""
        from prompt_toolkit.layout.containers import (
            Float, ConditionalContainer, HSplit, VSplit, Window,
        )
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.layout.dimension import Dimension as D
        from prompt_toolkit.filters import Condition
        from infinidev.ui.dialogs.settings_editor import (
            create_settings_editor, DropdownControl,
        )
        from infinidev.ui.dialogs.base import dialog_frame
        from infinidev.ui.theme import PRIMARY, SURFACE, SURFACE_LIGHT

        app = self._app

        def _on_focus_change(panel: str) -> None:
            try:
                if panel == "sections":
                    app.app.layout.focus(self._settings_sections_ctrl_window)
                elif panel == "dropdown":
                    app.app.layout.focus(self._settings_dropdown_window)
                else:
                    app.app.layout.focus(self._settings_settings_ctrl_window)
            except Exception:
                pass
            app.invalidate()

        def _on_edit_start() -> None:
            try:
                app.app.layout.focus(self._settings_edit_buffer_window)
            except Exception:
                pass
            app.invalidate()

        frame, state, sections_ctrl, settings_ctrl = create_settings_editor(
            on_save=lambda k, v: app.invalidate(),
            on_focus_change=_on_focus_change,
            on_edit_start=_on_edit_start,
        )
        self._settings_state = state

        # Stable windows for focus management
        self._settings_sections_ctrl_window = Window(content=sections_ctrl, width=16)
        self._settings_settings_ctrl_window = Window(content=settings_ctrl)

        self._settings_edit_buffer_window = Window(
            content=BufferControl(buffer=state.edit_buffer, focusable=True),
            height=1,
            style=f"bg:{SURFACE_LIGHT} #ffffff",
        )

        dropdown_ctrl = DropdownControl(state)
        self._settings_dropdown_window = Window(
            content=dropdown_ctrl,
            style=f"bg:{SURFACE}",
        )

        settings_area = HSplit([
            ConditionalContainer(
                content=self._settings_dropdown_window,
                filter=Condition(lambda: state.dropdown_open),
            ),
            ConditionalContainer(
                content=HSplit([
                    self._settings_settings_ctrl_window,
                    ConditionalContainer(
                        content=self._settings_edit_buffer_window,
                        filter=Condition(lambda: state.editing),
                    ),
                ]),
                filter=Condition(lambda: not state.dropdown_open),
            ),
        ], width=D(weight=1))

        body = VSplit([
            self._settings_sections_ctrl_window,
            Window(width=1, char="│", style=f"{PRIMARY}"),
            settings_area,
        ])

        actual_frame = dialog_frame("Settings", body, width=90, height=30, border_color=PRIMARY)

        dialog_float = Float(
            content=ConditionalContainer(
                content=actual_frame,
                filter=Condition(lambda: app.active_dialog == "settings_editor"),
            ),
            transparent=False,
        )
        if app._float_container:
            app._float_container.floats.append(dialog_float)

    # ── Findings dialog ─────────────────────────────────────────────

    def open_findings(self, filter_type: str | None = None) -> None:
        """Open the findings browser modal (lazy-inits on first call)."""
        if self._findings_list_ctrl is None:
            self._init_findings_dialog()

        from infinidev.db.service import get_all_findings
        try:
            findings = get_all_findings()
        except Exception:
            findings = []
        if filter_type:
            findings = [f for f in findings if f.get("finding_type") == filter_type]

        self._findings_list_ctrl.findings = findings
        self._findings_list_ctrl.cursor = 0
        self._app.active_dialog = "findings_browser"

        try:
            self._app.app.layout.focus(self._findings_list_window)
        except Exception:
            pass
        self._app.invalidate()

    def _init_findings_dialog(self) -> None:
        """Create the findings dialog and register as a Float."""
        from prompt_toolkit.layout.containers import (
            Float, ConditionalContainer, VSplit, Window,
        )
        from prompt_toolkit.layout.dimension import Dimension as D
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.mouse_events import MouseEventType
        from infinidev.ui.dialogs.findings_browser import (
            FindingsListControl, FindingsDetailControl,
        )
        from infinidev.ui.dialogs.base import dialog_frame
        from infinidev.ui.theme import PRIMARY

        app = self._app

        list_ctrl = FindingsListControl()
        detail_ctrl = FindingsDetailControl(list_ctrl)
        self._findings_list_ctrl = list_ctrl

        self._findings_list_window = Window(content=list_ctrl, width=D(weight=40))
        self._findings_detail_window = Window(content=detail_ctrl, width=D(weight=60))

        # Navigation keybindings
        nav_kb = KeyBindings()

        @nav_kb.add("up")
        def _up(event):
            list_ctrl.move_cursor(-1)

        @nav_kb.add("down")
        def _down(event):
            list_ctrl.move_cursor(1)

        @nav_kb.add("escape")
        def _close(event):
            app.active_dialog = None
            app.focus_chat()
            app.invalidate()

        list_ctrl._nav_kb = nav_kb
        list_ctrl.get_key_bindings = lambda: nav_kb
        list_ctrl.is_focusable = lambda: True

        def _mouse(mouse_event):
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                row = mouse_event.position.y
                if 0 <= row < len(list_ctrl.findings):
                    list_ctrl.cursor = row
            return None
        list_ctrl.mouse_handler = _mouse

        body = VSplit([
            self._findings_list_window,
            Window(width=1, char="│", style=f"{PRIMARY}"),
            self._findings_detail_window,
        ])

        frame = dialog_frame("Findings", body, width=90, height=30, border_color=PRIMARY)

        dialog_float = Float(
            content=ConditionalContainer(
                content=frame,
                filter=Condition(lambda: app.active_dialog == "findings_browser"),
            ),
            transparent=False,
        )
        if app._float_container:
            app._float_container.floats.append(dialog_float)
