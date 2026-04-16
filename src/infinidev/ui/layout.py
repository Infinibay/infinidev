"""Top-level layout construction for the Infinidev TUI.

Builds the full-screen layout: explorer | content (tabs) | sidebar,
with a FloatContainer layer on top for modal dialogs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.layout import Layout

from infinidev.ui.controls.scrollable_text import ScrollableTextControl
from infinidev.ui.theme import (
    EXPLORER_WIDTH,
    CHAT_INPUT_HEIGHT,
    STYLE_SIDEBAR_TITLE,
    PRIMARY,
    SURFACE_DARK,
    SURFACE_LIGHT,
    SHELL_INPUT_BG,
    SHELL_INPUT_FG,
    SHELL_BORDER_COLOR,
    SHELL_LABEL_FG,
    TEXT_MUTED,
)
from infinidev.ui.controls.status_bar import StatusBarControl, FooterControl

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


def build_layout(app_state: InfinidevApp) -> Layout:
    """Construct the full application layout.

    Returns a Layout whose root is a FloatContainer (for dialog overlays).
    """

    # ── Explorer panel (left) ───────────────────────────────────────

    explorer_title = Window(
        content=FormattedTextControl(lambda: [
            (f"#ffffff bg:{PRIMARY} bold", " EXPLORER "),
        ]),
        height=1,
    )

    explorer_body = DynamicContainer(lambda: app_state.get_explorer_content())

    explorer_panel = ConditionalContainer(
        content=HSplit([
            explorer_title,
            explorer_body,
        ], width=D(preferred=EXPLORER_WIDTH)),
        filter=Condition(lambda: app_state.explorer_visible),
    )

    explorer_border = ConditionalContainer(
        content=Window(width=1, char="│", style=f"{PRIMARY}"),
        filter=Condition(lambda: app_state.explorer_visible),
    )

    # ── Content area (center) ───────────────────────────────────────

    tab_bar = Window(
        content=FormattedTextControl(lambda: app_state.get_tab_bar_fragments()),
        height=1,
    )

    content_body = DynamicContainer(lambda: app_state.get_active_content())

    # Chat input — uses the real Buffer from the app
    # Shell mode (! prefix): red border line + dark red-tinted background
    def _is_shell_mode() -> bool:
        return app_state._chat_buffer.text.startswith("!")

    def _chat_input_style() -> str:
        if _is_shell_mode():
            return f"fg:{SHELL_INPUT_FG} bg:{SHELL_INPUT_BG}"
        return f"bg:{SURFACE_LIGHT}"

    shell_border_line = ConditionalContainer(
        content=Window(
            content=FormattedTextControl(lambda: [
                (f"{SHELL_LABEL_FG} bg:{SHELL_INPUT_BG} bold", " SHELL "),
                (f"{SHELL_BORDER_COLOR} bg:{SHELL_INPUT_BG}", " ─" * 40),
            ]),
            height=1,
            style=f"bg:{SHELL_INPUT_BG}",
        ),
        filter=Condition(_is_shell_mode),
    )

    chat_input_area = ConditionalContainer(
        content=HSplit([
            shell_border_line,
            Window(
                content=app_state._chat_input_control,
                height=CHAT_INPUT_HEIGHT,
                style=_chat_input_style,
            ),
        ]),
        filter=Condition(lambda: app_state.active_tab == "chat"),
    )

    # ── Sidebar (right) ─────────────────────────────────────────────

    def _sidebar_section(title: str, content_getter, scrollable: bool = False):
        if scrollable:
            control = ScrollableTextControl(content_getter)
        else:
            control = FormattedTextControl(content_getter)

        title_win = Window(
            content=FormattedTextControl(lambda t=title: [
                (STYLE_SIDEBAR_TITLE, f" {t} "),
            ]),
            height=1,
        )

        if scrollable:
            from infinidev.ui.controls.clickable_scrollbar import scrollable_window
            _, content_container = scrollable_window(
                control, display_arrows=False,
                height=D(min=2, max=15, preferred=4),
                style=f"bg:{SURFACE_LIGHT}",
                wrap_lines=True,
            )
            return HSplit([title_win, content_container])
        else:
            return HSplit([title_win, Window(
                content=control,
                height=D(min=2, max=15, preferred=4),
                style=f"bg:{SURFACE_LIGHT}",
                wrap_lines=True,
            )])

    context_section = _sidebar_section("CONTEXT", lambda: app_state.get_context_fragments())
    plan_section = _sidebar_section("THINKING", lambda: app_state.get_plan_fragments(), scrollable=True)
    steps_section = _sidebar_section("STEPS", lambda: app_state.get_steps_fragments(), scrollable=True)
    actions_section = _sidebar_section("ACTIONS", lambda: app_state.get_actions_fragments())
    logs_section = _sidebar_section("LOGS", lambda: app_state.get_logs_fragments())

    # ── Assemble 3-column layout ────────────────────────────────────

    # Sidebar toggle indicator — shows when sidebar is hidden
    from prompt_toolkit.mouse_events import MouseEventType

    def _sidebar_toggle_click(mouse_event):
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            app_state.toggle_sidebar()

    def _sidebar_hide_click(mouse_event):
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            app_state.toggle_sidebar()

    def _sidebar_indicator():
        if app_state.sidebar_visible:
            return FormattedText([])
        return FormattedText([
            (f"{PRIMARY} bold", " ◆ ", _sidebar_toggle_click),
        ])

    sidebar_toggle_indicator = ConditionalContainer(
        content=Window(
            content=FormattedTextControl(_sidebar_indicator),
            width=3,
            style=f"bg:{SURFACE_DARK}",
        ),
        filter=Condition(lambda: not app_state.sidebar_visible),
    )

    sidebar_border_conditional = ConditionalContainer(
        content=Window(width=1, char="│", style=f"{PRIMARY}"),
        filter=Condition(lambda: app_state.sidebar_visible),
    )

    # Hide button — appears at top of sidebar when visible
    sidebar_hide_button = Window(
        content=FormattedTextControl(lambda: FormattedText([
            ("", "  "),
            (f"{PRIMARY} bold", "◆", _sidebar_hide_click),
        ])),
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )

    sidebar_content = ConditionalContainer(
        content=HSplit(
            [
                sidebar_hide_button,
                context_section,
                plan_section,
                steps_section,
                actions_section,
                logs_section,
                Window(),  # spacer
            ],
            width=D(weight=30),
        ),
        filter=Condition(lambda: app_state.sidebar_visible),
    )

    main_body = VSplit([
        explorer_panel,
        explorer_border,
        # Content takes remaining space
        HSplit(
            [
                tab_bar,
                content_body,
                chat_input_area,
            ],
            width=D(weight=70),
        ),
        sidebar_border_conditional,
        sidebar_content,
        sidebar_toggle_indicator,
    ])

    # ── Status bar + footer ─────────────────────────────────────────

    app_state.status_bar_control = StatusBarControl()
    app_state.footer_control = FooterControl(app_state)

    status_bar = Window(
        content=app_state.status_bar_control,
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )

    footer = Window(
        content=app_state.footer_control,
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )

    # ── Root: float container (for dialogs) wrapping full layout ────

    root = FloatContainer(
        content=HSplit([
            main_body,
            status_bar,
            footer,
        ]),
        floats=[],  # Dialogs added in Phase 8
    )

    app_state._float_container = root

    return Layout(root, focused_element=app_state._chat_input_control)
