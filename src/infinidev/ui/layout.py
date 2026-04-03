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
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.layout import Layout

from infinidev.ui.theme import (
    EXPLORER_WIDTH,
    CHAT_INPUT_HEIGHT,
    STYLE_SIDEBAR_TITLE,
    PRIMARY,
    SURFACE_DARK,
    SURFACE_LIGHT,
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
    chat_input_area = ConditionalContainer(
        content=Window(
            content=app_state._chat_input_control,
            height=CHAT_INPUT_HEIGHT,
            style=f"bg:{SURFACE_LIGHT}",
        ),
        filter=Condition(lambda: app_state.active_tab == "chat"),
    )

    # ── Sidebar (right) ─────────────────────────────────────────────

    sidebar_border = Window(width=1, char="│", style=f"{PRIMARY}")

    def _sidebar_section(title: str, content_getter, scrollable: bool = False):
        from prompt_toolkit.layout.margins import ScrollbarMargin
        margins = [ScrollbarMargin()] if scrollable else []
        return HSplit([
            Window(
                content=FormattedTextControl(lambda t=title: [
                    (STYLE_SIDEBAR_TITLE, f" {t} "),
                ]),
                height=1,
            ),
            Window(
                content=FormattedTextControl(content_getter),
                height=D(min=2, max=15, preferred=4),
                style=f"bg:{SURFACE_LIGHT}",
                wrap_lines=True,
                right_margins=margins,
            ),
        ])

    context_section = _sidebar_section("CONTEXT", lambda: app_state.get_context_fragments())
    plan_section = _sidebar_section("THINKING", lambda: app_state.get_plan_fragments(), scrollable=True)
    steps_section = _sidebar_section("STEPS", lambda: app_state.get_steps_fragments(), scrollable=True)
    actions_section = _sidebar_section("ACTIONS", lambda: app_state.get_actions_fragments())
    logs_section = _sidebar_section("LOGS", lambda: app_state.get_logs_fragments())

    # ── Assemble 3-column layout ────────────────────────────────────

    main_body = VSplit([
        explorer_panel,
        explorer_border,
        # Content takes 70% of remaining space
        HSplit(
            [
                tab_bar,
                content_body,
                chat_input_area,
            ],
            width=D(weight=70),
        ),
        sidebar_border,
        # Sidebar takes 30%
        HSplit(
            [
                context_section,
                plan_section,
                steps_section,
                actions_section,
                logs_section,
                Window(),  # spacer
            ],
            width=D(weight=30),
        ),
    ])

    # ── Status bar + footer ─────────────────────────────────────────

    app_state.status_bar_control = StatusBarControl()
    app_state.footer_control = FooterControl()

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

    return Layout(root)
