"""Top-level layout construction for the Infinidev TUI.

**Transcript-first.** The conversation owns the full width of the
terminal; the composer sits under it in a rounded frame; one status line
closes the screen. The explorer and the sidebar still exist with all
their panels — they are toggles (Ctrl+B / Alt+.), not permanent columns.

That is the whole design change from the previous three-column layout:
side panels that are always open cost ~60% of the width on a standard
terminal and turn a conversation into an IDE chrome demo. Modern coding
CLIs put the transcript in the middle of the screen and let the user pull
in panels on demand; this does the same without dropping a single panel.

    ┌──────────────────────────────────────────┐
    │  ▌ user message                          │   ← full-width transcript
    │  ● assistant reply                       │
    │    ⏵ 3 tools                             │
    │ ╭──────────────────────────────────────╮ │   ← composer
    │ │ › ask anything…                      │ │
    │ ╰──────────────────────────────────────╯ │
    │  model · branch · 82% context     ? help │   ← single status line
    └──────────────────────────────────────────┘
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
    SIDEBAR_WIDTH,
    SURFACE_DARK,
    SURFACE_LIGHT,
    TEXT_DIM,
    TEXT_MUTED,
)

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


def build_layout(app_state: InfinidevApp) -> Layout:
    """Construct the full application layout.

    Returns a Layout whose root is a FloatContainer (for dialog overlays).
    """

    # ── Explorer panel (left) ───────────────────────────────────────

    explorer_title = Window(
        content=FormattedTextControl(lambda: [
            (f"{TEXT_MUTED} bold", " FILES "),
        ]),
        height=1,
    )

    explorer_body = DynamicContainer(lambda: app_state.get_explorer_content())

    explorer_panel = ConditionalContainer(
        content=HSplit(
            [explorer_title, explorer_body],
            # Fixed, not proportional: a file tree needs the width of its
            # deepest path, not a third of the terminal.
            width=D(min=18, max=EXPLORER_WIDTH, preferred=EXPLORER_WIDTH),
        ),
        filter=Condition(lambda: app_state.explorer_visible),
    )

    explorer_border = ConditionalContainer(
        content=Window(width=1, char="│", style=f"{TEXT_DIM}"),
        filter=Condition(lambda: app_state.explorer_visible),
    )

    # ── Content area (center) ───────────────────────────────────────

    # The tab bar earns its row only when there is more than one tab —
    # otherwise it is a permanently-lit label above every conversation.
    tab_bar = ConditionalContainer(
        content=Window(
            content=FormattedTextControl(lambda: app_state.get_tab_bar_fragments()),
            height=1,
            style=f"bg:{SURFACE_LIGHT}",
        ),
        filter=Condition(lambda: bool(getattr(app_state, "_tab_names", None))),
    )

    content_body = DynamicContainer(lambda: app_state.get_active_content())

    # Chat input — rounded frame, inline placeholder, shell-mode aware.
    from infinidev.ui.controls.composer import composer_container

    chat_input_area = composer_container(app_state, app_state._chat_input_control)

    # ── Sidebar (right) ─────────────────────────────────────────────

    def _sidebar_section(
        title: str,
        content_getter,
        scrollable: bool = False,
        *,
        max_height: int = 10,
        visible=None,
    ):
        """One titled block in the sidebar.

        Titles are a dim label with a rule, not a full-width bar of solid
        colour — five saturated headers stacked down the right edge pulled
        the eye away from the conversation, which is the opposite of what a
        secondary panel should do. Sections size to their content and can
        hide entirely when they have nothing to say.
        """
        control = (
            ScrollableTextControl(content_getter)
            if scrollable
            else FormattedTextControl(content_getter)
        )

        title_win = VSplit(
            [
                Window(
                    content=FormattedTextControl(
                        lambda t=title: [(f"{TEXT_MUTED} bold", f" {t} ")]
                    ),
                    width=len(title) + 2,
                ),
                Window(char="─", style=f"{TEXT_DIM}"),
            ],
            height=1,
        )

        def _height(getter=content_getter, cap=max_height):
            """Size the section to its content, capped.

            A plain ``D(min=1, max=cap)`` lets the enclosing HSplit hand
            each section a share of the leftover space, which spreads four
            short sections across the whole panel with ragged gaps between
            them. Measuring the fragments keeps every block tight.
            """
            try:
                text = "".join(fragment[1] for fragment in getter())
            except Exception:
                return D(min=1, max=cap, preferred=1)
            if not text.strip():
                return D(min=1, max=cap, preferred=1)
            # A trailing newline terminates the last line, it does not add
            # an empty one — counting it reserved a blank row per section.
            lines = text.count("\n") + (0 if text.endswith("\n") else 1)
            return D(min=1, max=cap, preferred=max(1, min(lines, cap)))

        if scrollable:
            from infinidev.ui.controls.clickable_scrollbar import scrollable_window
            _, content_container = scrollable_window(
                control, display_arrows=False,
                height=_height,
                wrap_lines=True,
            )
            body = content_container
        else:
            body = Window(
                content=control,
                height=_height,
                dont_extend_height=True,
                wrap_lines=True,
            )

        section = HSplit([title_win, body, Window(height=1)])
        if visible is None:
            return section
        return ConditionalContainer(content=section, filter=Condition(visible))

    def _has(getter) -> bool:
        try:
            fragments = getter()
        except Exception:
            return False
        return any(text.strip() for _, text, *_ in fragments)

    context_section = _sidebar_section(
        "CONTEXT", lambda: app_state.get_context_fragments(), max_height=6
    )
    plan_section = _sidebar_section(
        "THINKING", lambda: app_state.get_plan_fragments(), scrollable=True,
        max_height=8, visible=lambda: _has(app_state.get_plan_fragments),
    )
    steps_section = _sidebar_section(
        "STEPS", lambda: app_state.get_steps_fragments(), scrollable=True,
        max_height=12, visible=lambda: _has(app_state.get_steps_fragments),
    )
    actions_section = _sidebar_section(
        "ACTIVITY", lambda: app_state.get_actions_fragments(), max_height=4,
        visible=lambda: _has(app_state.get_actions_fragments),
    )
    files_section = _sidebar_section(
        "FILES CHANGED", lambda: app_state.get_files_fragments(), max_height=10,
        visible=lambda: _has(app_state.get_files_fragments),
    )
    logs_section = _sidebar_section(
        "LOGS", lambda: app_state.get_logs_fragments(), max_height=6,
        visible=lambda: _has(app_state.get_logs_fragments),
    )

    # ── Assemble 3-column layout ────────────────────────────────────

    # Sidebar toggle indicator — shows when sidebar is hidden
    from prompt_toolkit.mouse_events import MouseEventType

    def _sidebar_hide_click(mouse_event):
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            app_state.toggle_sidebar()

    # NOTE: the old always-on ◆ toggle column is gone. It cost three
    # columns of transcript width on every screen just to advertise a
    # keybinding that Alt+. (and `?`) already document, and a closed
    # sidebar should cost nothing at all.

    sidebar_border_conditional = ConditionalContainer(
        content=Window(width=1, char="│", style=f"{TEXT_DIM}"),
        filter=Condition(lambda: app_state.sidebar_visible),
    )

    # Header row: what the panel is, and how to put it away.
    sidebar_header = Window(
        content=FormattedTextControl(lambda: FormattedText([
            (f"{TEXT_MUTED} bold", " SESSION"),
            (f"{TEXT_DIM}", "   alt+. to hide", _sidebar_hide_click),
        ])),
        height=1,
    )

    sidebar_content = ConditionalContainer(
        content=HSplit(
            [
                sidebar_header,
                Window(height=1),
                context_section,
                actions_section,
                steps_section,
                files_section,
                plan_section,
                logs_section,
                Window(),  # spacer pushes the sections to the top
            ],
            # Fixed width: proportional sizing gave the panel 60 columns on
            # a wide terminal to show four-word status lines.
            width=D(min=24, max=SIDEBAR_WIDTH, preferred=SIDEBAR_WIDTH),
        ),
        filter=Condition(lambda: app_state.sidebar_visible),
    )

    # The transcript column carries no explicit weight: with both panels
    # hidden (the default) it takes the whole terminal, and when one is
    # toggled on it yields exactly that panel's width.
    #
    # One column of margin on each side. Two cells of terminal width buy a
    # noticeable amount of calm: text that starts at column 0 and runs to
    # the last cell reads as output, text with air around it reads as a
    # document.
    transcript_column = VSplit(
        [
            Window(width=1),
            HSplit(
                [
                    tab_bar,
                    content_body,
                    chat_input_area,
                ]
            ),
            Window(width=1),
        ]
    )

    main_body = VSplit([
        explorer_panel,
        explorer_border,
        transcript_column,
        sidebar_border_conditional,
        sidebar_content,
    ])

    # ── Status line ─────────────────────────────────────────────────
    #
    # One line replaces the old status bar + footer pair. The attribute
    # names stay (`status_bar_control`, `footer_control`) because the app
    # and the workers drive them by name from a dozen call sites.

    from infinidev.ui.controls.status_line import StatusLineControl

    status_line_control = StatusLineControl(app_state)
    app_state.status_bar_control = status_line_control
    app_state.footer_control = status_line_control
    app_state.status_line_control = status_line_control

    status_line = VSplit(
        [
            Window(width=1, style=f"bg:{SURFACE_DARK}"),
            Window(content=status_line_control, style=f"bg:{SURFACE_DARK}"),
            Window(width=1, style=f"bg:{SURFACE_DARK}"),
        ],
        height=1,
    )

    # ── Root: float container (for dialogs) wrapping full layout ────

    root = FloatContainer(
        content=HSplit([
            main_body,
            status_line,
        ]),
        floats=[],  # Dialogs added in Phase 8
    )

    app_state._float_container = root

    return Layout(root, focused_element=app_state._chat_input_control)
