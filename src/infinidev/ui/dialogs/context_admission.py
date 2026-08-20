"""Modal decision for a task starting on a small context window."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import ConditionalContainer, Float, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.widgets import Button

from infinidev.ui.dialogs.base import dialog_frame
from infinidev.ui.theme import SURFACE, TEXT, TEXT_MUTED, WARNING

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp

DIALOG_NAME = "context_admission"


def create_context_admission_dialog(app_state: InfinidevApp) -> Float:
    """Create the modal that chooses a larger model or pressure compaction."""

    def state() -> dict[str, object]:
        return getattr(app_state, "_context_admission_state", {}) or {}

    def summary() -> list[tuple[str, str]]:
        replacement = state().get("replacement_model")
        active_window = int(state().get("active_window") or 0)
        message = (
            f"This model has a {active_window:,}-token context window. "
            "It will need early context compaction.\n"
        )
        if replacement:
            message += f"A compatible larger model is available: {replacement}."
        else:
            message += "No verified compatible model above 200k tokens is available."
        return [(TEXT, message)]

    explanation = Window(
        content=FormattedTextControl(summary),
        height=3,
        wrap_lines=True,
        style=f"bg:{SURFACE}",
    )
    compact_button = Button(
        "Compact and continue",
        handler=lambda: app_state._resolve_context_admission(False),
        width=24,
        left_symbol="[",
        right_symbol="]",
    )
    replacement_button = Button(
        "Use larger model",
        handler=lambda: app_state._resolve_context_admission(True),
        width=22,
        left_symbol="[",
        right_symbol="]",
    )
    buttons = [
        Window(),
        compact_button,
        ConditionalContainer(
            content=VSplit([Window(width=2), replacement_button]),
            filter=Condition(lambda: bool(state().get("replacement_model"))),
        ),
        Window(),
    ]
    app_state._context_admission_compact_button = compact_button
    app_state._context_admission_switch_button = replacement_button

    body = HSplit([
        Window(
            content=FormattedTextControl(lambda: [
                (f"{WARNING} bold", "Limited context window\n"),
                (TEXT_MUTED, "Choose how to start this task; the current session stays active."),
            ]),
            height=2,
            wrap_lines=True,
            style=f"bg:{SURFACE}",
        ),
        Window(height=1, style=f"bg:{SURFACE}"),
        explanation,
        Window(height=1, style=f"bg:{SURFACE}"),
        VSplit(buttons, height=1, style=f"bg:{SURFACE}"),
    ], style=f"bg:{SURFACE}")
    frame = dialog_frame(
        "Context window decision",
        body,
        width=82,
        height=13,
        border_color=WARNING,
        hints="esc compact and continue",
    )
    return Float(
        content=ConditionalContainer(
            content=frame,
            filter=Condition(lambda: app_state.active_dialog == DIALOG_NAME),
        ),
        transparent=False,
    )
