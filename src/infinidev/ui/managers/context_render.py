"""Render the sidebar context/usage bars as prompt_toolkit fragments.

Extracted from ``InfinidevApp.get_context_fragments`` and
``InfinidevApp._usage_bar_fragments``. Pure: takes the
``_context_status`` dict and the app's ``_context_flow`` string, returns
a ``FormattedText``. No ``self``, no side effects — trivially testable.
"""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText

from infinidev.ui.theme import (
    TEXT,
    TEXT_MUTED,
    ACCENT,
    PROGRESS_GOOD,
    PROGRESS_WARNING,
    PROGRESS_CRITICAL,
    BAR_WIDTH,
    BAR_FILLED,
    BAR_EMPTY,
)


def build_usage_bar_fragments(
    label: str,
    used: int,
    available: int,
    pct: float,
) -> list[tuple[str, str]]:
    """Render a ``label: used/avail [███---] pct%`` row."""
    pct_val = min(pct, 1.0)
    if pct_val > 0.8:
        color = PROGRESS_CRITICAL
    elif pct_val > 0.5:
        color = PROGRESS_WARNING
    else:
        color = PROGRESS_GOOD

    filled = int(BAR_WIDTH * pct_val)
    empty = BAR_WIDTH - filled

    return [
        (f"{TEXT} bold", f"{label} "),
        (f"{TEXT_MUTED}", f"{used}/{available} "),
        (f"{color}", BAR_FILLED * filled),
        (f"{TEXT_MUTED}", BAR_EMPTY * empty),
        (f"{color} bold", f" {pct_val * 100:.0f}%"),
    ]


def build_context_fragments(
    context_status: dict,
    context_flow: str,
) -> FormattedText:
    """Render the full sidebar context block (model + chat/task bars)."""
    model = context_status.get("model", "unknown")
    max_ctx = context_status.get("max_context", 4096)
    flow_part = f"  {context_flow}" if context_flow else ""

    fragments: list[tuple[str, str]] = []
    fragments.append((f"{TEXT} bold", f"{model}"))
    fragments.append((f"{TEXT_MUTED}", f" ({max_ctx} ctx)"))
    if flow_part:
        fragments.append((f"{ACCENT} bold", flow_part))
    fragments.append(("", "\n"))

    chat = context_status.get("chat", {})
    fragments.extend(build_usage_bar_fragments(
        "Chat",
        chat.get("current_tokens", 0),
        chat.get("remaining_tokens", 0),
        chat.get("usage_percentage", 0.0),
    ))
    fragments.append(("", "\n"))

    tasks = context_status.get("tasks", {})
    fragments.extend(build_usage_bar_fragments(
        "Task",
        tasks.get("current_tokens", 0),
        tasks.get("remaining_tokens", 0),
        tasks.get("usage_percentage", 0.0),
    ))

    return FormattedText(fragments)
