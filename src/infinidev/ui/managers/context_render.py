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


def _format_count(value: int | None) -> str:
    """Format a token count in a compact way, '?' when unknown."""
    if value is None:
        return "?"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def build_usage_bar_fragments(
    label: str,
    used: int,
    available: int | None,
    pct: float,
) -> list[tuple[str, str]]:
    """Render a ``label: [███---] used/avail`` row.

    Bars come first with fixed width so successive rows line up
    visually regardless of the numeric widths of used/avail.  When
    *available* is ``None`` (unknown context window), the bar is
    muted and the available side is shown as ``*``.
    """
    pct_val = min(pct, 1.0)
    used_str = _format_count(used)
    label_col = f"{label:<5}"

    if available is None:
        return [
            (f"{TEXT} bold", label_col),
            (f"{TEXT_MUTED}", BAR_EMPTY * BAR_WIDTH),
            (f"{TEXT_MUTED}", f" {used_str}/*"),
        ]

    if pct_val > 0.8:
        color = PROGRESS_CRITICAL
    elif pct_val > 0.5:
        color = PROGRESS_WARNING
    else:
        color = PROGRESS_GOOD

    filled = int(BAR_WIDTH * pct_val)
    empty = BAR_WIDTH - filled
    avail_str = _format_count(available)

    return [
        (f"{TEXT} bold", label_col),
        (f"{color}", BAR_FILLED * filled),
        (f"{TEXT_MUTED}", BAR_EMPTY * empty),
        (f"{TEXT_MUTED}", f" {used_str}/{avail_str}"),
    ]


def build_context_fragments(
    context_status: dict,
    context_flow: str,
) -> FormattedText:
    """Render the full sidebar context block (model + chat/task bars)."""
    model = context_status.get("model", "unknown")
    max_ctx = context_status.get("max_context")  # may be None (unknown)
    flow_part = f"  {context_flow}" if context_flow else ""

    fragments: list[tuple[str, str]] = []
    fragments.append((f"{TEXT} bold", f"{model}"))
    if max_ctx is None:
        fragments.append((f"{TEXT_MUTED}", " (* ctx)"))
    else:
        fragments.append((f"{TEXT_MUTED}", f" ({_format_count(max_ctx)} ctx)"))
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
