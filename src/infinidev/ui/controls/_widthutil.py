"""Shared terminal-width helpers for the single-line status/footer bars.

Both :class:`StatusBarControl` and :class:`FooterControl` need to clamp
their content to the live terminal width. The column lookup and the
cell-aware truncation lived as byte-for-byte copies in both modules; a
single source here keeps them from drifting.
"""

from __future__ import annotations

from prompt_toolkit.utils import get_cwidth


def terminal_cols() -> int | None:
    """Current terminal column count, or None when no app is running.

    Returns None in headless contexts (e.g. tests) so callers skip
    width clamping and render their content raw.
    """
    try:
        from prompt_toolkit.application import get_app_or_none
        app = get_app_or_none()
        if app is None:
            return None
        return app.output.get_size().columns
    except Exception:
        return None


def truncate_cells(text: str, max_width: int) -> str:
    """Truncate *text* to *max_width* terminal columns, adding an ellipsis."""
    if max_width <= 0:
        return ""
    if get_cwidth(text) <= max_width:
        return text
    if max_width == 1:
        return "…"
    budget = max_width - 1  # leave room for the ellipsis (1 column)
    out = ""
    width = 0
    for ch in text:
        cw = get_cwidth(ch)
        if width + cw > budget:
            break
        out += ch
        width += cw
    return out + "…"
