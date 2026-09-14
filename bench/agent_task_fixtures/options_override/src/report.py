"""Markdown report rendering."""

from __future__ import annotations

from src.config import resolve_options


def render(values: list[float], overrides: dict | None = None) -> str:
    """Render ``values`` joined by the configured separator.

    ``overrides`` carries the caller's choices for this one report; anything it
    leaves out keeps its default.
    """
    options = resolve_options(overrides)
    precision = int(options["precision"])
    separator = str(options["separator"])
    return separator.join(f"{value:.{precision}f}" for value in values)
