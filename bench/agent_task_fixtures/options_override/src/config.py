"""Options for report rendering."""

from __future__ import annotations

DEFAULTS: dict[str, object] = {"precision": 2, "separator": ", "}


def resolve_options(overrides: dict | None = None) -> dict:
    """Return the options for one report, defaults included."""
    options: dict = dict(overrides or {})
    options.update(DEFAULTS)
    return options
