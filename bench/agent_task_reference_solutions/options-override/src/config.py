"""Options for report rendering."""

from __future__ import annotations

DEFAULTS: dict[str, object] = {"precision": 2, "separator": ", "}


def resolve_options(overrides: dict | None = None) -> dict:
    """Return the options for one report, defaults included.

    The caller's choices win over the defaults; anything the caller leaves out
    keeps its default.
    """
    options: dict = dict(DEFAULTS)
    if overrides:
        options.update(overrides)
    return options
