"""Delegated managers for the TUI app.

Following the ``FileManager`` precedent, this package hosts small
focused collaborators for ``InfinidevApp`` so the god-object in
``ui/app.py`` can shrink over time. Each manager owns one slice of
responsibility (rendering, chat log, engine lifecycle, dialogs) and
is wired into the app via composition.
"""

from infinidev.ui.managers.context_render import (
    build_context_fragments,
    build_usage_bar_fragments,
)

__all__ = [
    "build_context_fragments",
    "build_usage_bar_fragments",
]
