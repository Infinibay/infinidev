"""Custom widgets for Infinidev TUI context window visualization.

Provides progress bar widgets similar to Textual's Progress bar,
as well as styled message widgets for queued vs processed messages.
"""

from __future__ import annotations

from enum import Enum
from rich.text import Text
from textual import events
from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Static
from textual.message import Message
from textual.style import Style


class ContextProgressBar(Widget):
    """A progress bar specifically for context window usage.

    Displays context window with visual indicators for usage level.
    """

    CSS = """
    ContextProgressBar {
        height: 2;
        padding: 1 2;
        margin-top: 1;
    }

    .context-label {
        width: auto;
        text-align: left;
        color: $text;
        text-style: bold;
        margin-right: 1;
    }

    .context-usage {
        width: 10;
        text-align: center;
        color: $text;
    }

    .context-bar {
        width: 25;
        height: 1;
        background: $surface-darken-2;
        margin-left: 1;
        margin-right: 1;
    }

    .context-fill {
        height: 1;
    }

    .context-fill.critical {
        color: #ff4444;
    }

    .context-fill.warning {
        color: #ffaa00;
    }

    .context-fill.good {
        color: #44ff44;
    }

    .context-remaining {
        color: $text-muted;
        text-align: right;
    }
    """

    def __init__(self, label: str, usage: float, remaining: int, max_context: int = 4096):
        super().__init__()
        self._label = label
        self._usage = usage
        self._remaining = remaining
        self._max_context = max_context
        self._update_style()

    def _update_style(self) -> None:
        """Update style based on usage percentage."""
        self._fill_class = "critical" if self._usage < 0.3 else "warning" if self._usage < 0.7 else "good"

    @property
    def usage(self) -> float:
        """Get usage percentage (0.0 to 1.0)."""
        return self._usage

    @usage.setter
    def usage(self, value: float) -> None:
        """Set usage percentage."""
        self._usage = max(0.0, min(1.0, value))
        self._update_style()
        self.refresh()

    @property
    def remaining(self) -> int:
        """Get remaining tokens."""
        return self._remaining

    @remaining.setter
    def remaining(self, value: int) -> None:
        """Set remaining tokens."""
        self._remaining = max(0, value)
        self.refresh()

    def render(self) -> Text:
        """Render the context progress bar."""
        percentage = self._usage * 100
        bar_width = 25
        filled = int(bar_width * (1.0 - self._usage))  # Fill based on remaining
        empty = bar_width - filled

        bar = Text()
        bar.append(f"{self._label}: ", "bold")
        bar.append(f"{percentage:.0f}% used, ", "")
        bar.append(f"{self._remaining} tokens", "")

        # Progress bar visualization
        bar.append("[", "bold dim")
        bar.append("█" * filled, style=f"context-fill {self._fill_class}")
        bar.append("░" * empty, style="dim")
        bar.append("]", "bold dim")

        return bar


