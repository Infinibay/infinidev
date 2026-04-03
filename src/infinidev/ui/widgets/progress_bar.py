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


class ProgressBar(Widget):
    """A progress bar widget with customizable color and label.

    Displays a horizontal progress bar showing percentage completion.
    Similar to Textual's Progress bar but simpler and more visual.
    """

    CSS = """
    ProgressBar {
        height: 1;
        padding: 1;
        background: $surface-darken-1;
        border: solid $primary;
    }

    .progress-fill {
        color: $accent;
        text-style: bold;
    }

    .progress-label {
        width: 10;
        text-align: center;
        color: $text-muted;
    }

    .progress-empty {
        color: $text-muted;
    }
    """

    def __init__(self, percentage: float = 0.0, label: str = "", color: str = "$accent"):
        super().__init__()
        self._percentage = percentage
        self._label = label
        self._color = color

    @property
    def percentage(self) -> float:
        """Get current percentage (0.0 to 1.0)."""
        return self._percentage

    @percentage.setter
    def percentage(self, value: float) -> None:
        """Set the progress percentage."""
        self._percentage = max(0.0, min(1.0, value))
        self.refresh()

    def render(self) -> Text:
        """Render the progress bar."""
        percentage = self._percentage
        bar_width = 20  # Number of characters in the bar
        filled = int(bar_width * percentage)
        empty = bar_width - filled

        # Calculate color based on percentage
        if percentage < 0.3:
            color = "#ff4444"  # Red - critical
        elif percentage < 0.7:
            color = "#ffaa00"  # Yellow - warning
        else:
            color = "#44ff44"  # Green - good

        # Build the progress bar text
        bar = Text()
        bar.append("[", "bold")
        bar.append("█" * filled, style=f"{color} bold")
        bar.append("░" * empty, style="$text-muted")
        bar.append("] ", "bold")

        # Add label if provided
        if self._label:
            bar.append(f"{self._label}")
            bar.append(" ", "")

        # Add percentage
        bar.append(f"{percentage * 100:.1f}%", f"bold {color}")

        return bar

    def action_increment(self, amount: float = 0.05) -> None:
        """Increment the progress."""
        self.percentage = min(1.0, self._percentage + amount)

    def action_decrement(self, amount: float = 0.05) -> None:
        """Decrement the progress."""
        self.percentage = max(0.0, self._percentage - amount)


