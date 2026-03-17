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


class QueuedMessageStatus(Enum):
    """Status of a queued message."""
    QUEUED = "queued"
    PROCESSING = "processing"
    PROCESSED = "processed"
    ERROR = "error"


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


class QueuedMessageWidget(Widget):
    """A widget for displaying queued/in-progress messages.

    Messages in this widget appear dimmed/opaque to indicate they
    are queued and not yet being processed by the agent.
    """

    CSS = """
    QueuedMessageWidget {
        height: auto;
        margin-top: 0;
        padding: 0 1;
        background: $surface-lighten-1;
        opacity: 0.5;
        border-left: wide $text-muted;
    }

    .queued-header {
        color: $text-muted;
        text-style: italic;
        margin-bottom: 0.5;
    }

    .queued-content {
        color: $text-muted;
        margin-top: 0.5;
    }
    """

    class MessageProcessed(Message):
        """Emitted when a queued message is processed."""
        def __init__(self, widget: QueuedMessageWidget):
            super().__init__()
            self.widget = widget

    def __init__(self, content: str, user: str = "User", queued_index: int = 1):
        super().__init__()
        self._user = user
        self._content = content
        self._processed = False
        self._processed_widget: Vertical | None = None
        self._queued_index = queued_index

    @property
    def is_processed(self) -> bool:
        """Check if this message has been processed."""
        return self._processed

    @property
    def queued_index(self) -> int:
        """Get the queue position of this message."""
        return self._queued_index

    def render(self) -> Text:
        """Render the queued message."""
        text = Text()

        # Header with icon and user
        text.append("⏳ ", "bold dim")
        text.append(f"[Queued message #{self._queued_index} - Not yet processed] ", "italic dim")
        text.append(f"{self._user}: ", "dim")

        # Content
        text.append(self._content, "dim")

        return text

    def mark_processed(self) -> None:
        """Mark this message as processed."""
        self._processed = True
        self.refresh()
        self.post_message(self.MessageProcessed(self))

    def to_processed_widget(self) -> Vertical:
        """Convert to a processed message widget.

        Returns a Vertical container that displays the message as
        a normal, fully-visible user message.
        """
        from rich.text import Text as RichText

        processed = Vertical()
        processed.add_class("processed-message")
        processed.add_class("user-msg")

        # Header
        header = Static(f"[bold]You[/bold]: {self._content}")
        header.add_class("user-header")
        processed.mount(header)

        return processed

    def update_status(self, status: QueuedMessageStatus) -> None:
        """Update the widget's visual status."""
        if status == QueuedMessageStatus.PROCESSED:
            self.mark_processed()
