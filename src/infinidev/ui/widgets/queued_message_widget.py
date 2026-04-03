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
from infinidev.ui.widgets.queued_message_status import QueuedMessageStatus


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

