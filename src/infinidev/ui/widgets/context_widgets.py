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
from infinidev.ui.widgets.progress_bar import ProgressBar
from infinidev.ui.widgets.context_progress_bar import ContextProgressBar
from infinidev.ui.widgets.queued_message_widget import QueuedMessageWidget
