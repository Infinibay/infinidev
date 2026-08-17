"""Programmable notification subsystem.

Users (or the agent itself) can register notifications with a trigger and
a delivery channel. A background scheduler thread evaluates time-based
triggers every poll cycle, and the agent fires event-based triggers
directly via the manage_notifications tool.

Trigger types:
  - ``interval``:  fire every N seconds
  - ``cron``:      fire when wall clock matches a 5-field cron expression
  - ``script``:    fire when a user-supplied shell command exits with a
                   specified code (and optionally when stdout matches a regex)
  - ``file``:      fire when a file's mtime or sha256 changes
  - ``agent``:     fire only when explicitly invoked by the agent via
                   ``action="fire"`` (used for "on plan done" style events)

Channels:
  - ``console``:  write to a log file (and emit on the engine log)
  - ``webhook``:  HTTP POST a JSON body to a configured URL

Storage is user-level (~/.infinidev/notifications.db) so notifications
configured for one project carry over to the next.
"""

from __future__ import annotations

from infinidev.notifications.models import (
    ChannelConfig,
    Notification,
    NotificationChannel,
    NotificationTrigger,
    TriggerSpec,
)
from infinidev.notifications.storage import (
    DEFAULT_DB_PATH,
    NotificationStore,
    get_default_store,
)
from infinidev.notifications.scheduler import (
    NotificationScheduler,
    get_default_scheduler,
)

__all__ = [
    "ChannelConfig",
    "DEFAULT_DB_PATH",
    "Notification",
    "NotificationChannel",
    "NotificationScheduler",
    "NotificationStore",
    "NotificationTrigger",
    "TriggerSpec",
    "get_default_scheduler",
    "get_default_store",
]