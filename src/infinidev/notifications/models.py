"""Data models for the programmable notifications subsystem."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NotificationTrigger(str, Enum):
    """Kinds of triggers a notification can listen for."""

    INTERVAL = "interval"
    CRON = "cron"
    SCRIPT = "script"
    FILE = "file"
    AGENT = "agent"


class NotificationChannel(str, Enum):
    """Delivery mechanisms."""

    CONSOLE = "console"
    WEBHOOK = "webhook"


@dataclass(frozen=True)
class TriggerSpec:
    """Typed payload describing *how* a notification fires.

    Fields are kept loose so the storage layer can serialize arbitrary
    trigger configs as JSON; each trigger type validates the fields it
    needs at evaluation time.
    """

    type: str = "agent"
    # interval: seconds between fires
    every_seconds: int | None = None
    # cron: 5-field expression "minute hour dom month dow"
    cron: str | None = None
    # script: shell command and optional exit-code filter
    command: str | None = None
    working_dir: str | None = None
    expected_exit_code: int | None = None  # None means any non-zero
    # script: optional regex that stdout must match (re.search)
    stdout_match: str | None = None
    # file: path and what change to detect
    path: str | None = None
    watch: str = "mtime"  # "mtime" or "sha256"
    # agent: payload schema (free-form) — agent provides it at fire-time
    payload_schema: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | None) -> "TriggerSpec":
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return cls()
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class ChannelConfig:
    """Typed payload describing *where* a delivered notification goes."""

    type: str = "console"
    # console: path to log file; defaults to ~/.infinidev/notifications.log
    log_path: str | None = None
    # webhook: target URL and optional headers + method (default POST)
    url: str | None = None
    method: str = "POST"
    headers: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | None) -> "ChannelConfig":
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return cls()
        allowed = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if not isinstance(allowed.get("headers", {}), dict):
            allowed["headers"] = {}
        return cls(**allowed)


@dataclass
class Notification:
    """A registered notification rule."""

    id: int
    name: str
    enabled: bool
    trigger: TriggerSpec
    channel: ChannelConfig
    title: str = ""
    template: str = "{name} fired at {fired_at}"
    created_at: float = 0.0
    last_fired_at: float | None = None
    fire_count: int = 0

    def to_row(self) -> dict[str, Any]:
        """Serialize to a dict compatible with the SQLite row schema."""
        return {
            "id": self.id,
            "name": self.name,
            "enabled": 1 if self.enabled else 0,
            "trigger_json": self.trigger.to_json(),
            "channel_json": self.channel.to_json(),
            "title": self.title,
            "template": self.template,
            "created_at": self.created_at,
            "last_fired_at": self.last_fired_at,
            "fire_count": self.fire_count,
        }

    @classmethod
    def from_row(cls, row: Any) -> "Notification":
        return cls(
            id=row["id"],
            name=row["name"],
            enabled=bool(row["enabled"]),
            trigger=TriggerSpec.from_json(row["trigger_json"]),
            channel=ChannelConfig.from_json(row["channel_json"]),
            title=row["title"] or "",
            template=row["template"] or "{name} fired at {fired_at}",
            created_at=float(row["created_at"] or 0.0),
            last_fired_at=(
                float(row["last_fired_at"]) if row["last_fired_at"] is not None else None
            ),
            fire_count=int(row["fire_count"] or 0),
        )