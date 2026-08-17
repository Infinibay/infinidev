"""Tool: register, list, fire, and inspect programmable notifications.

The tool is the only handle the agent has on the notifications
subsystem. The scheduler runs in a daemon thread regardless of whether
this tool is invoked, so notifications registered by a previous
session keep firing.
"""

from __future__ import annotations

import json
import logging
from typing import Type

from pydantic import BaseModel

from infinidev.notifications.channels import deliver as deliver_payload
from infinidev.notifications.models import (
    ChannelConfig,
    NotificationTrigger,
    TriggerSpec,
)
from infinidev.notifications.scheduler import get_default_scheduler
from infinidev.notifications.storage import (
    NotificationStore,
    get_default_store,
)
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.meta.notifications_input import ManageNotificationsInput

logger = logging.getLogger(__name__)


class ManageNotificationsTool(InfinibayBaseTool):
    """Manage user-programmable notifications (cron/interval/script/file/agent)."""

    name: str = "manage_notifications"
    description: str = (
        "Create, list, enable, disable, fire, or delete programmable notifications. "
        "Triggers: interval (every N seconds), cron (5-field expression), "
        "script (run a command, fire on exit-code match or stdout regex), "
        "file (fire when a path's mtime or sha256 changes), "
        "agent (fire only when explicitly invoked via action='fire'). "
        "Channels: console (append to ~/.infinidev/notifications.log) or webhook (HTTP POST). "
        "The scheduler runs in a daemon thread — registered notifications keep firing across sessions."
    )
    args_schema: Type[BaseModel] = ManageNotificationsInput

    def _run(
        self,
        action: str,
        name: str | None = None,
        title: str | None = None,
        template: str | None = None,
        trigger_type: str | None = None,
        trigger: dict | None = None,
        channel_type: str | None = None,
        channel: dict | None = None,
        enabled: bool | None = None,
        limit: int | None = None,
    ) -> str:
        store = get_default_store()
        scheduler = get_default_scheduler()

        if action == "list":
            return self._render_list(store)

        if action == "history":
            return self._render_history(store, name, limit)

        if action == "create":
            return self._create(
                store,
                name=name,
                title=title,
                template=template,
                trigger_type=trigger_type,
                trigger=trigger,
                channel_type=channel_type,
                channel=channel,
                enabled=enabled,
            )

        if action not in {"fire", "enable", "disable", "delete"}:
            return self._error(f"unknown action {action!r}")

        if not name:
            return self._error(
                f"action={action!r} requires a notification name (use action='list' to see existing)"
            )

        if action == "delete":
            ok = store.delete_by_name(name)
            return self._success({"name": name, "deleted": ok})

        if action in ("enable", "disable"):
            notif = store.get_by_name(name)
            if not notif:
                return self._error(f"no notification named {name!r}")
            wanted = action == "enable"
            store.update_enabled(notif.id, wanted)
            return self._success({"name": name, "enabled": wanted})

        # action == "fire"
        notif = store.get_by_name(name)
        if not notif:
            return self._error(f"no notification named {name!r}")
        payload = trigger if isinstance(trigger, dict) else {}
        result = scheduler.fire_agent(name, payload)
        return self._success(result)

    # ── Action helpers ──────────────────────────────────────────────────
    def _create(
        self,
        store: NotificationStore,
        *,
        name: str | None,
        title: str | None,
        template: str | None,
        trigger_type: str | None,
        trigger: dict | None,
        channel_type: str | None,
        channel: dict | None,
        enabled: bool | None,
    ) -> str:
        if not name:
            return self._error("create requires name")
        if not trigger_type:
            return self._error("create requires trigger_type")
        if not channel_type:
            return self._error("create requires channel_type")

        try:
            ttype = NotificationTrigger(trigger_type)
        except ValueError:
            return self._error(
                f"unknown trigger_type {trigger_type!r}; "
                f"expected one of {[t.value for t in NotificationTrigger]}"
            )

        trigger_dict = trigger if isinstance(trigger, dict) else {}
        try:
            spec = _build_trigger_spec(ttype, trigger_dict)
        except ValueError as exc:
            return self._error(str(exc))

        channel_dict = channel if isinstance(channel, dict) else {}
        try:
            chan = _build_channel_config(channel_type, channel_dict)
        except ValueError as exc:
            return self._error(str(exc))

        try:
            notif = store.create(
                name=name,
                trigger=spec,
                channel=chan,
                title=title or name,
                template=template or "{name} fired at {fired_at}",
                enabled=True if enabled is None else bool(enabled),
            )
        except ValueError as exc:
            return self._error(str(exc))

        # Wake the scheduler so the new rule is picked up immediately.
        get_default_scheduler().start()

        return self._success(
            {
                "id": notif.id,
                "name": notif.name,
                "trigger_type": spec.type,
                "channel_type": chan.type,
                "enabled": notif.enabled,
                "message": (
                    f"Notification {notif.name!r} registered. "
                    f"Trigger={spec.type}, channel={chan.type}."
                ),
            }
        )

    def _render_list(self, store: NotificationStore) -> str:
        items = store.list_all()
        if not items:
            return "No notifications registered. Use action='create' to add one."
        lines = [f"{len(items)} registered notification(s):"]
        for n in items:
            spec = n.trigger
            chan = n.channel
            extra = ""
            if spec.type == "interval":
                extra = f" every {spec.every_seconds}s"
            elif spec.type == "cron":
                extra = f" cron={spec.cron!r}"
            elif spec.type == "script":
                extra = f" cmd={spec.command!r}"
            elif spec.type == "file":
                extra = f" path={spec.path!r} watch={spec.watch}"
            lines.append(
                f"  - id={n.id} {n.name!r} "
                f"[{'on' if n.enabled else 'off'}] "
                f"trigger={spec.type}{extra} "
                f"channel={chan.type} "
                f"fired={n.fire_count}"
            )
        return "\n".join(lines)

    def _render_history(
        self, store: NotificationStore, name: str | None, limit: int | None
    ) -> str:
        cap = max(1, min(200, int(limit or 50)))
        if name:
            notif = store.get_by_name(name)
            if not notif:
                return self._error(f"no notification named {name!r}")
            rows = store.history(notif.id, limit=cap)
        else:
            rows = store.history(limit=cap)
        if not rows:
            return "No fire history yet."
        lines = [f"Last {len(rows)} fire(s):"]
        for row in rows:
            err = f" err={row['error']}" if row.get("error") else ""
            lines.append(
                f"  - id={row['id']} {row['notification_name']!r} "
                f"at={row['fired_at']:.0f} "
                f"status={row['status']}{err}"
            )
        return "\n".join(lines)


# ── Spec builders (factored out for unit-testability) ──────────────────────


def _build_trigger_spec(ttype: NotificationTrigger, raw: dict) -> TriggerSpec:
    """Convert raw dict from the LLM into a typed TriggerSpec."""
    if ttype == NotificationTrigger.INTERVAL:
        secs = raw.get("every_seconds") or raw.get("seconds")
        if not isinstance(secs, int) or secs <= 0:
            raise ValueError("interval trigger requires positive integer every_seconds")
        return TriggerSpec(type=ttype.value, every_seconds=int(secs))

    if ttype == NotificationTrigger.CRON:
        cron = raw.get("cron") or raw.get("expression")
        if not isinstance(cron, str) or not cron.strip():
            raise ValueError("cron trigger requires non-empty cron expression")
        from infinidev.notifications.scheduler import parse_cron  # local import: tested separately
        parse_cron(cron)  # raises ValueError if invalid
        return TriggerSpec(type=ttype.value, cron=cron.strip())

    if ttype == NotificationTrigger.SCRIPT:
        cmd = raw.get("command") or raw.get("cmd")
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("script trigger requires non-empty command")
        wd = raw.get("working_dir") or raw.get("cwd")
        exit_code = raw.get("expected_exit_code")
        if exit_code is None:
            exit_code = raw.get("exit_code")
        if exit_code is not None and not isinstance(exit_code, int):
            raise ValueError("expected_exit_code must be an integer or null")
        regex = raw.get("stdout_match") or raw.get("match")
        return TriggerSpec(
            type=ttype.value,
            command=cmd,
            working_dir=wd if isinstance(wd, str) else None,
            expected_exit_code=exit_code if isinstance(exit_code, int) else None,
            stdout_match=regex if isinstance(regex, str) else None,
        )

    if ttype == NotificationTrigger.FILE:
        path = raw.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ValueError("file trigger requires non-empty path")
        watch = raw.get("watch") or "mtime"
        if watch not in ("mtime", "sha256"):
            raise ValueError("file watch must be 'mtime' or 'sha256'")
        return TriggerSpec(type=ttype.value, path=path, watch=watch)

    if ttype == NotificationTrigger.AGENT:
        return TriggerSpec(type=ttype.value)

    raise ValueError(f"unsupported trigger_type {ttype!r}")


def _build_channel_config(channel_type: str, raw: dict) -> ChannelConfig:
    if channel_type == "console":
        return ChannelConfig(
            type="console",
            log_path=raw.get("log_path") if isinstance(raw.get("log_path"), str) else None,
        )
    if channel_type == "webhook":
        url = raw.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("webhook channel requires non-empty url")
        method = raw.get("method") or "POST"
        if method not in ("POST", "PUT"):
            raise ValueError("webhook method must be POST or PUT")
        headers = raw.get("headers") or {}
        if not isinstance(headers, dict):
            raise ValueError("webhook headers must be an object")
        return ChannelConfig(
            type="webhook",
            url=url.strip(),
            method=method,
            headers={str(k): str(v) for k, v in headers.items()},
        )
    raise ValueError(f"unsupported channel_type {channel_type!r}")


# Re-export for callers that want to invoke channels directly (e.g. tests).
__all__ = [
    "ManageNotificationsTool",
    "deliver_payload",
    "_build_channel_config",
    "_build_trigger_spec",
]