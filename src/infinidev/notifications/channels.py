"""Delivery channels for notifications."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from infinidev.notifications.models import ChannelConfig

logger = logging.getLogger(__name__)


def _resolve_log_path(configured: str | None) -> Path:
    """Pick a log path; default to ``~/.infinidev/notifications.log``."""
    if configured:
        return Path(configured).expanduser()
    path = Path.home() / ".infinidev" / "notifications.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def deliver_console(config: ChannelConfig, payload: dict[str, Any]) -> None:
    """Append a JSONL line to the configured log file + log via stdlib logger.

    The stdlib log line goes to the engine's normal logging sink so the
    operator sees it in their console output. The file is the durable
    record.
    """
    log_path = _resolve_log_path(config.log_path)
    record = {
        "ts": time.time(),
        "name": payload.get("name"),
        "title": payload.get("title"),
        "body": payload.get("body"),
    }
    line = json.dumps(record, sort_keys=True)
    try:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError as exc:
        logger.warning("Notification log write failed (%s): %s", log_path, exc)
    logger.info("notification fired: %s", line)


def deliver_webhook(config: ChannelConfig, payload: dict[str, Any]) -> None:
    """POST a JSON body to ``config.url``. Best-effort with 5s timeout."""
    if not config.url:
        raise ValueError("webhook channel requires a non-empty url")
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    headers.update(config.headers or {})
    req = urllib.request.Request(
        config.url,
        data=body,
        headers=headers,
        method=config.method or "POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # noqa: S310
            resp.read(1024)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"webhook POST failed: {exc}") from exc


def deliver(config: ChannelConfig, payload: dict[str, Any]) -> str:
    """Dispatch a payload through its channel. Returns status string."""
    ctype = (config.type or "console").lower()
    if ctype == "console":
        deliver_console(config, payload)
        return "delivered"
    if ctype == "webhook":
        deliver_webhook(config, payload)
        return "delivered"
    raise ValueError(f"unknown channel type: {config.type!r}")


def file_signature(path: str, watch: str) -> str | None:
    """Compute a file signature (mtime+size or sha256) for change detection.

    Returns ``None`` if the path does not exist or cannot be read; the
    scheduler treats a ``None`` signature as "no change yet" so a missing
    file doesn't immediately fire.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    if watch == "sha256":
        try:
            with open(path, "rb") as fh:
                digest = hashlib.sha256(fh.read()).hexdigest()
        except OSError:
            return None
        return f"sha256:{digest}:{st.st_size}"
    return f"mtime:{st.st_mtime_ns}:{st.st_size}"


def render_template(template: str, payload: dict[str, Any]) -> str:
    """Safe ``str.format`` with a default fallback on bad templates."""
    try:
        return template.format(**payload)
    except (KeyError, IndexError, ValueError):
        return template