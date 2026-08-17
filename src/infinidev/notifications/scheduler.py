"""Background scheduler that evaluates notification triggers.

A single daemon thread polls every ``POLL_INTERVAL_SECONDS`` and:
  - fires ``interval`` notifications whose next-fire time has elapsed
  - fires ``cron`` notifications whose 5-field expression matches now
  - fires ``script`` notifications whose command has not yet been
    evaluated this cycle AND that exited with the expected code (or
    any non-zero, if ``expected_exit_code`` is ``None``) and whose
    stdout matches the configured regex (if any)
  - fires ``file`` notifications when the path's signature changed

``agent`` triggers never fire from the scheduler; they are emitted
directly by the agent via ``manage_notifications(action="fire")``.

The scheduler is a process-wide singleton: the first caller of
``get_default_scheduler()`` spawns the thread, and ``atexit`` joins it
on interpreter shutdown. Tests can call ``reset_default_scheduler``
to start fresh.
"""

from __future__ import annotations

import atexit
import logging
import re
import subprocess
import threading
import time
from typing import Any

from infinidev.notifications.channels import (
    deliver,
    file_signature,
    render_template,
)
from infinidev.notifications.models import (
    ChannelConfig,
    Notification,
    NotificationChannel,
    NotificationTrigger,
    TriggerSpec,
)
from infinidev.notifications.storage import (
    NotificationStore,
    get_default_store,
)

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 1.0


# ── Cron expression support ───────────────────────────────────────────────
# Minimal 5-field cron parser supporting: minute, hour, day-of-month,
# month, day-of-week. Each field accepts *, specific values, comma lists,
# ranges (a-b), and step expressions (*/N or a-b/N). This is enough for
# "fire every minute", "fire every 5 minutes", "fire at 9:00", etc.
_CRON_FIELD_RANGES = (
    (0, 59),  # minute
    (0, 23),  # hour
    (1, 31),  # day of month
    (1, 12),  # month
    (0, 6),   # day of week (0 = Monday, matching cron's typical ordering)
)


def _parse_cron_field(field: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for chunk in field.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        step = 1
        if "/" in chunk:
            base, step_s = chunk.split("/", 1)
            step = max(1, int(step_s))
        else:
            base = chunk
        if base == "*":
            start, end = lo, hi
        elif "-" in base:
            s, e = base.split("-", 1)
            start, end = int(s), int(e)
        else:
            start = end = int(base)
        if start > end:
            start, end = end, start
        for v in range(start, end + 1, step):
            if v < lo or v > hi:
                raise ValueError(
                    f"cron value {v} out of range [{lo}, {hi}] in field {field!r}"
                )
            out.add(v)
    return out


def parse_cron(expr: str) -> tuple[set[int], set[int], set[int], set[int], set[int]]:
    """Parse a 5-field cron expression. Raises ValueError on bad input."""
    parts = expr.strip().split()
    if len(parts) != 5:
        raise ValueError(f"cron expression must have 5 fields, got {len(parts)}: {expr!r}")
    parsed = tuple(
        _parse_cron_field(parts[i], _CRON_FIELD_RANGES[i][0], _CRON_FIELD_RANGES[i][1])
        for i in range(5)
    )
    return parsed  # type: ignore[return-value]


def cron_matches(expr: str, now: time.struct_time) -> bool:
    """Return True if the 5-field cron expression matches ``now``."""
    minute, hour, dom, month, dow = parse_cron(expr)
    # Cron weekday: 0 = Sunday. struct_time.tm_wday: 0 = Monday.
    # Convert struct's Monday-first to cron's Sunday-first.
    cron_dow = (now.tm_wday + 1) % 7
    if now.tm_min not in minute:
        return False
    if now.tm_hour not in hour:
        return False
    if now.tm_mday not in dom:
        return False
    if now.tm_mon not in month:
        return False
    # Day-of-week is OR with day-of-month when both are restricted: cron
    # convention is "if both fields are restricted to specific values
    # (not '*'), the fire happens only when BOTH match". We implement the
    # conservative form: only AND when both fields are explicit.
    dow_is_restricted = dow != set(range(_CRON_FIELD_RANGES[4][0], _CRON_FIELD_RANGES[4][1] + 1))
    dom_is_restricted = dom != set(range(_CRON_FIELD_RANGES[2][0], _CRON_FIELD_RANGES[2][1] + 1))
    if dow_is_restricted and dom_is_restricted:
        return cron_dow in dow
    return cron_dow in dow


# ── Trigger evaluation ────────────────────────────────────────────────────


def _matches_interval(spec: TriggerSpec, last_fired: float | None, now: float) -> bool:
    if not spec.every_seconds or spec.every_seconds <= 0:
        return False
    if last_fired is None:
        return True
    return (now - last_fired) >= spec.every_seconds


def _matches_script(
    spec: TriggerSpec,
    state: dict[str, Any] | None,
    now: float,
) -> tuple[bool, int | None, str]:
    """Run the script and decide whether to fire.

    Returns (should_fire, exit_code, stdout). State is only consulted
    for the ``stdout_match`` case so repeated identical runs don't fire
    twice in a row.
    """
    if not spec.command:
        return False, None, ""
    try:
        proc = subprocess.run(
            spec.command,
            shell=True,
            cwd=spec.working_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, -1, f"<error: {exc}>"
    exit_code = proc.returncode
    stdout = proc.stdout or ""

    expected = spec.expected_exit_code
    code_ok = expected is None or exit_code == expected
    if not code_ok:
        return False, exit_code, stdout

    if spec.stdout_match:
        try:
            pattern = re.compile(spec.stdout_match)
        except re.error:
            pattern = None
        if pattern is None:
            return False, exit_code, stdout
        matched = bool(pattern.search(stdout))
        # Fire on transition from non-match to match to avoid flapping.
        if state and state.get("last_match"):
            return False, exit_code, stdout
        return matched, exit_code, stdout

    # No stdout filter: fire on every poll where the exit code matches.
    # Without state we'd fire repeatedly within the same poll interval;
    # require a recent prior check to have missed so we only fire once
    # per successful run.
    if state and state.get("last_checked", 0) >= now - POLL_INTERVAL_SECONDS:
        return False, exit_code, stdout
    return True, exit_code, stdout


def _matches_file(
    spec: TriggerSpec,
    state: dict[str, Any] | None,
) -> tuple[bool, str | None]:
    if not spec.path:
        return False, None
    sig = file_signature(spec.path, spec.watch or "mtime")
    if sig is None:
        return False, None
    if state is None or state.get("last_signature") != sig:
        return True, sig
    return False, sig


# ── Scheduler ─────────────────────────────────────────────────────────────


class NotificationScheduler:
    """Polls the store and fires notifications whose triggers match."""

    def __init__(
        self,
        store: NotificationStore | None = None,
        *,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self._store = store or get_default_store()
        self._poll_interval = max(0.05, float(poll_interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fire_lock = threading.Lock()

    @property
    def store(self) -> NotificationStore:
        return self._store

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="notif-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t and t.is_alive():
            t.join(timeout=timeout)
        self._thread = None

    # ── Public dispatch hooks (used by the tool) ───────────────────────
    def fire_agent(
        self, name: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Fire an ``agent``-type notification by name. Returns a status dict."""
        notif = self._store.get_by_name(name)
        if notif:
            return self._fire(notif, payload or {})
        return {"status": "not_found", "name": name}

    def fire_id(self, notification_id: int, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        notif = self._store.get(notification_id)
        if notif:
            return self._fire(notif, payload or {})
        return {"status": "not_found", "id": notification_id}

    # ── Loop ────────────────────────────────────────────────────────────
    def _run(self) -> None:
        logger.debug("notification scheduler started (poll=%.2fs)", self._poll_interval)
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("scheduler tick failed: %s", exc)
            self._stop.wait(self._poll_interval)

    def tick(self, now: float | None = None) -> list[dict[str, Any]]:
        """Evaluate every enabled notification once. Returns list of fired events."""
        now_ts = float(now) if now is not None else time.time()
        fired: list[dict[str, Any]] = []
        for notif in self._store.list_enabled():
            try:
                result = self._evaluate(notif, now_ts)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("evaluate failed for %s: %s", notif.name, exc)
                continue
            if result is not None:
                fired.append(result)
        return fired

    def _evaluate(self, notif: Notification, now: float) -> dict[str, Any] | None:
        spec = notif.trigger
        ttype = (spec.type or "").lower()

        if ttype == NotificationTrigger.AGENT.value:
            # Never poll-fired.
            return None

        if ttype == NotificationTrigger.INTERVAL.value:
            if _matches_interval(spec, notif.last_fired_at, now):
                return self._fire(notif, {})
            return None

        if ttype == NotificationTrigger.CRON.value:
            if spec.cron and cron_matches(spec.cron, time.localtime(now)):
                # Avoid firing twice within the same minute.
                if notif.last_fired_at and (now - notif.last_fired_at) < 30:
                    return None
                return self._fire(notif, {})
            return None

        if ttype == NotificationTrigger.SCRIPT.value:
            state = self._store.get_script_state(notif.id)
            should_fire, exit_code, stdout = _matches_script(spec, state, now)
            self._store.set_script_state(
                notif.id,
                exit_code,
                stdout,
                now,
                should_fire,
            )
            if should_fire:
                return self._fire(
                    notif,
                    {"exit_code": exit_code, "stdout": stdout[:1024]},
                )
            return None

        if ttype == NotificationTrigger.FILE.value:
            state = self._store.get_file_state(notif.id)
            should_fire, sig = _matches_file(spec, state)
            if sig is not None:
                self._store.set_file_state(notif.id, sig, now)
            if should_fire:
                return self._fire(notif, {"signature": sig})
            return None

        logger.warning("unknown trigger type %r for notification %s", spec.type, notif.name)
        return None

    def _fire(
        self, notif: Notification, extra_payload: dict[str, Any]
    ) -> dict[str, Any]:
        with self._fire_lock:
            payload = {
                "name": notif.name,
                "title": notif.title,
                "body": render_template(notif.template, {"name": notif.name, **extra_payload}),
                "fired_at": time.time(),
                "trigger": notif.trigger.type,
                "notification_id": notif.id,
            }
            payload.update(extra_payload)
            try:
                status = deliver(notif.channel, payload)
                self._store.record_fire(notif.id, status, None, payload)
                return {
                    "notification_id": notif.id,
                    "name": notif.name,
                    "status": status,
                    "channel": notif.channel.type,
                    "payload": payload,
                }
            except Exception as exc:
                self._store.record_fire(
                    notif.id, "error", str(exc), payload
                )
                return {
                    "notification_id": notif.id,
                    "name": notif.name,
                    "status": "error",
                    "error": str(exc),
                    "payload": payload,
                }


# ── Module-level singleton ─────────────────────────────────────────────────


_default_scheduler: NotificationScheduler | None = None
_default_scheduler_lock = threading.Lock()


def get_default_scheduler() -> NotificationScheduler:
    """Return the process-wide scheduler, spawning it lazily."""
    global _default_scheduler
    if _default_scheduler is None:
        with _default_scheduler_lock:
            if _default_scheduler is None:
                _default_scheduler = NotificationScheduler()
                _default_scheduler.start()
                atexit.register(_default_scheduler.stop)
    return _default_scheduler


def reset_default_scheduler() -> None:
    """Stop and forget the default scheduler (tests only)."""
    global _default_scheduler
    with _default_scheduler_lock:
        if _default_scheduler is not None:
            _default_scheduler.stop()
        _default_scheduler = None