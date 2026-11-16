"""Bounded host observations; importing the reader never starts the agent or a database."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
import functools
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import stat
import threading
import time
from typing import Any
import uuid

logger = logging.getLogger(__name__)
_MAX_BYTES = 8192
_MAX_RECORDS = 32
_LEASE = 15.0
_HEARTBEAT = 5.0
_PHASES = {"preparing", "working", "inference", "tool", "awaiting_input", "closed", "unknown"}
_EVENTS = {"parent_started", "model_request", "provider_request", "model_response",
           "tool_started", "tool_completed", "child_started", "child_run_closed",
           "awaiting_input", "input_resumed", "parent_closed", "scope_interrupted"}
_TERMINAL = {"answered", "cancelled", "exhausted", "failed"}
_STRINGS = {"session_id", "owner_instance_id", "process_domain", "process_start",
            "host_turn_id", "child_run_id", "phase", "event", "tool", "last_tool",
            "terminal_status", "final_message_sha256", "usage_source"}
_INTEGERS = {"schema_version", "pid", "sequence", "event_sequence", "prompt_tokens",
             "completion_tokens", "model_elapsed_ms"}
_TIMES = {"started_at", "last_event_at", "last_progress_at", "heartbeat_at",
          "heartbeat_monotonic", "tool_completed_at", "closed_at"}
_FIELDS = _STRINGS | _INTEGERS | _TIMES
_NULLABLE = {"process_domain", "process_start", "child_run_id", "tool", "last_tool",
             "terminal_status", "final_message_sha256", "usage_source", "prompt_tokens",
             "completion_tokens", "model_elapsed_ms", "last_progress_at",
             "tool_completed_at", "closed_at"}
_CURRENT: contextvars.ContextVar[StatusPublisher | None] = contextvars.ContextVar(
    "principal_host_status", default=None,
)
_OWNER_LOCK = threading.Lock()


def _clocks() -> tuple[float, float]:
    return time.time(), time.monotonic()


def _process_identity(pid: int) -> tuple[str, str] | None:
    """A local boot/PID-namespace domain plus start ticks; unsupported means unknown."""
    try:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        namespace = os.readlink(f"/proc/{pid}/ns/pid")
        fields = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()
        if len(fields) < 20 or fields[0] in {"Z", "X", "x"} or not fields[19].isdigit():
            return None
        return boot + "/" + namespace, fields[19]
    except (OSError, ValueError):
        return None


def _bucket(workspace: Path, session_id: str, *, create: bool = False) -> Path | None:
    path = Path(workspace).resolve()
    for part in (".infinidev", "session-status", "v1", hashlib.sha256(session_id.encode()).hexdigest()):
        path = path / part
        if path.is_symlink():
            raise OSError("symlink status directory")
        if create:
            path.mkdir(mode=0o700, exist_ok=True)
        if not path.is_dir():
            return None
    return path


def _hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(c in "0123456789abcdef" for c in value)


def _valid(data: Any, session_id: str) -> bool:
    if not isinstance(data, dict) or set(data) != _FIELDS:
        return False
    for key, value in data.items():
        if value is None and key in _NULLABLE:
            continue
        if key in _STRINGS and (not isinstance(value, str) or len(value) > 512):
            return False
        if key in _INTEGERS and (type(value) is not int or value < 0):
            return False
        if key in _TIMES and (type(value) not in {int, float} or not math.isfinite(value) or value < 0):
            return False
    if (data["schema_version"] != 1 or data["session_id"] != session_id
            or data["phase"] not in _PHASES or data["event"] not in _EVENTS
            or data["pid"] <= 0 or not _hex(data["owner_instance_id"], 32)
            or not _hex(data["host_turn_id"], 32)):
        return False
    if data["child_run_id"] is not None and not _hex(data["child_run_id"], 32):
        return False
    if data["phase"] == "closed":
        return (data["terminal_status"] in _TERMINAL
                and _hex(data["final_message_sha256"], 64) and data["closed_at"] is not None)
    return data["terminal_status"] is None and data["final_message_sha256"] is None


def _read(path: Path, session_id: str) -> dict | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        return None
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_BYTES:
                return None
            raw = handle.read(_MAX_BYTES + 1)
        data = json.loads(raw)
        return data if len(raw) <= _MAX_BYTES and _valid(data, session_id) else None
    except (OSError, ValueError, UnicodeError, RecursionError):
        return None


def _fresh(data: dict) -> bool:
    identity = _process_identity(data["pid"])
    age = _clocks()[1] - data["heartbeat_monotonic"]
    return (identity is not None and identity == (data["process_domain"], data["process_start"])
            and 0 <= age <= _LEASE)


def _definitely_dead(data: dict) -> bool:
    local = _process_identity(os.getpid())
    if local is None or data["process_domain"] != local[0]:
        return False
    current = _process_identity(data["pid"])
    if current is not None:
        return current[1] != data["process_start"]
    # Only a local, matching PID namespace may establish absence.
    return Path("/proc/self/stat").is_file() and not Path(f"/proc/{data['pid']}").exists()


def read_status(workspace: Path, session_id: str, *, expected_turn: str | None = None,
                expected_run: str | None = None) -> dict[str, Any]:
    """Observe a named workspace/session without creating state or starting services."""
    result: dict[str, Any] = {"schema_version": 1, "session_id": session_id,
                              "state": "unknown", "reason": "missing", "last_known": None}
    try:
        directory = _bucket(workspace, session_id)
        if directory is None:
            return result
        paths = []
        for count, path in enumerate(directory.iterdir()):
            if count >= 64:
                return {**result, "reason": "scan_limit"}
            if path.name.endswith(".json"):
                paths.append(path)
            if len(paths) > _MAX_RECORDS:
                return {**result, "reason": "record_limit"}
        records = [_read(path, session_id) for path in paths]
    except (OSError, ValueError, UnicodeError):
        return {**result, "reason": "unavailable"}
    if any(row is None for row in records):
        return {**result, "reason": "invalid_record"}
    if not records:
        return result
    latest = max(records, key=lambda row: row["heartbeat_at"])
    result["last_known"] = latest
    pending = [row for row in records if row["phase"] != "closed" and not _definitely_dead(row)]
    named = expected_turn is not None or expected_run is not None
    if named:
        result["scope"] = "named_turn"
        matches = [row for row in records
                   if (expected_turn is None or row["host_turn_id"] == expected_turn)
                   and (expected_run is None or row["child_run_id"] == expected_run)]
        if not matches:
            return {**result, "reason": "turn_mismatch"}
        if len(matches) > 1:
            return {**result, "reason": "multiple_matching_owners"}
        chosen = matches[0]
        # Other owners qualify session-wide uncertainty, not the authority of
        # this explicitly selected observation. No record is retired here.
        others = [(row, _fresh(row)) for row in pending if row is not chosen]
        result["other_active_owners"] = sum(
            fresh and row["phase"] != "unknown" for row, fresh in others
        )
        result["other_owners_unconfirmed"] = any(
            not fresh or row["phase"] == "unknown" for row, fresh in others
        )
    else:
        if any(not _fresh(row) for row in pending):
            return {**result, "reason": "owner_unconfirmed"}
        active = [row for row in pending if row["phase"] != "unknown"]
        if any(row["phase"] == "unknown" for row in pending):
            return {**result, "reason": "scope_interrupted"}
        if len(active) > 1:
            return {**result, "reason": "multiple_active_owners"}
        chosen = active[0] if active else latest
    result["last_known"] = chosen
    if not _fresh(chosen):
        return {**result, "reason": "owner_unconfirmed"}
    if chosen["phase"] == "closed" and not named:
        return {**result, "reason": "between_turns"}
    if chosen["phase"] == "unknown":
        return {**result, "reason": "scope_interrupted"}
    return {**result, "state": chosen["phase"], "reason": "host_observation",
            "scope": "named_turn" if named else "active_principal",
            "observation": chosen}


class StatusPublisher:
    """Coalesced metadata publication with bounded caller wait and no raw payloads."""

    def __init__(self, workspace: Path, session_id: str, owner_instance_id: str) -> None:
        now, monotonic = _clocks()
        identity = _process_identity(os.getpid())
        self.workspace, self.session_id = Path(workspace), session_id
        self.host_turn_id = uuid.uuid4().hex
        self._data = {key: None for key in _FIELDS}
        self._data.update(schema_version=1, session_id=session_id, owner_instance_id=owner_instance_id,
                          pid=os.getpid(), process_domain=identity[0] if identity else None,
                          process_start=identity[1] if identity else None, host_turn_id=self.host_turn_id,
                          sequence=1, event_sequence=1, phase="preparing", event="parent_started",
                          started_at=now, last_event_at=now, heartbeat_at=now,
                          heartbeat_monotonic=monotonic)
        self._condition = threading.Condition()
        self._changed = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._published = 0
        self._failed = False
        self._finished = False

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True, name="session-status")
            self._changed.set()
            self._thread.start()

    def _tick(self, *, event: bool = True, progress: bool = False) -> None:
        now, monotonic = _clocks()
        self._data.update(sequence=self._data["sequence"] + 1,
                          heartbeat_at=now, heartbeat_monotonic=monotonic)
        if event:
            self._data.update(event_sequence=self._data["event_sequence"] + 1, last_event_at=now)
        if progress:
            self._data["last_progress_at"] = now

    def record(self, phase: str, event: str, *, child_run_id: str | None = None,
               tool_name: str | None = None, progress: bool = False,
               prompt_tokens: int | None = None, completion_tokens: int | None = None,
               model_elapsed_ms: int | None = None) -> None:
        if phase not in _PHASES - {"closed"} or event not in _EVENTS:
            return
        with self._condition:
            if self._finished or self._stop.is_set():
                return
            self._data.update(phase=phase, event=event)
            if child_run_id is not None:
                self._data["child_run_id"] = child_run_id
            self._data["tool"] = tool_name if phase in {"tool", "awaiting_input"} else None
            if event == "tool_completed":
                self._data.update(last_tool=tool_name, tool_completed_at=_clocks()[0])
            if event == "model_response":
                self._data.update(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                                  model_elapsed_ms=model_elapsed_ms,
                                  usage_source="response_counter_delta" if (
                                      prompt_tokens is not None or completion_tokens is not None) else None)
            self._tick(progress=progress)
        self._changed.set()

    def finish(self, terminal_status: str, final_message_sha256: str) -> None:
        if terminal_status not in _TERMINAL or not _hex(final_message_sha256, 64):
            return
        with self._condition:
            if self._finished:
                return
            self._finished = True
            self._data.update(phase="closed", event="parent_closed", terminal_status=terminal_status,
                              closed_at=_clocks()[0], final_message_sha256=final_message_sha256, tool=None)
            self._tick(progress=True)
        self._changed.set()

    def heartbeat(self) -> None:
        with self._condition:
            self._tick(event=False)
        self._changed.set()

    def flush(self, timeout: float = 1.0) -> bool:
        with self._condition:
            target = self._data["sequence"]
            self._condition.wait_for(lambda: self._failed or self._published >= target, timeout)
            return self._published >= target

    def close(self) -> None:
        if not self._finished:
            self.record("unknown", "scope_interrupted")
        self._stop.set()
        self._changed.set()
        if self._thread is not None:
            self._thread.join(timeout=0.25)

    def _write(self, directory: Path, data: dict) -> None:
        if not _valid(data, self.session_id):
            raise ValueError("invalid status primitives")
        payload = json.dumps(data, ensure_ascii=True).encode()
        if len(payload) > _MAX_BYTES:
            raise ValueError("status size limit")
        destination = directory / (data["owner_instance_id"] + ".json")
        if destination.is_symlink():
            raise OSError("symlink status record")
        temporary = directory / (".tmp-" + uuid.uuid4().hex)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _prune(self, directory: Path) -> None:
        dead = []
        for count, path in enumerate(directory.iterdir()):
            if count >= 64:
                break
            if path.suffix == ".json":
                data = _read(path, self.session_id)
                if data and _definitely_dead(data):
                    dead.append((data["heartbeat_at"], path))
        for _, path in sorted(dead, reverse=True)[2:]:
            path.unlink(missing_ok=True)

    def _run(self) -> None:
        try:
            directory = _bucket(self.workspace, self.session_id, create=True)
            self._prune(directory)
            while True:
                if not self._changed.wait(timeout=_HEARTBEAT):
                    with self._condition:
                        self._tick(event=False)
                self._changed.clear()
                with self._condition:
                    data = dict(self._data)
                self._write(directory, data)
                with self._condition:
                    self._published = data["sequence"]
                    self._condition.notify_all()
                    if self._stop.is_set() and self._published >= self._data["sequence"]:
                        return
        except (OSError, ValueError):
            with self._condition:
                self._failed = True
                self._condition.notify_all()
            logger.warning("Session status unavailable; observation remains unknown")


def current_observer() -> StatusPublisher | None:
    """Return the observation bound to this principal call, never a global session guess."""
    return _CURRENT.get()


@contextmanager
def bind_observer(publisher: StatusPublisher | None):
    """Carry the parent identity through synchronous runtime and tool callbacks."""
    token = _CURRENT.set(publisher)
    try:
        yield
    finally:
        _CURRENT.reset(token)


@contextmanager
def awaiting_input():
    """Observe a blocking host callback without receiving its prompt or reply."""
    publisher = current_observer()
    if publisher is None:
        yield
        return
    with publisher._condition:
        phase, tool = publisher._data["phase"], publisher._data["tool"]
    publisher.record("awaiting_input", "awaiting_input", tool_name=tool)
    try:
        yield
    finally:
        publisher.record(phase, "input_resumed", tool_name=tool)


def observed_principal(function):
    """Scope host observations to the entire principal request, including post-hooks."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        publisher = None
        try:
            engine = kwargs["engine"]
            workspace = kwargs.get("workspace_path") or kwargs.get("repository_path")
            if workspace:
                with _OWNER_LOCK:
                    previous = getattr(engine, "_session_status_publisher", None)
                    if (isinstance(previous, StatusPublisher)
                            and previous._data["pid"] == os.getpid()
                            and previous._thread is not None
                            and not previous._thread.is_alive()):
                        owner = previous._data["owner_instance_id"]
                    else:
                        # A stalled prior writer must never overwrite a newer turn's file.
                        owner = uuid.uuid4().hex
                    publisher = StatusPublisher(Path(workspace), kwargs["session_id"], owner)
                    engine._session_status_publisher = publisher
                    publisher.start()
        except Exception:
            # Observability failure must not change the task or expose raw exception text.
            logger.warning("Session status unavailable; observation remains unknown")
            publisher = None
        with bind_observer(publisher):
            try:
                return function(*args, **kwargs)
            finally:
                if publisher is not None:
                    publisher.close()
    return wrapped
