"""Private, bounded storage for full command output.

The public ``artifacts`` catalog contains only an opaque storage key. Command
text and output live in mode-0600 files below ``.infinidev/private`` and never
enter indexed database columns. The store is deliberately internal; model-facing
range reads are implemented by the command-output tool, not by exposing paths.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import stat
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from infinidev.code_intel._db import execute_with_retry
from infinidev.config.settings import get_base_dir, settings

logger = logging.getLogger(__name__)

_ARTIFACT_TYPE = "command_output"
_STORE_VERSION = 1
_STORAGE_ID_RE = re.compile(r"^[0-9a-f]{48}$")
_STORAGE_REF_RE = re.compile(r"^([0-9a-f]{48})\.([0-9a-f]{64})$")
_STREAMS = frozenset({"stdout", "stderr"})
# A model-visible read is intentionally much smaller than the capture quota.
# This is a byte limit (not a character estimate), so one call has a finite
# worst-case prompt cost even when the artifact contains non-ASCII text.
COMMAND_OUTPUT_MAX_READ_BYTES = 64 * 1024


class CommandOutputStoreError(RuntimeError):
    """Base error for a rejected or failed private-store operation."""


class CommandOutputConfigError(CommandOutputStoreError):
    """Raised when capture was enabled without complete finite limits."""


class CommandOutputQuotaError(CommandOutputStoreError):
    """Raised when an output would exceed a configured storage quota."""


class CommandOutputIntegrityError(CommandOutputStoreError):
    """Raised when private storage is missing, linked, replaced, or corrupt."""


@dataclass(frozen=True)
class CommandOutputStoreConfig:
    """Finite limits required before private output capture may run."""

    max_artifact_bytes: int
    max_session_bytes: int
    max_project_bytes: int
    timeout_seconds: int
    retention_seconds: int
    sweep_grace_seconds: int

    @classmethod
    def from_settings(cls) -> CommandOutputStoreConfig:
        """Build and validate capture limits from global settings."""
        names = {
            "max_artifact_bytes": "COMMAND_OUTPUT_MAX_ARTIFACT_BYTES",
            "max_session_bytes": "COMMAND_OUTPUT_MAX_SESSION_BYTES",
            "max_project_bytes": "COMMAND_OUTPUT_MAX_PROJECT_BYTES",
            "timeout_seconds": "COMMAND_OUTPUT_STORE_TIMEOUT_SECONDS",
            "retention_seconds": "COMMAND_OUTPUT_RETENTION_SECONDS",
            "sweep_grace_seconds": "COMMAND_OUTPUT_SWEEP_GRACE_SECONDS",
        }
        values: dict[str, int] = {}
        invalid: list[str] = []
        for field_name, setting_name in names.items():
            value = getattr(settings, setting_name, None)
            if type(value) is not int or value <= 0:
                invalid.append(setting_name)
            else:
                values[field_name] = value
        if invalid:
            joined = ", ".join(invalid)
            raise CommandOutputConfigError(
                f"command-output capture requires positive integer settings: {joined}"
            )
        if values["max_session_bytes"] < values["max_artifact_bytes"]:
            raise CommandOutputConfigError(
                "COMMAND_OUTPUT_MAX_SESSION_BYTES must be at least "
                "COMMAND_OUTPUT_MAX_ARTIFACT_BYTES"
            )
        if values["max_project_bytes"] < values["max_session_bytes"]:
            raise CommandOutputConfigError(
                "COMMAND_OUTPUT_MAX_PROJECT_BYTES must be at least "
                "COMMAND_OUTPUT_MAX_SESSION_BYTES"
            )
        return cls(**values)


@dataclass(frozen=True)
class CommandOutputHandle:
    """Opaque catalog identity safe to pass through the tool transcript."""

    artifact_id: int
    artifact_type: str
    stream: str
    char_count: int
    byte_count: int

    def to_dict(self) -> dict[str, int | str]:
        """Return path-free, non-sensitive handle metadata."""
        return {
            "artifact_id": self.artifact_id,
            "type": self.artifact_type,
            "stream": self.stream,
            "char_count": self.char_count,
            "byte_count": self.byte_count,
        }


class CommandOutputStore:
    """Persist and validate command output outside searchable artifact content."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        db_path: str | None = None,
        config: CommandOutputStoreConfig | None = None,
    ) -> None:
        self._runtime_base = get_base_dir()
        self._root = root or self._runtime_base / "private" / "command_output"
        self._enforce_runtime_root = root is None
        self._db_path = db_path or settings.DB_PATH
        self._config = config or CommandOutputStoreConfig.from_settings()

    @property
    def root(self) -> Path:
        """Private root, exposed only for internal maintenance and tests."""
        return self._root

    def store_streams(
        self,
        *,
        project_id: int,
        session_id: str,
        streams: Mapping[str, str],
    ) -> dict[str, CommandOutputHandle]:
        """Atomically catalog one or more decoded output streams.

        All streams succeed together. Any limit, filesystem, integrity, timeout,
        or database failure removes completed files where possible and returns no
        handles to the caller (by raising).
        """
        project_id, session_id = self._validate_scope(project_id, session_id)
        if not streams:
            return {}

        encoded: dict[str, tuple[str, bytes]] = {}
        for stream, text in streams.items():
            if stream not in _STREAMS or not isinstance(text, str):
                raise CommandOutputStoreError("invalid command-output stream")
            payload = text.encode("utf-8")
            if len(payload) > self._config.max_artifact_bytes:
                raise CommandOutputQuotaError(
                    f"{stream} exceeds the per-artifact command-output quota"
                )
            encoded[stream] = (text, payload)

        deadline = time.monotonic() + self._config.timeout_seconds
        created: list[Path] = []
        inserted_ids: list[int] = []
        try:
            with self._locked(deadline):
                self._check_deadline(deadline)
                self._sweep_locked(deadline)
                project_used, session_used = self._usage_locked(
                    project_id, session_id, deadline
                )
                requested = sum(len(payload) for _, payload in encoded.values())
                if session_used + requested > self._config.max_session_bytes:
                    raise CommandOutputQuotaError(
                        "command-output session quota would be exceeded"
                    )
                if project_used + requested > self._config.max_project_bytes:
                    raise CommandOutputQuotaError(
                        "command-output project quota would be exceeded"
                    )

                records: list[dict[str, object]] = []
                for stream, (text, payload) in encoded.items():
                    self._check_deadline(deadline)
                    storage_id = self._new_storage_id()
                    # Register both intended final paths before either write.
                    # A crash/failure between blob and sidecar must still clean
                    # the already-renamed blob from this transaction.
                    created.extend((
                        self._blob_path(storage_id),
                        self._sidecar_path(storage_id),
                    ))
                    record = self._write_item(
                        storage_id=storage_id,
                        project_id=project_id,
                        session_id=session_id,
                        stream=stream,
                        text=text,
                        payload=payload,
                        deadline=deadline,
                    )
                    records.append(record)

                # Verify every final blob + sidecar before making any catalog row.
                for record in records:
                    self._verify_record_files(record, deadline)

                def _insert(conn):
                    ids: list[int] = []
                    try:
                        for record in records:
                            storage_ref = self._storage_ref(record)
                            cursor = conn.execute(
                                "INSERT INTO artifacts "
                                "(project_id, session_id, type, name, file_path, "
                                "description, content) "
                                "VALUES (?, ?, ?, NULL, ?, NULL, NULL)",
                                (
                                    project_id,
                                    session_id,
                                    _ARTIFACT_TYPE,
                                    storage_ref,
                                ),
                            )
                            ids.append(int(cursor.lastrowid))
                        conn.commit()
                        return ids
                    except Exception:
                        conn.rollback()
                        raise

                inserted_ids = execute_with_retry(_insert, db_path=self._db_path)
                if len(inserted_ids) != len(records):
                    raise CommandOutputStoreError("incomplete command-output catalog insert")

                # Catch replacement between pre-insert verification and commit.
                for record in records:
                    self._verify_record_files(record, deadline)

                return {
                    str(record["stream"]): CommandOutputHandle(
                        artifact_id=artifact_id,
                        artifact_type=_ARTIFACT_TYPE,
                        stream=str(record["stream"]),
                        char_count=int(record["char_length"]),
                        byte_count=int(record["byte_length"]),
                    )
                    for record, artifact_id in zip(records, inserted_ids, strict=True)
                }
        except Exception:
            if inserted_ids:
                self._delete_catalog_ids(inserted_ids)
            self._remove_paths(created)
            raise

    def read_text(
        self,
        handle: CommandOutputHandle,
        *,
        project_id: int,
        session_id: str,
    ) -> str:
        """Validate a handle and reconstruct its exact decoded pre-cut text.

        This internal whole-artifact read exists for integrity checks and tests.
        The model-facing tool uses :meth:`read_range` and never exposes storage
        paths or performs a path-based fallback.
        """
        text = self._read_validated_text(
            handle, project_id=project_id, session_id=session_id
        )
        if len(text) != handle.char_count:
            raise CommandOutputIntegrityError("command-output handle length mismatch")
        return text

    def read_range(
        self,
        handle: CommandOutputHandle,
        *,
        project_id: int,
        session_id: str,
        offset: int = 0,
        limit: int = COMMAND_OUTPUT_MAX_READ_BYTES,
    ) -> tuple[str, int, int, bool]:
        """Read a bounded UTF-8 byte range from a validated command output.

        ``offset`` and ``limit`` are byte counts. The returned text contains only
        complete UTF-8 characters: when either boundary falls inside a multibyte
        sequence, that partial character is omitted. The tuple is
        ``(content, returned_start, returned_end, has_more)`` where the offsets
        describe the exact source bytes represented by ``content``.
        """
        if type(offset) is not int or offset < 0:
            raise CommandOutputStoreError("command-output offset must be a non-negative integer")
        if (
            type(limit) is not int
            or limit <= 0
            or limit > COMMAND_OUTPUT_MAX_READ_BYTES
        ):
            raise CommandOutputStoreError(
                "command-output limit must be between 1 and "
                f"{COMMAND_OUTPUT_MAX_READ_BYTES} bytes"
            )
        if offset > handle.byte_count:
            raise CommandOutputStoreError("command-output offset exceeds artifact length")

        text = self._read_validated_text(
            handle, project_id=project_id, session_id=session_id
        )
        payload = text.encode("utf-8")
        if offset < len(payload) and payload[offset] & 0xC0 == 0x80:
            raise CommandOutputStoreError(
                "command-output offset must be on a UTF-8 character boundary"
            )
        requested_end = min(len(payload), offset + limit)
        returned_end = requested_end

        # Shorten only the end so the next call can resume at ``returned_end``
        # without skipping or duplicating a character. The start must already be
        # a boundary; silently advancing it would make exact reconstruction
        # impossible.
        while returned_end > offset:
            try:
                content = payload[offset:returned_end].decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                if exc.end == len(payload[offset:returned_end]):
                    returned_end = offset + exc.start
                else:
                    raise CommandOutputIntegrityError(
                        "command-output blob is not UTF-8"
                    ) from exc
        else:
            content = ""
        if not content and requested_end < len(payload):
            raise CommandOutputStoreError(
                "command-output limit is too small for the next UTF-8 character"
            )

        return content, offset, returned_end, returned_end < len(payload)

    def _read_validated_text(
        self,
        handle: CommandOutputHandle,
        *,
        project_id: int,
        session_id: str,
    ) -> str:
        """Resolve one opaque handle and validate catalog, sidecar, and blob."""
        project_id, session_id = self._validate_scope(project_id, session_id)
        if (
            not isinstance(handle, CommandOutputHandle)
            or handle.artifact_type != _ARTIFACT_TYPE
            or handle.stream not in _STREAMS
            or type(handle.artifact_id) is not int
            or handle.artifact_id <= 0
            or type(handle.char_count) is not int
            or handle.char_count < 0
            or type(handle.byte_count) is not int
            or handle.byte_count < 0
        ):
            raise CommandOutputIntegrityError("invalid command-output handle")

        deadline = time.monotonic() + self._config.timeout_seconds
        with self._locked(deadline):
            def _query(conn):
                return conn.execute(
                    "SELECT id, project_id, session_id, type, file_path, "
                    "name, description, content FROM artifacts "
                    "WHERE id = ? AND project_id = ? AND session_id = ? AND type = ?",
                    (handle.artifact_id, project_id, session_id, _ARTIFACT_TYPE),
                ).fetchone()

            row = execute_with_retry(_query, db_path=self._db_path)
            if row is None:
                raise CommandOutputIntegrityError("command-output handle scope mismatch")
            if (
                row["name"] is not None
                or row["description"] is not None
                or row["content"] is not None
            ):
                raise CommandOutputIntegrityError(
                    "command-output catalog metadata is unsafe"
                )
            storage_id, sidecar_sha256 = self._parse_storage_ref(row["file_path"])
            record = self._load_sidecar(
                storage_id, expected_sha256=sidecar_sha256
            )
            self._validate_sidecar_scope(
                record, project_id, session_id, handle.stream
            )
            payload = self._read_regular_file(self._blob_path(storage_id))
            self._validate_payload(record, payload)
            if len(payload) != handle.byte_count:
                raise CommandOutputIntegrityError(
                    "command-output handle length mismatch"
                )
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CommandOutputIntegrityError(
                    "command-output blob is not UTF-8"
                ) from exc
            if len(text) != handle.char_count:
                raise CommandOutputIntegrityError(
                    "command-output handle length mismatch"
                )
            return text

    def sweep(self) -> None:
        """Apply finite retention and remove old managed orphan/temp files."""
        deadline = time.monotonic() + self._config.timeout_seconds
        with self._locked(deadline):
            self._sweep_locked(deadline)

    @staticmethod
    def _validate_scope(project_id: int, session_id: str) -> tuple[int, str]:
        if type(project_id) is not int or project_id <= 0:
            raise CommandOutputStoreError("missing or invalid project scope")
        if not isinstance(session_id, str) or not session_id.strip():
            raise CommandOutputStoreError("missing or invalid session scope")
        if len(session_id) > 512 or "\x00" in session_id:
            raise CommandOutputStoreError("invalid session scope")
        return project_id, session_id

    def _ensure_root(self) -> None:
        root = self._root.absolute()
        base = self._runtime_base.absolute()
        if self._enforce_runtime_root and base not in root.parents:
            raise CommandOutputIntegrityError("private store root is outside runtime data")

        # Traverse with directory descriptors and O_NOFOLLOW. Checking only the
        # final path is insufficient: an existing intermediate symlink is
        # followed by lstat(root), making a linked store look like a real dir.
        parts = root.parts
        if not parts or not root.is_absolute():
            raise CommandOutputIntegrityError("private store root must be absolute")
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(parts[0], flags)
        except OSError as exc:
            raise CommandOutputIntegrityError(
                "private store root anchor is unavailable"
            ) from exc

        created = False
        try:
            for index, part in enumerate(parts[1:], start=1):
                try:
                    child_fd = os.open(part, flags, dir_fd=fd)
                except FileNotFoundError:
                    created = True
                    try:
                        os.mkdir(part, mode=0o700, dir_fd=fd)
                        child_fd = os.open(part, flags, dir_fd=fd)
                    except (FileExistsError, OSError) as exc:
                        raise CommandOutputIntegrityError(
                            "private store directory creation failed"
                        ) from exc
                except OSError as exc:
                    raise CommandOutputIntegrityError(
                        "private store path contains a symlink or non-directory"
                    ) from exc

                os.close(fd)
                fd = child_fd
                st = os.fstat(fd)
                if not stat.S_ISDIR(st.st_mode):
                    raise CommandOutputIntegrityError(
                        "private store path contains a non-directory"
                    )
                current = Path(*parts[: index + 1])
                managed = created or current == root
                if self._enforce_runtime_root and (
                    current == base or base in current.parents
                ):
                    managed = True
                if managed:
                    os.fchmod(fd, 0o700)
        finally:
            os.close(fd)

    @contextmanager
    def _locked(self, deadline: float) -> Iterator[None]:
        self._ensure_root()
        lock_path = self._root / ".lock"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(lock_path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                raise CommandOutputIntegrityError("private store lock is not regular")
            try:
                import fcntl
            except ImportError as exc:
                raise CommandOutputStoreError("file locking is unavailable") from exc
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self._check_deadline(deadline)
                    time.sleep(0.01)
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass
            os.close(fd)

    def _write_item(
        self,
        *,
        storage_id: str,
        project_id: int,
        session_id: str,
        stream: str,
        text: str,
        payload: bytes,
        deadline: float,
    ) -> dict[str, object]:
        del text
        created_at = int(time.time())
        digest = hashlib.sha256(payload).hexdigest()
        record: dict[str, object] = {
            "version": _STORE_VERSION,
            "storage_id": storage_id,
            "project_id": project_id,
            "session_id": session_id,
            "stream": stream,
            "byte_length": len(payload),
            "char_length": len(payload.decode("utf-8")),
            "sha256": digest,
            "created_at": created_at,
        }
        sidecar = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._atomic_write(self._blob_path(storage_id), payload, deadline)
        self._atomic_write(self._sidecar_path(storage_id), sidecar, deadline)
        self._fsync_directory()
        return record

    def _atomic_write(self, final_path: Path, payload: bytes, deadline: float) -> None:
        self._check_deadline(deadline)
        temp_path = self._root / f".tmp-{secrets.token_hex(24)}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(temp_path, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(payload)
            while view:
                self._check_deadline(deadline)
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write to command-output store")
                view = view[written:]
            os.fsync(fd)
        except Exception:
            os.close(fd)
            temp_path.unlink(missing_ok=True)
            raise
        else:
            os.close(fd)
        try:
            if final_path.exists() or final_path.is_symlink():
                raise CommandOutputIntegrityError("random storage-name collision")
            os.replace(temp_path, final_path)
            os.chmod(final_path, 0o600, follow_symlinks=False)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    def _verify_record_files(self, expected: Mapping[str, object], deadline: float) -> None:
        self._check_deadline(deadline)
        storage_id = self._validate_storage_id(expected.get("storage_id"))
        actual = self._load_sidecar(storage_id)
        if actual != dict(expected):
            raise CommandOutputIntegrityError("command-output sidecar was replaced or corrupt")
        payload = self._read_regular_file(self._blob_path(storage_id))
        self._validate_payload(actual, payload)

    def _load_sidecar(
        self, storage_id: str, *, expected_sha256: str | None = None
    ) -> dict[str, object]:
        raw = self._read_regular_file(self._sidecar_path(storage_id))
        if expected_sha256 is not None:
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if not secrets.compare_digest(actual_sha256, expected_sha256):
                raise CommandOutputIntegrityError(
                    "command-output sidecar substitution detected"
                )
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CommandOutputIntegrityError("invalid command-output sidecar") from exc
        if not isinstance(value, dict):
            raise CommandOutputIntegrityError("invalid command-output sidecar")
        required = {
            "version", "storage_id", "project_id", "session_id", "stream",
            "byte_length", "char_length", "sha256", "created_at",
        }
        if set(value) != required or value.get("version") != _STORE_VERSION:
            raise CommandOutputIntegrityError("unsupported command-output sidecar")
        if self._validate_storage_id(value.get("storage_id")) != storage_id:
            raise CommandOutputIntegrityError("command-output storage identity mismatch")
        return value

    @staticmethod
    def _validate_sidecar_scope(
        record: Mapping[str, object], project_id: int, session_id: str, stream: str
    ) -> None:
        if (
            record.get("project_id") != project_id
            or record.get("session_id") != session_id
            or record.get("stream") != stream
        ):
            raise CommandOutputIntegrityError("command-output sidecar scope mismatch")

    @staticmethod
    def _validate_payload(record: Mapping[str, object], payload: bytes) -> None:
        byte_length = record.get("byte_length")
        char_length = record.get("char_length")
        digest = record.get("sha256")
        if type(byte_length) is not int or byte_length < 0 or len(payload) != byte_length:
            raise CommandOutputIntegrityError("command-output byte length mismatch")
        if not isinstance(digest, str) or not secrets.compare_digest(
            hashlib.sha256(payload).hexdigest(), digest
        ):
            raise CommandOutputIntegrityError("command-output hash mismatch")
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CommandOutputIntegrityError("command-output blob is not UTF-8") from exc
        if type(char_length) is not int or char_length < 0 or len(text) != char_length:
            raise CommandOutputIntegrityError("command-output character length mismatch")

    @staticmethod
    def _validate_storage_id(value: object) -> str:
        if not isinstance(value, str) or _STORAGE_ID_RE.fullmatch(value) is None:
            raise CommandOutputIntegrityError("invalid command-output storage identity")
        return value

    @staticmethod
    def _storage_ref(record: Mapping[str, object]) -> str:
        storage_id = CommandOutputStore._validate_storage_id(
            record.get("storage_id")
        )
        sidecar = json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return f"{storage_id}.{hashlib.sha256(sidecar).hexdigest()}"

    @staticmethod
    def _parse_storage_ref(value: object) -> tuple[str, str]:
        if not isinstance(value, str):
            raise CommandOutputIntegrityError(
                "invalid command-output storage reference"
            )
        match = _STORAGE_REF_RE.fullmatch(value)
        if match is None:
            raise CommandOutputIntegrityError(
                "invalid command-output storage reference"
            )
        return match.group(1), match.group(2)

    def _read_regular_file(self, path: Path) -> bytes:
        # Validate the resolved location as well as opening with O_NOFOLLOW.
        # The first check rejects a redirected parent; the second rejects a
        # final-component symlink before open. O_NOFOLLOW closes the race between
        # this check and os.open.
        root_real = os.path.realpath(self._root)
        parent_real = os.path.realpath(path.parent)
        candidate_real = os.path.realpath(path)
        if (
            parent_real != root_real
            or os.path.dirname(candidate_real) != root_real
        ):
            raise CommandOutputIntegrityError(
                "command-output private file unavailable: resolved outside its store"
            )

        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except (FileNotFoundError, OSError) as exc:
            raise CommandOutputIntegrityError("command-output private file unavailable") from exc
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode) != 0o600:
                raise CommandOutputIntegrityError("command-output private file permissions invalid")
            chunks: list[bytes] = []
            total = 0
            hard_limit = self._config.max_artifact_bytes + 64 * 1024
            while True:
                chunk = os.read(fd, min(64 * 1024, hard_limit - total + 1))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > hard_limit:
                    raise CommandOutputIntegrityError("command-output private file is oversized")
            after = os.fstat(fd)
            if (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
            ):
                raise CommandOutputIntegrityError("command-output private file changed during read")
            return b"".join(chunks)
        finally:
            os.close(fd)

    def _usage_locked(
        self, project_id: int, session_id: str, deadline: float
    ) -> tuple[int, int]:
        def _query(conn):
            return conn.execute(
                "SELECT project_id, session_id, file_path FROM artifacts "
                "WHERE type = ? AND project_id = ?",
                (_ARTIFACT_TYPE, project_id),
            ).fetchall()

        rows = execute_with_retry(_query, db_path=self._db_path)
        project_used = 0
        session_used = 0
        for row in rows:
            self._check_deadline(deadline)
            storage_id, sidecar_sha256 = self._parse_storage_ref(row["file_path"])
            record = self._load_sidecar(
                storage_id, expected_sha256=sidecar_sha256
            )
            self._validate_sidecar_scope(
                record, row["project_id"], row["session_id"], str(record.get("stream"))
            )
            payload = self._read_regular_file(self._blob_path(storage_id))
            self._validate_payload(record, payload)
            size = len(payload)
            project_used += size
            if row["session_id"] == session_id:
                session_used += size
        return project_used, session_used

    def _sweep_locked(self, deadline: float) -> None:
        now = int(time.time())

        def _query(conn):
            return conn.execute(
                "SELECT id, file_path FROM artifacts WHERE type = ?",
                (_ARTIFACT_TYPE,),
            ).fetchall()

        rows = execute_with_retry(_query, db_path=self._db_path)
        referenced: set[str] = set()
        expired_ids: list[int] = []
        expired_paths: list[Path] = []
        for row in rows:
            self._check_deadline(deadline)
            storage_id, sidecar_sha256 = self._parse_storage_ref(row["file_path"])
            referenced.add(storage_id)
            try:
                record = self._load_sidecar(
                    storage_id, expected_sha256=sidecar_sha256
                )
                created_at = record.get("created_at")
                if type(created_at) is not int:
                    raise CommandOutputIntegrityError("invalid command-output timestamp")
            except CommandOutputIntegrityError:
                # Corruption remains fail-closed; only old filesystem orphans are
                # reclaimed below, never guessed from malformed metadata.
                continue
            if now - created_at >= self._config.retention_seconds:
                expired_ids.append(int(row["id"]))
                expired_paths.extend((self._blob_path(storage_id), self._sidecar_path(storage_id)))

        if expired_ids:
            self._delete_catalog_ids(expired_ids)
            self._remove_paths(expired_paths)
            referenced -= {
                path.stem for path in expired_paths if path.suffix in {".blob", ".json"}
            }

        for entry in os.scandir(self._root):
            self._check_deadline(deadline)
            if entry.name == ".lock":
                continue
            match = re.fullmatch(r"([0-9a-f]{48})\.(blob|json)", entry.name)
            is_temp = re.fullmatch(r"\.tmp-[0-9a-f]{48}", entry.name) is not None
            if not match and not is_temp:
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if now - int(st.st_mtime) < self._config.sweep_grace_seconds:
                continue
            if match and match.group(1) in referenced:
                continue
            # unlink removes a symlink itself and never follows its target.
            try:
                os.unlink(entry.path)
            except FileNotFoundError:
                pass
        self._remove_dangling_catalog_rows(deadline)
        self._fsync_directory()

    def _remove_dangling_catalog_rows(self, deadline: float) -> None:
        """Reclaim crash remnants only after the configured grace period."""
        modifier = f"-{self._config.sweep_grace_seconds} seconds"

        def _query(conn):
            return conn.execute(
                "SELECT id, file_path FROM artifacts "
                "WHERE type = ? AND created_at <= datetime('now', ?)",
                (_ARTIFACT_TYPE, modifier),
            ).fetchall()

        rows = execute_with_retry(_query, db_path=self._db_path)
        dangling_ids: list[int] = []
        dangling_paths: list[Path] = []
        for row in rows:
            self._check_deadline(deadline)
            try:
                storage_id, sidecar_sha256 = self._parse_storage_ref(
                    row["file_path"]
                )
                record = self._load_sidecar(
                    storage_id, expected_sha256=sidecar_sha256
                )
                payload = self._read_regular_file(self._blob_path(storage_id))
                self._validate_payload(record, payload)
            except CommandOutputIntegrityError:
                dangling_ids.append(int(row["id"]))
                try:
                    storage_id, _ = self._parse_storage_ref(row["file_path"])
                except CommandOutputIntegrityError:
                    continue
                dangling_paths.extend((
                    self._blob_path(storage_id),
                    self._sidecar_path(storage_id),
                ))
        if dangling_ids:
            self._delete_catalog_ids(dangling_ids)
            self._remove_paths(dangling_paths)

    def _delete_catalog_ids(self, artifact_ids: list[int]) -> None:
        if not artifact_ids:
            return

        def _delete(conn):
            placeholders = ",".join("?" for _ in artifact_ids)
            conn.execute(
                f"DELETE FROM artifacts WHERE type = ? AND id IN ({placeholders})",
                [_ARTIFACT_TYPE, *artifact_ids],
            )
            conn.commit()

        try:
            execute_with_retry(_delete, db_path=self._db_path)
        except Exception:
            logger.warning("Could not clean command-output catalog rows", exc_info=True)

    @staticmethod
    def _remove_paths(paths: list[Path]) -> None:
        for path in paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.debug("Could not clean private command-output file", exc_info=True)

    def _new_storage_id(self) -> str:
        for _ in range(8):
            storage_id = secrets.token_hex(24)
            if not self._blob_path(storage_id).exists() and not self._sidecar_path(storage_id).exists():
                return storage_id
        raise CommandOutputStoreError("could not allocate a random storage identity")

    def _blob_path(self, storage_id: str) -> Path:
        return self._root / f"{self._validate_storage_id(storage_id)}.blob"

    def _sidecar_path(self, storage_id: str) -> Path:
        return self._root / f"{self._validate_storage_id(storage_id)}.json"

    def _fsync_directory(self) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        fd = os.open(self._root, flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        if time.monotonic() > deadline:
            raise CommandOutputStoreError("command-output store operation timed out")


__all__ = [
    "CommandOutputConfigError",
    "CommandOutputHandle",
    "CommandOutputIntegrityError",
    "CommandOutputQuotaError",
    "CommandOutputStore",
    "CommandOutputStoreConfig",
    "CommandOutputStoreError",
]
