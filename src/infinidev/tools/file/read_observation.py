"""Host-only revisions of files actually opened by a local read tool."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TextIO


@dataclass(frozen=True)
class FileReadObservation:
    path: str
    fingerprint: str


_observations: ContextVar[list[FileReadObservation] | None] = ContextVar(
    "file_read_observations", default=None,
)


@contextmanager
def capture_file_reads() -> Iterator[list[FileReadObservation]]:
    """Collect observations for one dispatch without changing its tool text."""
    observations: list[FileReadObservation] = []
    token = _observations.set(observations)
    try:
        yield observations
    finally:
        _observations.reset(token)


def file_fingerprint(info: os.stat_result) -> str | None:
    """Identify a regular-file revision using filesystem change metadata."""
    if not stat.S_ISREG(info.st_mode):
        return None
    stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    return hashlib.sha256(repr(stamp).encode("ascii")).hexdigest()


def read_observed_lines(stream: TextIO, path: str) -> list[str]:
    """Read normally and observe the opened file when the host requests it."""
    observations = _observations.get()
    if observations is None:
        return stream.readlines()
    try:
        before = file_fingerprint(os.fstat(stream.fileno()))
    except OSError:
        before = None
    lines = stream.readlines()
    if before is not None:
        try:
            after = file_fingerprint(os.fstat(stream.fileno()))
        except OSError:
            after = None
        if before == after:
            observations.append(FileReadObservation(path, before))
    return lines
