"""SQLite database layer for code intelligence and tools.

This module is the canonical home for the SQLite connection helpers
and the ``execute_with_retry`` wrapper. It lives under ``code_intel``
deliberately: importing it does NOT trigger ``infinidev/tools/__init__.py``
(which loads every tool class and transitively pulls litellm — adding
~4 seconds of cold-start to any code-intel query).

``infinidev.tools.base.db`` re-exports from here for backward
compatibility with the rest of the codebase.
"""

import logging
import random
import re
import sqlite3
import threading
import time
from typing import Any, Callable, TypeVar

from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _new_connection(db_path: str) -> sqlite3.Connection:
    """Open a brand-new SQLite connection with the project's pragmas."""
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        # Install the busy handler before journal_mode: switching a new
        # connection into WAL itself needs an exclusive lock.
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA mmap_size = 67108864")  # 64MB memory-mapped I/O
        conn.execute("PRAGMA wal_autocheckpoint = 1000")  # defer checkpoints
        return conn
    except sqlite3.Error:
        conn.close()
        raise


# Per-thread connection cache. SQLite connections aren't safe to share
# across threads (default ``check_same_thread=True``), so we keep one
# per thread. The vast majority of writes happen on the engine's main
# thread, so in practice the cache holds exactly one connection.
_conn_cache = threading.local()


def get_pooled_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Return a thread-local cached connection, opening one if needed."""
    if db_path is None:
        db_path = settings.DB_PATH

    cached: sqlite3.Connection | None = getattr(_conn_cache, "conn", None)
    cached_path: str | None = getattr(_conn_cache, "path", None)
    if cached is not None and cached_path == db_path:
        return cached

    conn = _new_connection(db_path)
    _conn_cache.conn = conn
    _conn_cache.path = db_path
    return conn


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a brand-new connection with required pragmas."""
    return _new_connection(db_path or get_db_path())


def get_db_path() -> str:
    """Get database path from settings."""
    return settings.DB_PATH


def execute_with_retry(
    fn: Callable[[sqlite3.Connection], T],
    db_path: str | None = None,
    max_retries: int | None = None,
    base_delay: float | None = None,
) -> T:
    """Execute fn(conn) with exponential backoff retry."""
    if db_path is None:
        db_path = settings.DB_PATH
    if max_retries is None:
        max_retries = settings.MAX_RETRIES
    if base_delay is None:
        base_delay = settings.RETRY_BASE_DELAY

    from infinidev.engine.static_analysis_timer import measure as _sa_measure
    with _sa_measure("db_write"):
        for attempt in range(max_retries):
            conn: sqlite3.Connection | None = None
            try:
                # Opening the connection is part of the retried operation:
                # PRAGMA journal_mode can contend before fn ever runs.
                conn = get_pooled_connection(db_path)
                return fn(conn)
            except sqlite3.DatabaseError as error:
                err_msg = str(error).lower()
                locked = "locked" in err_msg or "busy" in err_msg
                if locked and attempt < max_retries - 1:
                    if conn is not None:
                        try:
                            conn.rollback()
                        except sqlite3.Error:
                            pass
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 0.1)
                    time.sleep(delay)
                    continue

                stale = (
                    "not a database" in err_msg
                    or "disk image is malformed" in err_msg
                    or "no such table" in err_msg
                )
                if stale and attempt < max_retries - 1:
                    if conn is not None:
                        try:
                            conn.close()
                        except sqlite3.Error:
                            pass
                    _conn_cache.conn = None
                    _conn_cache.path = None
                    continue

                if conn is not None:
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                    try:
                        conn.close()
                    except sqlite3.Error:
                        pass
                _conn_cache.conn = None
                _conn_cache.path = None
                raise
            except Exception:
                if conn is not None:
                    try:
                        conn.rollback()
                    except sqlite3.Error:
                        pass
                raise
        raise sqlite3.OperationalError(f"Database busy after {max_retries} retries")


class DBConnection:
    """Context manager for database connections."""
    def __init__(self, db_path: str | None = None):
        self._db_path = db_path or settings.DB_PATH
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self._conn = get_connection(self._db_path)
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn:
            if exc_type is None:
                self._conn.commit()
            else:
                self._conn.rollback()
            self._conn.close()
            self._conn = None


def sanitize_fts5_query(query: str) -> str:
    """Parse a search query with operators into safe FTS5 MATCH syntax."""
    query = query.strip()
    if not query:
        return '""'

    phrases: list[str] = []
    def _capture_phrase(m: re.Match) -> str:
        phrases.append(m.group(0))
        return f"\x00PH{len(phrases) - 1}\x00"

    normalized = re.sub(r'"[^"]*"', _capture_phrase, query)
    or_groups = re.split(r'\s*\|\s*|\s+OR\s+', normalized, flags=re.IGNORECASE)

    fts_or_parts: list[str] = []
    for group in or_groups:
        group = group.strip()
        if not group:
            continue
        and_tokens = re.split(r'\s*&\s*|\s+AND\s+|\s+', group, flags=re.IGNORECASE)
        fts_and_parts: list[str] = []
        for token in and_tokens:
            token = token.strip()
            if not token:
                continue
            ph_match = re.match(r'\x00PH(\d+)\x00$', token)
            if ph_match:
                fts_and_parts.append(phrases[int(ph_match.group(1))])
            elif token.endswith('*') and len(token) > 1:
                fts_and_parts.append(f'"{token[:-1]}" *')
            else:
                clean = token.replace('"', '')
                if clean:
                    fts_and_parts.append(f'"{clean}"')
        if fts_and_parts:
            fts_or_parts.append(" ".join(fts_and_parts))

    if not fts_or_parts:
        return '""'
    return " OR ".join(fts_or_parts)


def parse_query_or_terms(query: str) -> list[str]:
    """Split a query on | / OR into sub-queries for multi-embedding search."""
    query = query.strip()
    if not query:
        return [query]

    phrases: list[str] = []
    def _capture(m: re.Match) -> str:
        phrases.append(m.group(0)[1:-1])
        return f"\x00PH{len(phrases) - 1}\x00"

    normalized = re.sub(r'"[^"]*"', _capture, query)
    or_groups = re.split(r'\s*\|\s*|\s+OR\s+', normalized, flags=re.IGNORECASE)

    terms: list[str] = []
    for group in or_groups:
        group = group.strip()
        if not group:
            continue
        group = re.sub(r'\s*&\s*|\s+AND\s+', ' ', group, flags=re.IGNORECASE)
        for i, ph in enumerate(phrases):
            group = group.replace(f"\x00PH{i}\x00", ph)
        group = group.replace('*', '').replace('"', '').strip()
        if group:
            terms.append(group)

    return terms if terms else [query]
