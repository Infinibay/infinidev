"""InteractionLogger — records tool call events and context messages.

Emits lightweight interaction events from tool execution and stores
the context messages (user input, step titles, step descriptions) that
provoked them.  All writes are append-only and non-blocking — failures
are logged but never propagate to the caller.

Interaction rows are written asynchronously via a dedicated writer
thread to keep the engine's hot path free of SQLite commit latency.
Context rows (``log_context``) remain synchronous because callers
need the ``lastrowid`` immediately, but they are infrequent (~3 per
step).  Call :func:`flush` at step boundaries to commit pending
context rows and drain the interaction queue.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any

from infinidev.code_intel._db import execute_with_retry, get_pooled_connection

logger = logging.getLogger(__name__)


# ── Async interaction writer ───────────────────────────────────────

_QUEUE_MAXSIZE = 10_000  # cap memory; put() blocks if full
_interaction_queue: queue.Queue[tuple | None] = queue.Queue(maxsize=_QUEUE_MAXSIZE)
_writer_thread: threading.Thread | None = None
_writer_lock = threading.Lock()

_INTERACTION_INSERT_SQL = (
    "INSERT INTO cr_interactions "
    "(task_id, session_id, context_id, iteration, event_type, "
    "target, target_type, weight, metadata, was_error, created_at) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def _ensure_writer() -> None:
    """Start (or restart) the background writer thread."""
    global _writer_thread
    if _writer_thread is not None and _writer_thread.is_alive():
        return
    with _writer_lock:
        if _writer_thread is not None and _writer_thread.is_alive():
            return
        _writer_thread = threading.Thread(
            target=_writer_loop, daemon=True, name="cr-writer",
        )
        _writer_thread.start()


def _writer_loop() -> None:
    """Pull interaction rows from the queue and batch-commit them."""
    while True:
        batch: list[tuple] = []
        try:
            item = _interaction_queue.get(timeout=2.0)
        except queue.Empty:
            continue

        if item is None:  # shutdown sentinel
            break

        # Flush sentinel — commit what we have and signal caller
        if isinstance(item, threading.Event):
            if batch:
                _commit_batch(batch)
            item.set()
            continue

        batch.append(item)

        # Drain any additional queued items
        while True:
            try:
                item = _interaction_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                _commit_batch(batch)
                return
            if isinstance(item, threading.Event):
                _commit_batch(batch)
                batch.clear()
                item.set()
                continue
            batch.append(item)

        if batch:
            _commit_batch(batch)


def _commit_batch(batch: list[tuple]) -> None:
    """Insert a batch of interaction rows in one transaction."""
    from infinidev.engine.static_analysis_timer import measure as _sa_measure
    try:
        with _sa_measure("db_write"):
            conn = get_pooled_connection()
            conn.executemany(_INTERACTION_INSERT_SQL, batch)
            conn.commit()
    except Exception:
        logger.debug("Writer batch commit failed (%d rows)", len(batch), exc_info=True)


def flush() -> None:
    """Commit pending context writes and drain the interaction queue.

    Call at step boundaries and at task end to ensure all data is
    persisted before the next phase reads it.  If the writer thread
    died, it is restarted automatically.
    """
    from infinidev.engine.static_analysis_timer import measure as _sa_measure
    # 1. Commit any pending synchronous writes (log_context rows)
    try:
        with _sa_measure("db_write"):
            conn = get_pooled_connection()
            conn.commit()
    except Exception:
        logger.debug("flush() sync commit failed", exc_info=True)

    # 2. If writer is dead, restart it so the queue drains
    if _writer_thread is not None and not _writer_thread.is_alive():
        logger.warning("cr-writer thread died — restarting")
        _ensure_writer()

    # 3. Wait for the writer thread to drain its queue
    if _writer_thread is not None and _writer_thread.is_alive() and not _interaction_queue.empty():
        done = threading.Event()
        _interaction_queue.put(done)
        done.wait(timeout=5.0)

# ── Tool → event mapping ────────────────────────────────────────────

_TOOL_EVENT_MAP: dict[str, tuple[str, str, float]] = {
    # tool_name → (event_type, target_type, weight)
    # File reads
    "read_file":        ("file_read",    "file", 1.0),
    "partial_read":     ("file_read",    "file", 1.0),
    "list_directory":   ("file_read",    "file", 0.3),
    "code_search":      ("file_read",    "file", 0.5),
    "glob":             ("file_read",    "file", 0.5),
    # File writes
    "replace_lines":    ("file_write",   "file", 2.0),
    "create_file":      ("file_write",   "file", 2.0),
    "edit_file":        ("file_write",   "file", 2.0),
    "multi_edit_file":  ("file_write",   "file", 2.0),
    "apply_patch":      ("file_write",   "file", 2.0),
    "add_content_above": ("file_write",  "file", 2.0),
    "add_content_below": ("file_write",  "file", 2.0),
    # Symbol reads
    "get_symbol_code":  ("symbol_read",  "symbol", 1.0),
    "list_symbols":     ("symbol_read",  "symbol", 0.5),
    "search_symbols":   ("symbol_read",  "symbol", 0.5),
    "find_references":  ("symbol_read",  "symbol", 1.0),
    # Symbol writes
    "edit_symbol":      ("symbol_write", "symbol", 2.5),
    "add_symbol":       ("symbol_write", "symbol", 2.5),
    "remove_symbol":    ("symbol_write", "symbol", 2.5),
    # Findings
    "record_finding":   ("finding_create", "finding", 1.5),
    "search_findings":  ("finding_read",   "finding", 0.8),
    "read_findings":    ("finding_read",   "finding", 0.8),
    # Shell
    "execute_command":  ("command_exec", "file", 0.3),
}

# Argument names to extract as the interaction target, in priority order
_TARGET_ARGS: dict[str, list[str]] = {
    "file":    ["path", "file_path", "directory", "dir_path"],
    "symbol":  ["qualified_name", "name", "symbol"],
    "finding": ["topic", "query", "finding_id"],
}


def _extract_target(tool_name: str, target_type: str, arguments: dict[str, Any]) -> str | None:
    """Extract the interaction target from tool call arguments.

    For file targets, normalizes to relative paths (strips workspace prefix)
    to avoid duplicate entries for the same file.
    """
    for arg_name in _TARGET_ARGS.get(target_type, []):
        val = arguments.get(arg_name)
        if val is not None:
            target = str(val)
            # Normalize file paths to relative
            if target_type == "file" and os.path.isabs(target):
                workspace = os.getcwd()
                if target.startswith(workspace + "/"):
                    target = target[len(workspace) + 1:]
            return target
    return None


def classify_tool_call(
    tool_name: str, arguments: dict[str, Any],
) -> tuple[str, str, str, float] | None:
    """Classify a tool call into an interaction event.

    Returns ``(event_type, target, target_type, weight)`` or *None*
    if the tool is not tracked (e.g. ``step_complete``, ``think``).
    """
    mapping = _TOOL_EVENT_MAP.get(tool_name)
    if mapping is None:
        return None
    event_type, target_type, weight = mapping
    target = _extract_target(tool_name, target_type, arguments)
    if target is None:
        return None
    return event_type, target, target_type, weight


# ── Persistence ──────────────────────────────────────────────────────

def log_context(
    session_id: str,
    task_id: str,
    context_type: str,
    content: str,
    iteration: int | None = None,
    step_index: int | None = None,
) -> int | None:
    """Store a context message and return its row id.

    Context types form the *escalera*:
      - ``task_input``        (level 1 — broadest)
      - ``step_title``        (level 2)
      - ``step_description``  (level 3 — most specific)

    The INSERT is NOT committed immediately — call :func:`flush` at
    step boundaries to batch multiple writes into one WAL checkpoint.
    ``lastrowid`` is available before commit in SQLite.
    """
    now = time.time()
    try:
        def _insert(conn):
            cur = conn.execute(
                "INSERT INTO cr_contexts "
                "(task_id, session_id, context_type, content, iteration, step_index, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id, session_id, context_type, content, iteration, step_index, now),
            )
            return cur.lastrowid
        return execute_with_retry(_insert)
    except Exception:
        logger.debug("Failed to log context", exc_info=True)
        return None


def log_interaction(
    session_id: str,
    task_id: str,
    context_id: int | None,
    iteration: int,
    event_type: str,
    target: str,
    target_type: str,
    weight: float,
    metadata: dict[str, Any] | None = None,
    *,
    was_error: bool = False,
) -> None:
    """Append an interaction event to the async write queue.

    The row is written by a background thread, so this call returns
    immediately (~0ms) without blocking the engine.

    The ``was_error`` flag is persisted so the predictive channel
    can exclude failed tool calls from historical scoring, and the
    snapshot step can compute productivity without counting errors.
    """
    _ensure_writer()
    meta_json = json.dumps(metadata) if metadata else None
    _interaction_queue.put((
        task_id, session_id, context_id, iteration,
        event_type, target, target_type, weight, meta_json,
        1 if was_error else 0, time.time(),
    ))


def log_tool_call(
    session_id: str,
    task_id: str,
    context_id: int | None,
    iteration: int,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    was_error: bool = False,
) -> None:
    """Convenience wrapper: classify a tool call and log the interaction."""
    classified = classify_tool_call(tool_name, arguments)
    if classified is None:
        return
    event_type, target, target_type, weight = classified
    log_interaction(
        session_id, task_id, context_id, iteration,
        event_type, target, target_type, weight,
        was_error=was_error,
    )


def compute_context_embedding(context_id: int) -> None:
    """Compute and store the embedding for a context entry.

    Designed to be called in a background thread so it doesn't block
    the main loop.  Silently no-ops on failure.
    """
    try:
        def _compute(conn):
            row = conn.execute(
                "SELECT content FROM cr_contexts WHERE id = ?", (context_id,),
            ).fetchone()
            if row is None:
                return
            from infinidev.tools.base.embeddings import store_context_embedding
            store_context_embedding(conn, context_id, row["content"])
            conn.commit()
        execute_with_retry(_compute)
    except Exception:
        logger.debug("Failed to compute context embedding", exc_info=True)


# Productivity multipliers used by snapshot_session_scores.  The
# predictive channel later multiplies historical interaction weights
# by these, so "productive" past actions propagate more signal than
# "exploratory" or "error" ones.
_PRODUCTIVITY_EDITED = 1.5       # any write hit this target → useful
_PRODUCTIVITY_NEUTRAL = 1.0      # single read, never edited → normal
_PRODUCTIVITY_EXPLORATORY = 0.6  # repeated reads, never edited → penalty


def snapshot_session_scores(session_id: str, task_id: str) -> None:
    """Pre-compute per-node scores for this session and persist them.

    Called once at task end so that future sessions can cheaply query
    historical relevance without re-scanning raw interactions.

    Phase 2 v3 changes:

    * Excludes interactions where ``was_error = 1`` from all
      aggregations — a failed tool call shouldn't count as evidence
      that the target was relevant.
    * Computes per-target ``productivity`` multiplier based on the
      read/write pattern over non-errored interactions:
        - any write  → _PRODUCTIVITY_EDITED (the model actually
          modified the target, so it was meaningfully useful)
        - single read, no write → _PRODUCTIVITY_NEUTRAL
        - multiple reads, no write → _PRODUCTIVITY_EXPLORATORY
          (the model re-read without committing — likely dead-end
          exploration)
    * Also stores ``was_edited`` as a fast-lookup boolean.

    The predictive channel later LEFT JOINs cr_session_scores on
    (session_id, target, target_type) and multiplies historical
    interaction contributions by ``productivity``, so "productive"
    past sessions carry more weight in the ranking.
    """
    # Drain async writer — the SELECT below needs all interactions committed.
    flush()
    now = time.time()
    try:
        def _snapshot(conn):
            rows = conn.execute(
                "SELECT target, target_type, "
                "       SUM(CASE WHEN was_error = 0 THEN weight ELSE 0 END) AS total_weight, "
                "       COUNT(*) AS cnt, "
                "       SUM(CASE WHEN weight >= 2.0 AND was_error = 0 THEN 1 ELSE 0 END) AS write_count, "
                "       SUM(CASE WHEN weight <  2.0 AND was_error = 0 THEN 1 ELSE 0 END) AS read_count "
                "FROM cr_interactions "
                "WHERE session_id = ? AND task_id = ? "
                "GROUP BY target, target_type",
                (session_id, task_id),
            ).fetchall()
            for row in rows:
                write_count = row["write_count"] or 0
                read_count = row["read_count"] or 0
                was_edited = write_count > 0
                if was_edited:
                    productivity = _PRODUCTIVITY_EDITED
                elif read_count == 1:
                    productivity = _PRODUCTIVITY_NEUTRAL
                else:
                    productivity = _PRODUCTIVITY_EXPLORATORY

                conn.execute(
                    "INSERT INTO cr_session_scores "
                    "(task_id, session_id, target, target_type, score, "
                    " access_count, productivity, was_edited, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (task_id, session_id, row["target"], row["target_type"],
                     row["total_weight"], row["cnt"],
                     productivity, 1 if was_edited else 0, now),
                )
            conn.commit()
        execute_with_retry(_snapshot)
    except Exception:
        logger.debug("Failed to snapshot session scores", exc_info=True)
