"""InteractionLogger — records tool call events and context messages.

Emits lightweight interaction events from tool execution and stores
the context messages (user input, step titles, step descriptions) that
provoked them.  All writes are append-only and non-blocking — failures
are logged but never propagate to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from infinidev.code_intel._db import execute_with_retry

logger = logging.getLogger(__name__)

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
            conn.commit()
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
) -> None:
    """Append an interaction event to the log."""
    now = time.time()
    meta_json = json.dumps(metadata) if metadata else None
    try:
        def _insert(conn):
            conn.execute(
                "INSERT INTO cr_interactions "
                "(task_id, session_id, context_id, iteration, event_type, "
                "target, target_type, weight, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (task_id, session_id, context_id, iteration,
                 event_type, target, target_type, weight, meta_json, now),
            )
            conn.commit()
        execute_with_retry(_insert)
    except Exception:
        logger.debug("Failed to log interaction", exc_info=True)


def log_tool_call(
    session_id: str,
    task_id: str,
    context_id: int | None,
    iteration: int,
    tool_name: str,
    arguments: dict[str, Any],
) -> None:
    """Convenience wrapper: classify a tool call and log the interaction."""
    classified = classify_tool_call(tool_name, arguments)
    if classified is None:
        return
    event_type, target, target_type, weight = classified
    log_interaction(
        session_id, task_id, context_id, iteration,
        event_type, target, target_type, weight,
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


def snapshot_session_scores(session_id: str, task_id: str) -> None:
    """Pre-compute per-node scores for this session and persist them.

    Called once at task end so that future sessions can cheaply query
    historical relevance without re-scanning raw interactions.
    """
    now = time.time()
    try:
        def _snapshot(conn):
            rows = conn.execute(
                "SELECT target, target_type, SUM(weight) as total_weight, COUNT(*) as cnt "
                "FROM cr_interactions "
                "WHERE session_id = ? AND task_id = ? "
                "GROUP BY target, target_type",
                (session_id, task_id),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT INTO cr_session_scores "
                    "(task_id, session_id, target, target_type, score, access_count, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (task_id, session_id, row["target"], row["target_type"],
                     row["total_weight"], row["cnt"], now),
                )
            conn.commit()
        execute_with_retry(_snapshot)
    except Exception:
        logger.debug("Failed to snapshot session scores", exc_info=True)
