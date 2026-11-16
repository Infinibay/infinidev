"""Bounded, content-free observations of retained child-run tool-result traces."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sqlite3
from typing import Any

from infinidev.code_intel._db import execute_with_retry

_SCAN_ROWS = 4096
_RESULT_ROWS = 512
_TRACE_BYTES = 16 * 1024
_FIELDS = ("trace", "run_id", "event", "iteration", "source", "tool",
           "output_sha256", "output_bytes", "status", "closure", "dispatches")


def _hex(value: Any, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and all(char in "0123456789abcdef" for char in value))


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _project(conn: sqlite3.Connection, session_id: str) -> list[dict[str, Any]]:
    table = conn.execute("SELECT type FROM pragma_table_list "
                         "WHERE schema='main' AND name='working_memory'").fetchone()
    if table is None or table[0] != "table":
        raise sqlite3.DatabaseError("trace table unavailable")
    # CASE guards JSON parsing and projection size even for corrupt/oversized rows.
    integer_fields = {"iteration", "output_bytes", "dispatches"}
    fields = ",".join(
        f"CASE WHEN json_type(safe_content,'$.{field}')="
        f"'{('integer' if field in integer_fields else 'text')}' "
        f"THEN json_extract(safe_content,'$.{field}') END AS {field}"
        for field in _FIELDS)
    rows = conn.execute(
        "WITH recent AS (SELECT rowid,substr(created_at,1,65) AS created_at,"
        "CASE WHEN typeof(content)='text' AND length(CAST(content AS BLOB))<=? "
        "AND json_valid(content) THEN content ELSE '{}' END AS safe_content,"
        "CASE WHEN typeof(content)='text' AND length(CAST(content AS BLOB))<=? "
        "AND json_valid(content) THEN 1 ELSE 0 END AS valid "
        "FROM working_memory NOT INDEXED WHERE session_id=? AND kind='runtime_trace' "
        "ORDER BY rowid DESC LIMIT ?) "
        f"SELECT rowid,created_at,valid,{fields} FROM recent",
        (_TRACE_BYTES, _TRACE_BYTES, session_id, _SCAN_ROWS + 1),
    ).fetchall()
    return [dict(row) for row in rows]


def read_tool_results(workspace: Path, session_id: str, run_id: str) -> dict[str, Any]:
    """Observe an explicit session/run without loading settings or tool bodies.

    Hashes are host-recorded SHA256 of UTF-8 tool-result text, not revalidated
    artifacts or a judgment of useful progress. A retained window is never proof
    of complete history, successful work, process liveness or session-wide idle.
    """
    result: dict[str, Any] = {
        "schema_version": 1, "session_id": session_id, "run_id": run_id,
        "state": "unavailable", "reason": "missing_database",
        "results": None, "closure": None,
        "fingerprint": "runtime_trace.tool_result.output_sha256/sha256-utf8/v1",
    }
    if not _hex(run_id, 32) or not isinstance(session_id, str) or not session_id:
        return {**result, "reason": "invalid_selector"}
    try:
        directory = Path(workspace).resolve() / ".infinidev"
        database = directory / "infinidev.db"
        if directory.is_symlink() or database.is_symlink():
            return {**result, "reason": "unsafe_database"}
        if not database.exists():
            return result
        if not database.is_file():
            return {**result, "reason": "unsafe_database"}
        rows = execute_with_retry(lambda conn: _project(conn, session_id),
                                  db_path=str(database), read_only=True)
    except (OSError, sqlite3.Error, ValueError, UnicodeError):
        return {**result, "reason": "database_unavailable"}

    scan_clipped = len(rows) > _SCAN_ROWS
    rows = rows[:_SCAN_ROWS]
    if any(row["valid"] != 1 for row in rows):
        return {**result, "reason": "invalid_trace"}
    selected = sorted((row for row in rows if row["run_id"] == run_id),
                      key=lambda row: row["rowid"])
    if not selected:
        return {**result, "reason": "trace_not_found", "coverage": {
            "scope": "retained_window", "history_complete": False,
            "scanned_rows": len(rows), "scan_limit": _SCAN_ROWS,
            "scan_clipped": scan_clipped,
        }}
    tools = []
    closure = None
    for row in selected:
        if (row["trace"] != "turn_provenance" or not _integer(row["iteration"])
                or not isinstance(row["event"], str) or not 0 < len(row["event"]) <= 64
                or not row["event"].isprintable()
                or not isinstance(row["created_at"], str) or len(row["created_at"]) > 64):
            return {**result, "reason": "invalid_trace"}
        if row["event"] == "tool_result":
            if (row["source"] != "dispatcher" or not isinstance(row["tool"], str)
                    or not 0 < len(row["tool"]) <= 128 or not row["tool"].isprintable()
                    or not _hex(row["output_sha256"], 64)
                    or not _integer(row["output_bytes"])):
                return {**result, "reason": "invalid_trace"}
            tools.append({key: row[key] for key in ("rowid", "created_at", "iteration",
                                                   "tool", "output_sha256", "output_bytes")})
        elif row["event"] == "turn_closed":
            if (row["status"] not in {"answered", "cancelled", "exhausted", "failed"}
                    or row["closure"] not in {"finish_turn", "plain_text", "interrupted"}
                    or not _integer(row["dispatches"])):
                return {**result, "reason": "invalid_trace"}
            closure = {key: row[key] for key in ("rowid", "created_at", "status", "closure",
                                                 "dispatches")}

    clipped = len(tools) > _RESULT_ROWS
    tools = tools[-_RESULT_ROWS:]
    latest = tools[-1] if tools else None
    fingerprint = lambda row: (row["tool"], row["output_sha256"])
    trailing = 0
    for row in reversed(tools):
        if fingerprint(row) != fingerprint(latest):
            break
        trailing += 1
    counts = Counter(row["tool"] for row in tools)
    return {**result, "state": "observed", "reason": "retained_trace_window",
            "coverage": {"scope": "retained_window", "history_complete": False,
                         "ordering": "sqlite_rowid", "scanned_rows": len(rows),
                         "scan_limit": _SCAN_ROWS, "scan_clipped": scan_clipped,
                         "result_limit": _RESULT_ROWS, "results_clipped": clipped},
            "results": {"count": len(tools), "latest": latest,
                        "latest_occurrences": sum(fingerprint(row) == fingerprint(latest)
                                                  for row in tools),
                        "trailing_identical": trailing,
                        "last_different": tools[-trailing - 1] if trailing < len(tools) else None,
                        "by_tool": [{"tool": tool, "count": count,
                                     "distinct_output_hashes": len({row["output_sha256"]
                                         for row in tools if row["tool"] == tool})}
                                    for tool, count in sorted(counts.items())]},
            "closure": closure}
