"""SQLite storage for code intelligence symbols, references, and imports."""

from __future__ import annotations

import sqlite3
from typing import Any

from infinidev.code_intel.models import Symbol, Reference, Import
from infinidev.code_intel._db import execute_with_retry


def get_file_hash(project_id: int, file_path: str) -> str | None:
    """Return the stored content hash for a file, or None if not indexed."""
    def _query(conn: sqlite3.Connection):
        row = conn.execute(
            "SELECT content_hash FROM ci_files WHERE project_id = ? AND file_path = ?",
            (project_id, file_path),
        ).fetchone()
        return row["content_hash"] if row else None
    return execute_with_retry(_query)


def mark_file_indexed(
    project_id: int, file_path: str, language: str, content_hash: str, symbol_count: int,
) -> None:
    """Update or insert the file index entry."""
    def _upsert(conn: sqlite3.Connection):
        conn.execute(
            """\
            INSERT INTO ci_files (project_id, file_path, language, content_hash, symbol_count, indexed_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(project_id, file_path)
            DO UPDATE SET language=?, content_hash=?, symbol_count=?, indexed_at=CURRENT_TIMESTAMP
            """,
            (project_id, file_path, language, content_hash, symbol_count,
             language, content_hash, symbol_count),
        )
        conn.commit()
    execute_with_retry(_upsert)


def clear_file(project_id: int, file_path: str) -> None:
    """Delete all symbols, references, and imports for a file."""
    def _delete(conn: sqlite3.Connection):
        conn.execute("DELETE FROM ci_symbols WHERE project_id = ? AND file_path = ?", (project_id, file_path))
        conn.execute("DELETE FROM ci_references WHERE project_id = ? AND file_path = ?", (project_id, file_path))
        conn.execute("DELETE FROM ci_imports WHERE project_id = ? AND file_path = ?", (project_id, file_path))
        conn.commit()
    execute_with_retry(_delete)


def store_file_symbols(
    project_id: int,
    file_path: str,
    symbols: list[Symbol],
    references: list[Reference],
    imports: list[Import],
) -> None:
    """Store extracted symbols, references, and imports for a file.

    Clears existing data for the file first (atomic replace).
    """
    def _store(conn: sqlite3.Connection):
        # Clear old data
        conn.execute("DELETE FROM ci_symbols WHERE project_id = ? AND file_path = ?", (project_id, file_path))
        conn.execute("DELETE FROM ci_references WHERE project_id = ? AND file_path = ?", (project_id, file_path))
        conn.execute("DELETE FROM ci_imports WHERE project_id = ? AND file_path = ?", (project_id, file_path))

        # Insert symbols
        if symbols:
            conn.executemany(
                """\
                INSERT INTO ci_symbols
                (project_id, file_path, name, qualified_name, kind, line_start, line_end,
                 column_start, signature, type_annotation, docstring, parent_symbol,
                 visibility, is_async, is_static, is_abstract, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (project_id, file_path, s.name, s.qualified_name, s.kind.value,
                     s.line_start, s.line_end, s.column_start, s.signature,
                     s.type_annotation, s.docstring, s.parent_symbol,
                     s.visibility, s.is_async, s.is_static, s.is_abstract, s.language)
                    for s in symbols
                ],
            )

        # Insert references
        if references:
            conn.executemany(
                """\
                INSERT INTO ci_references
                (project_id, file_path, name, line, column_num, context, ref_kind,
                 resolved_file, resolved_line, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (project_id, file_path, r.name, r.line, r.column, r.context,
                     r.ref_kind, r.resolved_file, r.resolved_line, r.language)
                    for r in references
                ],
            )

        # Insert imports
        if imports:
            conn.executemany(
                """\
                INSERT INTO ci_imports
                (project_id, file_path, source, name, alias, line, is_wildcard,
                 resolved_file, language)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (project_id, file_path, i.source, i.name, i.alias, i.line,
                     i.is_wildcard, i.resolved_file, i.language)
                    for i in imports
                ],
            )

        conn.commit()

    execute_with_retry(_store)


def clear_project(project_id: int) -> None:
    """Delete all code intelligence data for a project."""
    def _delete(conn: sqlite3.Connection):
        conn.execute("DELETE FROM ci_symbols WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM ci_references WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM ci_imports WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM ci_files WHERE project_id = ?", (project_id,))
        conn.commit()
    execute_with_retry(_delete)
