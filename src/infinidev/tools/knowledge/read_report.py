"""Tool for reading research reports."""

import os
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class ReadReportInput(BaseModel):
    report_id: int | None = Field(
        default=None, description="Artifact ID of the report"
    )
    session_id: str | None = Field(
        default=None, description="Session ID to find associated report"
    )
    file_path: str | None = Field(
        default=None, description="Full or partial file path to match"
    )


class ReadReportTool(InfinibayBaseTool):
    name: str = "read_report"
    description: str = (
        "Read a research report. Looks up by artifact ID, session ID, or "
        "file path (fuzzy basename match). Content is read from the "
        "database first; falls back to filesystem only when needed."
    )
    args_schema: Type[BaseModel] = ReadReportInput

    def _run(
        self,
        report_id: int | None = None,
        session_id: str | None = None,
        file_path: str | None = None,
    ) -> str:
        if report_id is None and session_id is None and file_path is None:
            return self._error(
                "Provide at least one of: report_id, session_id, or file_path"
            )

        project_id = self.project_id

        def _lookup(conn: sqlite3.Connection) -> dict | None:
            row = None

            # Strategy 1: direct ID lookup
            if report_id is not None:
                row = conn.execute(
                    """SELECT id, file_path, description, content
                       FROM artifacts
                       WHERE id = ? AND type = 'report'""",
                    (report_id,),
                ).fetchone()

            # Strategy 2: find by session_id
            if row is None and session_id is not None:
                row = conn.execute(
                    """SELECT id, file_path, description, content
                       FROM artifacts
                       WHERE session_id = ? AND type = 'report'
                         AND project_id = ?
                       ORDER BY created_at DESC
                       LIMIT 1""",
                    (session_id, project_id),
                ).fetchone()

            # Strategy 3: fuzzy file_path match (basename LIKE)
            if row is None and file_path is not None:
                basename = os.path.basename(file_path)
                row = conn.execute(
                    """SELECT id, file_path, description, content
                       FROM artifacts
                       WHERE type = 'report'
                         AND project_id = ?
                         AND file_path LIKE ?
                       ORDER BY created_at DESC
                       LIMIT 1""",
                    (project_id, f"%{basename.replace('%', '').replace('_', '')}%"),
                ).fetchone()

            return dict(row) if row else None

        try:
            result = execute_with_retry(_lookup)
        except Exception as e:
            return self._error(f"Failed to lookup report: {e}")

        if result is None:
            return self._error("Report not found")

        # Prefer DB content; fall back to filesystem
        content = result.get("content")
        if not content:
            fpath = result.get("file_path", "")
            if fpath and os.path.exists(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    return self._error(f"Failed to read report file: {e}")
            else:
                return self._error(
                    f"Report found in DB (id={result['id']}) but content "
                    f"is empty and file not accessible: {fpath}"
                )

        return self._success({
            "artifact_id": result["id"],
            "file_path": result.get("file_path", ""),
            "description": result.get("description", ""),
            "content": content,
        })
