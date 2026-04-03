"""Tool for getting a compact summary of findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.knowledge.summarize_findings_input import SummarizeFindingsInput


class SummarizeFindingsTool(InfinibayBaseTool):
    name: str = "summarize_findings"
    description: str = (
        "Get a compact overview of findings: counts by type, confidence "
        "stats, topic list with status, and status breakdown. Useful "
        "before writing reports to see what you have."
    )
    args_schema: Type[BaseModel] = SummarizeFindingsInput

    def _run(self, session_id: str | None = None) -> str:
        project_id = self.project_id

        # Resolve session_id: None → current session, '0' → all project findings
        if session_id is None:
            session_id = self.session_id

        def _query(conn: sqlite3.Connection) -> dict:
            # Build WHERE clause
            if session_id is not None and session_id != "0":
                where = "WHERE f.session_id = ?"
                params: tuple = (session_id,)
            elif project_id:
                where = "WHERE f.project_id = ?"
                params = (project_id,)
            else:
                return {
                    "error": "No session_id or project_id available",
                    "by_type": [],
                    "topics": [],
                    "total_count": 0,
                    "status_breakdown": {},
                }

            # Group by finding_type: count, avg/min/max confidence
            by_type = conn.execute(
                f"""\
                SELECT finding_type,
                       COUNT(*) as count,
                       ROUND(AVG(confidence), 2) as avg_confidence,
                       ROUND(MIN(confidence), 2) as min_confidence,
                       ROUND(MAX(confidence), 2) as max_confidence
                FROM findings f
                {where}
                GROUP BY finding_type
                ORDER BY count DESC
                """,
                params,
            ).fetchall()

            # Topic list (no content, just metadata)
            topics = conn.execute(
                f"""\
                SELECT id, topic, finding_type, confidence, status
                FROM findings f
                {where}
                ORDER BY finding_type, confidence DESC
                """,
                params,
            ).fetchall()

            # Status breakdown
            status_rows = conn.execute(
                f"""\
                SELECT status, COUNT(*) as count
                FROM findings f
                {where}
                GROUP BY status
                """,
                params,
            ).fetchall()

            return {
                "by_type": [dict(r) for r in by_type],
                "topics": [dict(r) for r in topics],
                "total_count": sum(r["count"] for r in by_type),
                "status_breakdown": {r["status"]: r["count"] for r in status_rows},
            }

        result = execute_with_retry(_query)
        scope = f"session {session_id}" if session_id is not None and session_id != "0" else "project"
        self._log_tool_usage(
            f"Summarized {result['total_count']} findings for {scope}"
        )
        return self._success(result)

