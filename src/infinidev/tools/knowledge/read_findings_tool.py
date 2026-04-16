"""Tool for reading/searching research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, sanitize_fts5_query
from infinidev.tools.knowledge.read_findings_input import ReadFindingsInput


class ReadFindingsTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "read_findings"
    description: str = (
        "Read and search research findings. Supports full-text search "
        "and filtering by session, confidence, and type. By default only "
        "returns findings for the current session; pass session_id='0' to see all."
    )
    args_schema: Type[BaseModel] = ReadFindingsInput

    def _run(
        self,
        query: str | None = None,
        session_id: str | None = None,
        min_confidence: float = 0.0,
        finding_type: str | None = None,
        limit: int = 50,
    ) -> str:
        project_id = self.project_id

        # Resolve effective session_id: explicit arg > context > None
        # session_id='0' means "show all project findings" (no filter).
        if session_id is None:
            effective_session_id = self.session_id  # from agent context
        elif session_id == "0":
            effective_session_id = None  # explicitly disabled
        else:
            effective_session_id = session_id

        def _read(conn: sqlite3.Connection) -> list[dict]:
            if query:
                # Use FTS5 for full-text search
                conditions = ["f.confidence >= ?"]
                params: list = [min_confidence]

                if project_id:
                    conditions.append("f.project_id = ?")
                    params.append(project_id)
                if effective_session_id:
                    conditions.append("f.session_id = ?")
                    params.append(effective_session_id)
                if finding_type:
                    conditions.append("f.finding_type = ?")
                    params.append(finding_type)

                where = " AND ".join(conditions)
                params.append(limit)

                safe_query = sanitize_fts5_query(query)
                rows = conn.execute(
                    f"""SELECT f.id, f.topic, f.content, f.confidence,
                               f.agent_id, f.status, f.finding_type,
                               f.sources_json, f.validation_method,
                               f.reproducibility_score, f.session_id,
                               f.created_at
                        FROM findings f
                        JOIN findings_fts fts ON f.id = fts.rowid
                        WHERE fts.findings_fts MATCH ?
                          AND {where}
                        ORDER BY f.confidence DESC
                        LIMIT ?""",
                    [safe_query] + params,
                ).fetchall()
            else:
                conditions = ["confidence >= ?"]
                params = [min_confidence]

                if project_id:
                    conditions.append("project_id = ?")
                    params.append(project_id)
                if effective_session_id:
                    conditions.append("session_id = ?")
                    params.append(effective_session_id)
                if finding_type:
                    conditions.append("finding_type = ?")
                    params.append(finding_type)

                where = " AND ".join(conditions)
                params.append(limit)

                rows = conn.execute(
                    f"""SELECT id, topic, content, confidence,
                               agent_id, status, finding_type,
                               sources_json, validation_method,
                               reproducibility_score, session_id,
                               created_at
                        FROM findings
                        WHERE {where}
                        ORDER BY confidence DESC
                        LIMIT ?""",
                    params,
                ).fetchall()

            return [dict(r) for r in rows]

        try:
            findings = execute_with_retry(_read)
        except Exception as e:
            return self._error(f"Failed to read findings: {e}")

        return self._success({"findings": findings, "count": len(findings)})

