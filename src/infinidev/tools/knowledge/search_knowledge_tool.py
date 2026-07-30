"""Tool for unified cross-source knowledge search using FTS5."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, sanitize_fts5_query
from infinidev.tools.knowledge.search_knowledge_input import SearchKnowledgeInput


class SearchKnowledgeTool(InfinibayBaseTool):
    name: str = "search_knowledge"
    is_read_only: bool = True
    description: str = (
        "Search or browse saved knowledge (findings, reports). With a query "
        "this is full-text search — operators: | OR, & AND, * prefix, "
        "\"exact phrase\" — and results come back as snippets. Without one it "
        "lists findings by filter, with their full content."
    )
    args_schema: Type[BaseModel] = SearchKnowledgeInput

    def _run(
        self,
        query: str = "",
        sources: list[str] | None = None,
        limit: int = 20,
        min_confidence: float = 0.0,
        session_id: str | None = None,
        finding_type: str | None = None,
    ) -> str:
        if sources is None:
            sources = ["findings", "reports"]

        project_id = self.project_id
        query = (query or "").strip()

        # session_id: unset falls back to the agent's own session, '0' is the
        # explicit "every session in this project" escape hatch.
        if session_id is None:
            effective_session_id = self.session_id
        elif session_id == "0":
            effective_session_id = None
        else:
            effective_session_id = session_id

        def _search(conn: sqlite3.Connection) -> list[dict]:
            results = []
            safe_query = sanitize_fts5_query(query) if query else ""

            if "findings" in sources:
                try:
                    rows = self._findings_rows(
                        conn,
                        safe_query,
                        project_id=project_id,
                        session_id=effective_session_id,
                        finding_type=finding_type,
                        min_confidence=min_confidence,
                        limit=limit,
                    )
                    results.extend(rows)
                except sqlite3.OperationalError:
                    pass  # FTS table may not exist

            # An FTS MATCH needs something to match; browsing has no query,
            # so reports simply do not participate in that mode.
            if "reports" in sources and safe_query:
                try:
                    rows = conn.execute(
                        """SELECT a.id, a.file_path AS title,
                                  snippet(artifacts_fts, 1, '<b>', '</b>', '...', 64) AS snippet
                           FROM artifacts a
                           JOIN artifacts_fts ON a.id = artifacts_fts.rowid
                           WHERE artifacts_fts MATCH ?
                             AND a.type = 'report'
                             AND a.project_id = ?
                           ORDER BY rank
                           LIMIT ?""",
                        (safe_query, project_id, limit),
                    ).fetchall()
                    for r in rows:
                        results.append({
                            "source_type": "reports",
                            "id": r["id"],
                            "title": r["title"],
                            "snippet": r["snippet"],
                        })
                except sqlite3.OperationalError:
                    pass  # FTS table may not exist

            return results

        try:
            all_results = execute_with_retry(_search)
        except Exception as e:
            return self._error(f"Knowledge search failed: {e}")

        what = f"Searched '{query}'" if query else "Browsed"
        self._log_tool_usage(f"{what} across {sources} — {len(all_results)} results")
        return self._success({
            "query": query,
            "results": all_results,
            "count": len(all_results),
        })

    @staticmethod
    def _findings_rows(
        conn: sqlite3.Connection,
        safe_query: str,
        *,
        project_id,
        session_id,
        finding_type,
        min_confidence: float,
        limit: int,
    ) -> list[dict]:
        """Findings, searched or browsed.

        The two modes differ in more than a WHERE clause. A search ranks by
        FTS relevance and returns a snippet, which only means anything
        relative to a query; browsing ranks by confidence and returns the
        finding's content, because there is nothing to excerpt around.
        """
        conditions = ["f.confidence >= ?"]
        params: list = [min_confidence]
        if project_id:
            conditions.append("(f.project_id = ? OR f.project_id IS NULL)")
            params.append(project_id)
        if session_id:
            conditions.append("f.session_id = ?")
            params.append(session_id)
        if finding_type:
            conditions.append("f.finding_type = ?")
            params.append(finding_type)
        conditions.append("f.status != 'rejected'")
        where = " AND ".join(conditions)

        if safe_query:
            rows = conn.execute(
                f"""SELECT f.id, f.topic AS title, f.confidence, f.finding_type,
                           f.status, f.session_id,
                           snippet(findings_fts, 1, '<b>', '</b>', '...', 64) AS snippet
                    FROM findings f
                    JOIN findings_fts fts ON f.id = fts.rowid
                    WHERE fts.findings_fts MATCH ? AND {where}
                    ORDER BY rank
                    LIMIT ?""",
                [safe_query] + params + [limit],
            ).fetchall()
        else:
            rows = conn.execute(
                f"""SELECT f.id, f.topic AS title, f.confidence, f.finding_type,
                           f.status, f.session_id, f.content, f.created_at,
                           f.sources_json
                    FROM findings f
                    WHERE {where}
                    ORDER BY f.confidence DESC
                    LIMIT ?""",
                params + [limit],
            ).fetchall()

        return [{"source_type": "findings", **dict(r)} for r in rows]

