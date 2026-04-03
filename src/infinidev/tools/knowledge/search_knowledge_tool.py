"""Tool for unified cross-source knowledge search using FTS5."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, sanitize_fts5_query
from infinidev.tools.knowledge.search_knowledge_input import SearchKnowledgeInput


class SearchKnowledgeTool(InfinibayBaseTool):
    name: str = "search_knowledge"
    description: str = (
        "Unified search across knowledge sources (findings, reports). "
        "Uses full-text search for fast, relevant results. "
        "Query supports operators: | for OR, & for AND, * for prefix, \"quotes\" for exact phrases. "
        "Example: 'react | vue', 'auth & token*', '\"machine learning\" | AI'."
    )
    args_schema: Type[BaseModel] = SearchKnowledgeInput

    def _run(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 20,
        min_confidence: float = 0.0,
    ) -> str:
        if sources is None:
            sources = ["findings", "reports"]

        project_id = self.project_id

        def _search(conn: sqlite3.Connection) -> list[dict]:
            results = []
            safe_query = sanitize_fts5_query(query)

            if "findings" in sources:
                try:
                    rows = conn.execute(
                        """SELECT f.id, f.topic AS title,
                                  snippet(findings_fts, 1, '<b>', '</b>', '...', 64) AS snippet,
                                  f.confidence, f.finding_type, f.status
                           FROM findings f
                           JOIN findings_fts fts ON f.id = fts.rowid
                           WHERE fts.findings_fts MATCH ?
                             AND (f.project_id = ? OR f.project_id IS NULL)
                             AND f.confidence >= ?
                             AND f.status != 'rejected'
                           ORDER BY rank
                           LIMIT ?""",
                        (safe_query, project_id, min_confidence, limit),
                    ).fetchall()
                    for r in rows:
                        results.append({
                            "source_type": "findings",
                            "id": r["id"],
                            "title": r["title"],
                            "snippet": r["snippet"],
                            "confidence": r["confidence"],
                            "finding_type": r["finding_type"],
                        })
                except sqlite3.OperationalError:
                    pass  # FTS table may not exist

            if "reports" in sources:
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

        self._log_tool_usage(
            f"Searched '{query}' across {sources} — {len(all_results)} results"
        )
        return self._success({
            "query": query,
            "results": all_results,
            "count": len(all_results),
        })

