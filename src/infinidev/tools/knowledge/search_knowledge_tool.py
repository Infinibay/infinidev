"""One interface for textual, browsing, and semantic knowledge retrieval."""

import sqlite3
from typing import Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import (
    execute_with_retry,
    parse_query_or_terms,
    sanitize_fts5_query,
)
from infinidev.tools.knowledge.search_knowledge_input import SearchKnowledgeInput


class SearchKnowledgeTool(InfinibayBaseTool):
    name: str = "search_knowledge"
    is_read_only: bool = True
    description: str = (
        "Search or browse saved knowledge through one interface. mode='text' uses "
        "full-text operators (|, &, *, exact phrases) across findings/reports, or "
        "browses findings when query is empty. mode='semantic' finds conceptually "
        "similar findings and requires a query."
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
        mode: str = "text",
        threshold: float = 0.65,
        include_content: bool = False,
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

        if mode == "semantic":
            if not query:
                return self._error("semantic mode requires a non-empty query")
            return self._semantic_findings(
                query=query,
                project_id=project_id,
                session_id=effective_session_id,
                finding_type=finding_type,
                min_confidence=min_confidence,
                threshold=threshold,
                include_content=include_content,
                limit=limit,
            )

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
            "mode": "text",
            "query": query,
            "results": all_results,
            "count": len(all_results),
        })

    def _semantic_findings(
        self,
        *,
        query: str,
        project_id,
        session_id: str | None,
        finding_type: str | None,
        min_confidence: float,
        threshold: float,
        include_content: bool,
        limit: int,
    ) -> str:
        """Conceptual finding search behind the same public tool contract."""

        def _fetch(conn: sqlite3.Connection) -> list[dict]:
            conditions = ["status != 'rejected'", "confidence >= ?"]
            params: list = [min_confidence]
            if project_id:
                conditions.append("(project_id = ? OR project_id IS NULL)")
                params.append(project_id)
            if session_id:
                conditions.append("session_id = ?")
                params.append(session_id)
            if finding_type:
                conditions.append("finding_type = ?")
                params.append(finding_type)
            rows = conn.execute(
                "SELECT id, topic, content, sources_json, session_id, confidence, "
                "finding_type, status, created_at, embedding, embedding_space "
                "FROM findings WHERE "
                + " AND ".join(conditions)
                + " ORDER BY created_at DESC LIMIT 500",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

        try:
            candidates = execute_with_retry(_fetch)
            if not candidates:
                return self._success({
                    "mode": "semantic", "query": query, "results": [], "count": 0,
                })

            from infinidev.tools.base.dedup import _cosine_similarity
            from infinidev.tools.base.embeddings import (
                current_embedding_space,
                embed_passages,
                embed_queries,
                embedding_from_blob,
                embedding_is_current,
            )

            query_vectors = [
                np.asarray(item) for item in embed_queries(parse_query_or_terms(query))
            ]
            current_dim = int(query_vectors[0].shape[0])
            current_space = current_embedding_space()
            stale = [
                item for item in candidates
                if not embedding_is_current(
                    item.get("embedding"),
                    item.get("embedding_space"),
                    live_space=current_space,
                    dim=current_dim,
                )
            ]
            generated_vectors = embed_passages([
                f"{item['topic']} {(item.get('content') or '')[:500]}" for item in stale
            ]) if stale else []
            regenerated = {
                item["id"]: np.asarray(vector, dtype=np.float32)
                for item, vector in zip(stale, generated_vectors, strict=True)
            }
            if regenerated:
                def _store_refreshed(conn: sqlite3.Connection) -> None:
                    conn.executemany(
                        "UPDATE findings SET embedding = ?, embedding_space = ? "
                        "WHERE id = ?",
                        [
                            (vector.tobytes(), current_space, finding_id)
                            for finding_id, vector in regenerated.items()
                        ],
                    )
                    conn.commit()

                execute_with_retry(_store_refreshed)

            results: list[dict] = []
            for item in candidates:
                vector = (
                    regenerated[item["id"]]
                    if item["id"] in regenerated
                    else embedding_from_blob(item["embedding"])
                )
                similarity = max(
                    _cosine_similarity(query_vector, vector)
                    for query_vector in query_vectors
                )
                if similarity < threshold:
                    continue
                result = dict(item)
                result.pop("embedding", None)
                result.pop("embedding_space", None)
                if not include_content:
                    result.pop("content", None)
                    result.pop("sources_json", None)
                result["source_type"] = "findings"
                result["title"] = result.pop("topic")
                result["similarity"] = round(similarity, 4)
                results.append(result)

            results.sort(key=lambda item: item["similarity"], reverse=True)
            results = results[:limit]
            return self._success({
                "mode": "semantic",
                "query": query,
                "results": results,
                "count": len(results),
                "total_candidates": len(candidates),
                "threshold": threshold,
            })
        except Exception as exc:
            return self._error(f"Semantic knowledge search failed: {exc}")

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
