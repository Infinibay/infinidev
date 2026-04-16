"""Tool for searching findings by semantic similarity."""

import sqlite3
from typing import Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, parse_query_or_terms
from infinidev.tools.knowledge.search_findings_input import SearchFindingsInput


class SearchFindingsTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "search_findings"
    description: str = (
        "Search findings by semantic similarity. Returns findings whose "
        "topic and content are similar to the query, ranked by similarity score. "
        "Use this to check if a finding already exists before recording "
        "a new one, or to find related findings across tasks."
    )
    args_schema: Type[BaseModel] = SearchFindingsInput

    def _run(
        self,
        query: str,
        threshold: float = 0.65,
        session_id: str | None = None,
        include_content: bool = False,
        limit: int = 20,
    ) -> str:
        project_id = self.project_id

        # Resolve session_id
        if session_id is None:
            effective_session_id = self.session_id
        elif session_id == "0":
            effective_session_id = None
        else:
            effective_session_id = session_id

        # Fetch candidate findings
        def _fetch(conn: sqlite3.Connection) -> list[dict]:
            conditions = ["1=1"]
            params: list = []

            if project_id:
                conditions.append("project_id = ?")
                params.append(project_id)
            if effective_session_id:
                conditions.append("session_id = ?")
                params.append(effective_session_id)

            where = " AND ".join(conditions)

            cols = "id, topic, session_id, confidence, finding_type, status, created_at, embedding"
            if include_content:
                cols += ", content, sources_json"

            rows = conn.execute(
                f"SELECT {cols} FROM findings WHERE {where} "
                f"ORDER BY created_at DESC LIMIT 500",
                params,
            ).fetchall()
            return [dict(r) for r in rows]

        try:
            candidates = execute_with_retry(_fetch)
        except Exception as e:
            return self._error(f"Failed to fetch findings: {e}")

        if not candidates:
            return self._success({"matches": [], "count": 0, "total_candidates": 0})

        # Compute semantic similarity
        try:
            from infinidev.tools.base.dedup import _get_embed_fn, _cosine_similarity
            from infinidev.tools.base.embeddings import embedding_from_blob

            embed_fn = _get_embed_fn()

            # Parse OR-terms: embed each sub-query, use max similarity
            or_terms = parse_query_or_terms(query)
            query_vecs = [np.asarray(v) for v in embed_fn(or_terms)]

            # Split candidates into those with pre-computed embeddings and those without
            with_emb = [(i, c) for i, c in enumerate(candidates) if c.get("embedding")]
            without_emb = [(i, c) for i, c in enumerate(candidates) if not c.get("embedding")]

            scores = [0.0] * len(candidates)

            # Use stored embeddings (fast path)
            for i, c in with_emb:
                emb_vec = embedding_from_blob(c["embedding"])
                scores[i] = max(_cosine_similarity(qv, emb_vec) for qv in query_vecs)

            # Fallback: embed topic + content for candidates without stored embeddings
            if without_emb:
                texts = [
                    f"{c['topic']} {c.get('content', '')[:500]}" if include_content or c.get("content")
                    else c["topic"]
                    for _, c in without_emb
                ]
                embeddings = embed_fn(texts)
                for j, (i, _) in enumerate(without_emb):
                    emb_vec = np.asarray(embeddings[j])
                    scores[i] = max(_cosine_similarity(qv, emb_vec) for qv in query_vecs)

            # Filter and rank
            scored = []
            for i, c in enumerate(candidates):
                if scores[i] >= threshold:
                    match = dict(c)
                    match.pop("embedding", None)  # don't return raw blob
                    match["similarity"] = round(scores[i], 4)
                    scored.append(match)

            scored.sort(key=lambda x: x["similarity"], reverse=True)
            scored = scored[:limit]

        except Exception as e:
            return self._error(f"Similarity search failed: {e}")

        return self._success({
            "matches": scored,
            "count": len(scored),
            "total_candidates": len(candidates),
            "threshold": threshold,
        })

