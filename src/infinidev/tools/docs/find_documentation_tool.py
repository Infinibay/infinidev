"""Tool to find and read locally cached library documentation."""

import json
import logging
import sqlite3
from typing import Optional, Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob

logger = logging.getLogger(__name__)
from infinidev.tools.docs.find_documentation_input import FindDocumentationInput


class FindDocumentationTool(InfinibayBaseTool):
    name: str = "find_documentation"
    description: str = (
        "Find and read locally cached library documentation. "
        "Can list sections, read a specific section, or search within docs. "
        "If no docs exist, suggests calling update_documentation to fetch them."
    )
    args_schema: Type[BaseModel] = FindDocumentationInput

    def _run(
        self,
        library_name: str,
        language: str = "unknown",
        version: str = "latest",
        section: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:
        if section:
            return self._read_section(library_name, language, version, section)
        elif query:
            return self._search(library_name, language, version, query)
        else:
            return self._list_sections(library_name, language, version)

    def _resolve_version(self, conn: sqlite3.Connection, library_name: str, language: str, version: str) -> str | None:
        """If version='latest' and no exact match, find the most recent version."""
        if version != "latest":
            return version
        row = conn.execute(
            "SELECT version FROM library_docs WHERE library_name = ? AND language = ? ORDER BY updated_at DESC LIMIT 1",
            (library_name, language),
        ).fetchone()
        return row["version"] if row else None

    def _list_sections(self, library_name: str, language: str, version: str) -> str:
        def _query(conn: sqlite3.Connection):
            resolved = self._resolve_version(conn, library_name, language, version)
            if not resolved:
                return None
            return conn.execute(
                """\
                SELECT section_title, section_order, content
                FROM library_docs
                WHERE library_name = ? AND language = ? AND version = ?
                ORDER BY section_order
                """,
                (library_name, language, resolved),
            ).fetchall()

        rows = execute_with_retry(_query)
        if not rows:
            return self._success({
                "found": False,
                "message": f"No documentation found for {library_name} ({language}). "
                           f"Call update_documentation to fetch it.",
            })

        sections = []
        for row in rows:
            preview = row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"]
            sections.append({
                "title": row["section_title"],
                "order": row["section_order"],
                "preview": preview,
            })
        return self._success({
            "library": library_name,
            "language": language,
            "sections": sections,
        })

    def _read_section(self, library_name: str, language: str, version: str, section: str) -> str:
        def _query(conn: sqlite3.Connection):
            resolved = self._resolve_version(conn, library_name, language, version)
            if not resolved:
                return None
            return conn.execute(
                """\
                SELECT section_title, content, source_urls
                FROM library_docs
                WHERE library_name = ? AND language = ? AND version = ?
                  AND section_title LIKE ?
                ORDER BY section_order
                LIMIT 1
                """,
                (library_name, language, resolved, f"%{section}%"),
            ).fetchone()

        row = execute_with_retry(_query)
        if not row:
            return self._error(f"Section '{section}' not found for {library_name}")

        return self._success({
            "section": row["section_title"],
            "content": row["content"],
            "sources": json.loads(row["source_urls"] or "[]"),
        })

    def _search(self, library_name: str, language: str, version: str, query: str) -> str:
        # Try FTS5 first
        def _fts_query(conn: sqlite3.Connection):
            resolved = self._resolve_version(conn, library_name, language, version)
            if not resolved:
                return None, resolved
            rows = conn.execute(
                """\
                SELECT ld.section_title, snippet(library_docs_fts, 1, '>>>', '<<<', '...', 64) as snippet,
                       ld.content
                FROM library_docs_fts fts
                JOIN library_docs ld ON ld.id = fts.rowid
                WHERE library_docs_fts MATCH ?
                  AND ld.library_name = ? AND ld.language = ? AND ld.version = ?
                ORDER BY rank
                LIMIT 5
                """,
                (query, library_name, language, resolved),
            ).fetchall()
            return rows, resolved

        try:
            fts_rows, resolved_version = execute_with_retry(_fts_query)
        except Exception:
            fts_rows, resolved_version = None, None

        if fts_rows:
            results = [{"section": r["section_title"], "snippet": r["snippet"]} for r in fts_rows]
            return self._success({"query": query, "results": results})

        # Fallback: cosine similarity with embeddings
        query_emb = compute_embedding(query)
        if query_emb is None:
            return self._error(f"No FTS results and embedding computation failed for query: {query}")

        def _emb_query(conn: sqlite3.Connection):
            resolved = self._resolve_version(conn, library_name, language, version)
            if not resolved:
                return None
            return conn.execute(
                """\
                SELECT id, section_title, content, embedding
                FROM library_docs
                WHERE library_name = ? AND language = ? AND version = ?
                  AND embedding IS NOT NULL
                """,
                (library_name, language, resolved),
            ).fetchall()

        rows = execute_with_retry(_emb_query)
        if not rows:
            return self._success({
                "found": False,
                "message": f"No documentation found for {library_name} ({language}). "
                           f"Call update_documentation to fetch it.",
            })

        query_vec = np.frombuffer(query_emb, dtype=np.float32)
        scored = []
        for row in rows:
            doc_vec = embedding_from_blob(row["embedding"])
            sim = float(np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec) + 1e-9))
            scored.append((sim, row["section_title"], row["content"][:300]))

        scored.sort(reverse=True)
        results = [
            {"section": title, "similarity": round(sim, 3), "snippet": snippet}
            for sim, title, snippet in scored[:5]
        ]
        return self._success({"query": query, "results": results})

