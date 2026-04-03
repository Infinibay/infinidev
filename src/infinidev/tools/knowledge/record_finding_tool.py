"""Tool for recording research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry

FINDING_TYPES = ("observation", "hypothesis", "experiment", "proof", "conclusion", "project_context")
from infinidev.tools.knowledge.record_finding_input import RecordFindingInput


FINDING_TYPES = ("observation", "hypothesis", "experiment", "proof", "conclusion", "project_context")


class RecordFindingTool(InfinibayBaseTool):
    name: str = "record_finding"
    description: str = (
        "Record a research finding with confidence level and sources. "
        "Findings are created as 'provisional' and remain so until the "
        "Research Reviewer validates or rejects them."
    )
    args_schema: Type[BaseModel] = RecordFindingInput

    def _run(
        self,
        title: str,
        content: str,
        confidence: float = 0.5,
        tags: list[str] | None = None,
        finding_type: str = "observation",
        sources: list[str] | None = None,
        artifact_id: int | None = None,
    ) -> str:
        if tags is None:
            tags = []
        if sources is None:
            sources = []

        if finding_type not in FINDING_TYPES:
            return self._error(
                f"Invalid finding_type '{finding_type}'. "
                f"Must be one of: {', '.join(FINDING_TYPES)}"
            )

        agent_id = self._validate_agent_context()
        project_id = self.project_id
        session_id = self.session_id
        agent_run_id = self.agent_run_id

        # --- Semantic dedup check (same session + same finding_type) ---
        def _fetch_existing(conn: sqlite3.Connection) -> list[dict]:
            rows = conn.execute(
                "SELECT id, topic AS title, content FROM findings"
                " WHERE session_id = ? AND finding_type = ?",
                (session_id, finding_type),
            ).fetchall()
            return [{"id": r["id"], "title": r["title"], "content": r["content"]} for r in rows]

        try:
            existing = execute_with_retry(_fetch_existing)
            if existing:
                from infinidev.tools.base.dedup import find_semantic_duplicate

                match = find_semantic_duplicate(title, existing, threshold=0.85)
                if match:
                    existing_content = match.get("content", "")
                    preview = existing_content[:500]
                    if len(existing_content) > 500:
                        preview += "..."
                    return self._error(
                        f"Duplicate finding: '{match['title']}' (ID: {match['id']}) "
                        f"is {match['similarity']:.0%} similar to '{title}'.\n\n"
                        f"Existing finding content:\n{preview}\n\n"
                        f"If your new finding covers substantially different "
                        f"information, rephrase the title to clearly distinguish "
                        f"it and try again."
                    )
        except Exception:
            pass  # dedup is best-effort; don't block recording

        def _record(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                """INSERT INTO findings
                   (project_id, session_id, agent_run_id, topic, content,
                    sources_json, confidence, agent_id, status, finding_type,
                    artifact_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'provisional', ?, ?)""",
                (
                    project_id, session_id, agent_run_id, title, content,
                    json.dumps(sources), confidence, agent_id, finding_type,
                    artifact_id,
                ),
            )
            finding_id = cursor.lastrowid

            # Pre-compute embedding for semantic search
            try:
                from infinidev.tools.base.embeddings import store_finding_embedding
                store_finding_embedding(conn, finding_id, f"{title} {content[:500]}")
            except Exception:
                pass  # embedding is optional, don't block recording

            conn.commit()
            return finding_id

        try:
            finding_id = execute_with_retry(_record)
        except Exception as e:
            return self._error(f"Failed to record finding: {e}")

        self._log_tool_usage(
            f"Recorded finding #{finding_id}: {title[:60]} "
            f"(confidence={confidence})"
        )
        return self._success({
            "status": "Finding recorded successfully",
            "finding_id": finding_id,
            "title": title,
            "finding_type": finding_type,
            "confidence": confidence,
            "finding_status": "provisional",
        })

