"""Tool for recording research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.knowledge.finding_types import FINDING_TYPES
from infinidev.tools.knowledge.record_finding_input import RecordFindingInput


class RecordFindingTool(InfinibayBaseTool):
    name: str = "record_finding"
    description: str = (
        "Record a finding for future sessions. Three tiers: observational "
        "findings (observation/hypothesis/proof/...) are loaded into the "
        "system prompt next session via project_knowledge. Anchored "
        "memories (lesson/rule/landmine) are attached to a specific "
        "anchor (file, symbol, tool, or error pattern) and automatically "
        "injected next to the tool result when the agent touches that "
        "anchor. For anchored memories you MUST provide at least one "
        "anchor_* parameter — otherwise the memory will never fire. "
        "Findings are created as 'provisional' and promoted by review."
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
        anchor_file: str | None = None,
        anchor_symbol: str | None = None,
        anchor_tool: str | None = None,
        anchor_error: str | None = None,
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

        # Anchored types MUST have at least one anchor or the memory
        # is dead on arrival — it'll sit in the DB forever and never
        # fire. Reject up front with a clear message.
        _ANCHORED = {"lesson", "rule", "landmine"}
        if finding_type in _ANCHORED and not any(
            (anchor_file, anchor_symbol, anchor_tool, anchor_error)
        ):
            return self._error(
                f"finding_type='{finding_type}' requires at least one "
                f"anchor_* parameter (anchor_file, anchor_symbol, "
                f"anchor_tool, or anchor_error). Without an anchor the "
                f"memory will never auto-inject and you have effectively "
                f"lost it. If you want an un-anchored note, use "
                f"finding_type='observation' instead."
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
                    artifact_id,
                    anchor_file, anchor_symbol, anchor_tool, anchor_error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'provisional', ?, ?,
                           ?, ?, ?, ?)""",
                (
                    project_id, session_id, agent_run_id, title, content,
                    json.dumps(sources), confidence, agent_id, finding_type,
                    artifact_id,
                    anchor_file, anchor_symbol, anchor_tool, anchor_error,
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

        # Surface the anchor in the log line so a reader can tell at a
        # glance whether this finding will auto-inject.
        anchor_parts = []
        if anchor_file:
            anchor_parts.append(f"file={anchor_file}")
        if anchor_symbol:
            anchor_parts.append(f"symbol={anchor_symbol}")
        if anchor_tool:
            anchor_parts.append(f"tool={anchor_tool}")
        if anchor_error:
            anchor_parts.append(f"error={anchor_error[:30]}")
        anchor_tag = f" [anchor: {', '.join(anchor_parts)}]" if anchor_parts else ""

        self._log_tool_usage(
            f"Recorded {finding_type} #{finding_id}: {title[:60]} "
            f"(confidence={confidence}){anchor_tag}"
        )
        return self._success({
            "status": "Finding recorded successfully",
            "finding_id": finding_id,
            "title": title,
            "finding_type": finding_type,
            "confidence": confidence,
            "finding_status": "provisional",
            "anchored": bool(anchor_parts),
        })
