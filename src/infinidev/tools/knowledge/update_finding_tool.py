"""Tool for updating existing research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry

FINDING_TYPES = ("observation", "hypothesis", "experiment", "proof", "conclusion", "project_context")
from infinidev.tools.knowledge.update_finding_input import UpdateFindingInput


FINDING_TYPES = ("observation", "hypothesis", "experiment", "proof", "conclusion", "project_context")


class UpdateFindingTool(InfinibayBaseTool):
    name: str = "update_finding"
    description: str = (
        "Update an existing finding's content, confidence, type, tags, or sources. "
        "Only the fields you provide will be changed."
    )
    args_schema: Type[BaseModel] = UpdateFindingInput

    def _run(
        self,
        finding_id: int,
        title: str | None = None,
        content: str | None = None,
        confidence: float | None = None,
        finding_type: str | None = None,
        tags: list[str] | None = None,
        sources: list[str] | None = None,
    ) -> str:
        if finding_type is not None and finding_type not in FINDING_TYPES:
            return self._error(
                f"Invalid finding_type '{finding_type}'. "
                f"Must be one of: {', '.join(FINDING_TYPES)}"
            )

        updates: list[str] = []
        params: list = []

        if title is not None:
            updates.append("topic = ?")
            params.append(title)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if confidence is not None:
            updates.append("confidence = ?")
            params.append(confidence)
        if finding_type is not None:
            updates.append("finding_type = ?")
            params.append(finding_type)
        if tags is not None:
            updates.append("tags_json = ?")
            params.append(json.dumps(tags))
        if sources is not None:
            updates.append("sources_json = ?")
            params.append(json.dumps(sources))

        if not updates:
            return self._error("No fields to update. Provide at least one field.")

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(finding_id)

        def _update(conn: sqlite3.Connection) -> dict:
            row = conn.execute(
                "SELECT id, topic, status FROM findings WHERE id = ?",
                (finding_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Finding {finding_id} not found")

            conn.execute(
                f"UPDATE findings SET {', '.join(updates)} WHERE id = ?",
                params,
            )

            # Re-compute embedding if title or content changed
            if title is not None or content is not None:
                try:
                    new_row = conn.execute(
                        "SELECT topic, content FROM findings WHERE id = ?",
                        (finding_id,),
                    ).fetchone()
                    from infinidev.tools.base.embeddings import store_finding_embedding
                    store_finding_embedding(
                        conn, finding_id,
                        f"{new_row['topic']} {(new_row['content'] or '')[:500]}",
                    )
                except Exception:
                    pass

            conn.commit()
            return {"topic": row["topic"], "status": row["status"]}

        try:
            result = execute_with_retry(_update)
        except ValueError as e:
            return self._error(str(e))
        except Exception as e:
            return self._error(f"Failed to update finding: {e}")

        self._log_tool_usage(f"Updated finding #{finding_id}: {result['topic'][:60]}")
        return self._success({
            "finding_id": finding_id,
            "updated_fields": [u.split(" =")[0] for u in updates if u != "updated_at = CURRENT_TIMESTAMP"],
            "status": result["status"],
        })

