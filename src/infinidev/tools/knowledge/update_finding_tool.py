"""Tool for updating existing research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.knowledge.finding_types import ANCHORED_TYPES, FINDING_TYPES
from infinidev.tools.knowledge.update_finding_input import UpdateFindingInput


class UpdateFindingTool(InfinibayBaseTool):
    name: str = "update_finding"
    description: str = (
        "Update an existing finding's content, confidence, type, tags, "
        "sources, or anchor fields. Only the fields you provide will be "
        "changed. Useful for promoting an observation into a lesson by "
        "adding an anchor_file, or for moving an anchor when a symbol "
        "has been renamed. Pass empty string '' for an anchor_* field "
        "to clear it; None (the default) leaves it unchanged."
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
        anchor_file: str | None = None,
        anchor_symbol: str | None = None,
        anchor_tool: str | None = None,
        anchor_error: str | None = None,
    ) -> str:
        if finding_type is not None and finding_type not in FINDING_TYPES:
            return self._error(
                f"Invalid finding_type '{finding_type}'. "
                f"Must be one of: {', '.join(FINDING_TYPES)}"
            )

        # Anchor validation must be done inside _update where we can see
        # both the existing anchors AND the new values being set.
        def _validate_anchors(
            existing_anchors: dict,
            new_anchors: dict,
            new_type: str | None,
        ) -> str | None:
            """Return None if valid, or an error string."""
            if new_type not in ANCHORED_TYPES:
                return None

            # Merge existing + new: None means "keep existing", "" means "clear"
            _ANCHOR_KEYS = ("anchor_file", "anchor_symbol", "anchor_tool", "anchor_error")
            final_anchors = {}
            for k in _ANCHOR_KEYS:
                new_val = new_anchors.get(k)
                if new_val is None:
                    final_anchors[k] = existing_anchors.get(k, "")
                else:
                    final_anchors[k] = new_val  # "" clears, non-empty sets

            if not any(final_anchors.values()):
                return (
                    f"finding_type='{new_type}' requires at least one anchor_* "
                    f"parameter (anchor_file, anchor_symbol, anchor_tool, or "
                    f"anchor_error). Either provide an anchor now or use a "
                    f"non-anchored finding_type like 'observation'."
                )
            return None

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

        # Anchor updates. None = unchanged (not in SET clause at all).
        # Empty string = clear (SET anchor_X = NULL).
        def _anchor_update(col: str, value: str | None) -> None:
            if value is None:
                return
            if value == "":
                updates.append(f"{col} = NULL")
            else:
                updates.append(f"{col} = ?")
                params.append(value)

        _anchor_update("anchor_file", anchor_file)
        _anchor_update("anchor_symbol", anchor_symbol)
        _anchor_update("anchor_tool", anchor_tool)
        _anchor_update("anchor_error", anchor_error)

        if not updates:
            return self._error("No fields to update. Provide at least one field.")

        updates.append("updated_at = CURRENT_TIMESTAMP")
        params.append(finding_id)

        # --- Pre-validate anchor requirement before touching the DB ---
        # We need to know whether the existing record already has anchors
        # and whether this update is adding/changing them.
        def _fetch_existing_anchors(_conn: sqlite3.Connection) -> dict | None:
            row = _conn.execute(
                "SELECT anchor_file, anchor_symbol, anchor_tool, anchor_error "
                "FROM findings WHERE id = ?",
                (finding_id,),
            ).fetchone()
            return dict(row) if row else None

        def _update(conn: sqlite3.Connection) -> dict:
            existing = _fetch_existing_anchors(conn)
            if existing is None:
                raise ValueError(f"Finding {finding_id} not found")

            new_anchors = {
                "anchor_file": anchor_file,
                "anchor_symbol": anchor_symbol,
                "anchor_tool": anchor_tool,
                "anchor_error": anchor_error,
            }
            err = _validate_anchors(existing, new_anchors, finding_type)
            if err:
                raise ValueError(err)

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
