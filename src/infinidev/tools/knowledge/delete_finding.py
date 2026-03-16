"""Tool for deleting research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class DeleteFindingInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to delete")


class DeleteFindingTool(InfinibayBaseTool):
    name: str = "delete_finding"
    description: str = (
        "Permanently delete a research finding by its ID. "
        "Use reject_finding instead if you want to keep a record of why it was dismissed."
    )
    args_schema: Type[BaseModel] = DeleteFindingInput

    def _run(self, finding_id: int) -> str:
        def _delete(conn: sqlite3.Connection) -> dict:
            row = conn.execute(
                "SELECT id, topic FROM findings WHERE id = ?",
                (finding_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Finding {finding_id} not found")

            conn.execute("DELETE FROM findings WHERE id = ?", (finding_id,))
            conn.commit()
            return {"topic": row["topic"]}

        try:
            result = execute_with_retry(_delete)
        except ValueError as e:
            return self._error(str(e))
        except Exception as e:
            return self._error(f"Failed to delete finding: {e}")

        self._log_tool_usage(f"Deleted finding #{finding_id}: {result['topic'][:60]}")
        return self._success({
            "finding_id": finding_id,
            "deleted": True,
            "topic": result["topic"],
        })
