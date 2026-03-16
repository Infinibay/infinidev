"""Tool for rejecting/superseding research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class RejectFindingInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to reject")
    reason: str = Field(..., description="Reason for rejection")


class RejectFindingTool(InfinibayBaseTool):
    name: str = "reject_finding"
    description: str = (
        "Reject a research finding, changing its status to 'superseded'. "
        "Requires a reason for the rejection."
    )
    args_schema: Type[BaseModel] = RejectFindingInput

    def _run(self, finding_id: int, reason: str) -> str:
        agent_id = self._validate_agent_context()

        def _reject(conn: sqlite3.Connection) -> dict:
            row = conn.execute(
                "SELECT id, topic, status FROM findings WHERE id = ?",
                (finding_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Finding {finding_id} not found")
            if row["status"] == "superseded":
                raise ValueError(f"Finding {finding_id} is already superseded")

            conn.execute(
                """UPDATE findings
                   SET status = 'superseded', validation_method = ?
                   WHERE id = ?""",
                (f"Rejected by {agent_id}: {reason}", finding_id),
            )
            conn.commit()
            return {"topic": row["topic"]}

        try:
            result = execute_with_retry(_reject)
        except ValueError as e:
            return self._error(str(e))

        self._log_tool_usage(
            f"Rejected finding #{finding_id}: {result['topic'][:60]}"
        )
        return self._success({
            "finding_id": finding_id,
            "status": "superseded",
            "rejected_by": agent_id,
            "reason": reason,
        })
