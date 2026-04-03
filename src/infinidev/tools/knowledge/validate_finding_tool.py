"""Tool for validating research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.knowledge.validate_finding_input import ValidateFindingInput


class ValidateFindingTool(InfinibayBaseTool):
    name: str = "validate_finding"
    description: str = (
        "Validate a research finding, changing its status from "
        "'provisional' to 'active'. Optionally add validation method and score."
    )
    args_schema: Type[BaseModel] = ValidateFindingInput

    def _run(
        self,
        finding_id: int,
        validation_method: str | None = None,
        reproducibility_score: float | None = None,
    ) -> str:
        agent_id = self._validate_agent_context()

        def _validate(conn: sqlite3.Connection) -> dict:
            row = conn.execute(
                "SELECT id, topic, status FROM findings WHERE id = ?",
                (finding_id,),
            ).fetchone()

            if not row:
                raise ValueError(f"Finding {finding_id} not found")
            if row["status"] == "active":
                raise ValueError(f"Finding {finding_id} is already validated")
            if row["status"] == "superseded":
                raise ValueError(f"Finding {finding_id} has been superseded")

            updates = ["status = 'active'"]
            params: list = []

            if validation_method:
                updates.append("validation_method = ?")
                params.append(validation_method)
            if reproducibility_score is not None:
                updates.append("reproducibility_score = ?")
                params.append(reproducibility_score)

            params.append(finding_id)
            conn.execute(
                f"UPDATE findings SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            conn.commit()
            return {"topic": row["topic"]}

        try:
            result = execute_with_retry(_validate)
        except ValueError as e:
            return self._error(str(e))

        self._log_tool_usage(
            f"Validated finding #{finding_id}: {result['topic'][:60]}"
        )
        return self._success({
            "finding_id": finding_id,
            "status": "active",
            "validated_by": agent_id,
            "validation_method": validation_method,
            "reproducibility_score": reproducibility_score,
        })

