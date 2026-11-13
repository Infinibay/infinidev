"""Request another bounded window of read-only investigation."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ContinueInvestigationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    reason: str = Field(
        min_length=1, max_length=1000,
        description="What evidence is still needed to finish the user's read-only request.",
    )


class ContinueInvestigationTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "continue_investigation"
    description: str = (
        "Continue the same read-only investigation when more pages or sources are needed. "
        "Requires new successful reads since the previous extension and respects a total "
        "safety limit. Grants no write or command permissions. Use before the initial "
        "read budget ends; do not ask the user to repeat an already requested investigation."
    )
    args_schema: type[BaseModel] = ContinueInvestigationInput

    def _run(self, reason: str) -> str:
        # Only the chat loop owns and can extend its iteration budget.
        return json.dumps({"continued": False, "reason": "Requires the chat loop."})
