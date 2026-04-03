"""Tool for getting a compact summary of findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class SummarizeFindingsInput(BaseModel):
    session_id: str | None = Field(
        default=None,
        description=(
            "Session ID to summarize findings for. "
            "Omit or null to use the current session. "
            "Pass '0' to summarize all findings in the project."
        ),
    )


