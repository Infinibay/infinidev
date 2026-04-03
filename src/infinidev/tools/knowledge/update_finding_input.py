"""Tool for updating existing research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry

FINDING_TYPES = ("observation", "hypothesis", "experiment", "proof", "conclusion", "project_context")


class UpdateFindingInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to update")
    title: str | None = Field(default=None, description="New title (topic)")
    content: str | None = Field(default=None, description="New content")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="New confidence level"
    )
    finding_type: str | None = Field(
        default=None,
        description=f"New finding type: {', '.join(FINDING_TYPES)}",
    )
    tags: list[str] | None = Field(default=None, description="Replace tags")
    sources: list[str] | None = Field(default=None, description="Replace sources")


