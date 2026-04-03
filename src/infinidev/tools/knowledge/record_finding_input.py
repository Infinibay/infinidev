"""Tool for recording research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry

FINDING_TYPES = ("observation", "hypothesis", "experiment", "proof", "conclusion", "project_context")


class RecordFindingInput(BaseModel):
    title: str = Field(..., description="Finding title/topic")
    content: str = Field(..., description="Detailed finding content")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence level (0.0 to 1.0)"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    finding_type: str = Field(
        default="observation",
        description=f"Finding type: {', '.join(FINDING_TYPES)}",
    )
    sources: list[str] = Field(
        default_factory=list, description="Source URLs or references"
    )
    artifact_id: int | None = Field(default=None, description="Optional ID of a related artifact")


