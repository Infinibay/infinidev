"""Tool for validating research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class ValidateFindingInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to validate")
    validation_method: str | None = Field(
        default=None, description="Method used to validate the finding"
    )
    reproducibility_score: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="Reproducibility score (0.0 to 1.0)",
    )


