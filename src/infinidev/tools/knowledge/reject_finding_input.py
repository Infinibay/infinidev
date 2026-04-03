"""Tool for rejecting/superseding research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class RejectFindingInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to reject")
    reason: str = Field(..., description="Reason for rejection")


