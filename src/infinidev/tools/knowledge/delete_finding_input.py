"""Tool for deleting research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class DeleteFindingInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to delete")


