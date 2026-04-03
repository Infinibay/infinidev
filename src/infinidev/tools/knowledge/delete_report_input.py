"""Tool for deleting reports/artifacts."""

import os
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, get_db_path


class DeleteReportInput(BaseModel):
    artifact_id: int = Field(..., description="ID of the report artifact to delete")
    delete_file: bool = Field(
        default=True,
        description="Also delete the report file from disk",
    )


