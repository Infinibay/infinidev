"""Tool for reading research reports."""

import os
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class ReadReportInput(BaseModel):
    report_id: int | None = Field(
        default=None, description="Artifact ID of the report"
    )
    session_id: str | None = Field(
        default=None, description="Session ID to find associated report"
    )
    file_path: str | None = Field(
        default=None, description="Full or partial file path to match"
    )


