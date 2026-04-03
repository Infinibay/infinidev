"""Tool for writing research reports."""

import os
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, get_db_path


class WriteReportInput(BaseModel):
    title: str = Field(..., description="Report title")
    content: str = Field(..., description="Report content (markdown)")
    report_type: str = Field(
        default="research", description="Report type: 'research', 'analysis', 'summary'"
    )


