"""Tool to delete locally cached library documentation."""

import logging
import sqlite3
from typing import Optional, Type

from pydantic import BaseModel, Field

from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class DeleteDocumentationInput(BaseModel):
    library_name: str = Field(..., description="Name of the library")
    language: str = Field(default="unknown", description="Programming language")
    version: str = Field(default="latest", description="Library version")
    section: Optional[str] = Field(
        default=None,
        description="Specific section title to delete. If omitted, deletes ALL sections for this library/version.",
    )


