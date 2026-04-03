"""Tool to find and read locally cached library documentation."""

import json
import logging
import sqlite3
from typing import Optional, Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob

logger = logging.getLogger(__name__)


class FindDocumentationInput(BaseModel):
    library_name: str = Field(..., description="Name of the library to look up")
    language: str = Field(default="unknown", description="Programming language")
    version: str = Field(default="latest", description="Library version")
    section: Optional[str] = Field(default=None, description="Specific section title to read")
    query: Optional[str] = Field(default=None, description="Search query within the docs")


