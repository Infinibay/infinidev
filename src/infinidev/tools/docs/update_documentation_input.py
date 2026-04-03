"""Tool to fetch and generate library documentation from the web."""

import logging
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class UpdateDocumentationInput(BaseModel):
    library_name: str = Field(..., description="Name of the library to document")
    language: str = Field(default="unknown", description="Programming language")
    version: str = Field(default="latest", description="Library version")


