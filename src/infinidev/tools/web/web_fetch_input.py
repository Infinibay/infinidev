"""Web fetch tool for extracting readable content from URLs."""

import sqlite3
from typing import Literal, Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.db.service import execute_with_retry
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.web.rate_limiter import web_rate_limiter
from infinidev.tools.web.robots_checker import robots_checker


class WebFetchInput(BaseModel):
    url: str = Field(..., description="URL to fetch content from")
    format: Literal["markdown", "text"] = Field(
        default="markdown", description="Output format: 'markdown' or 'text'"
    )
    bypass_cache: bool = Field(
        default=False,
        description="Set to true to skip the cache and fetch fresh content",
    )


