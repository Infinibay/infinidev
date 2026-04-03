"""Tool for searching code examples and documentation on the web."""

import json
import logging
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.web.backends import search_ddg

logger = logging.getLogger(__name__)

# In-memory cache for code search results
_cache: dict[str, str] = {}



from infinidev.tools.web.code_search_web_input import CodeSearchWebInput
from infinidev.tools.web.code_search_web_tool import CodeSearchWebTool
