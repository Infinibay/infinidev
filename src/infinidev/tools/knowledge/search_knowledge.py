"""Tool for unified cross-source knowledge search using FTS5."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, sanitize_fts5_query


from infinidev.tools.knowledge.search_knowledge_input import SearchKnowledgeInput
from infinidev.tools.knowledge.search_knowledge_tool import SearchKnowledgeTool
