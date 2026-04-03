"""Tool for reading/searching research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, sanitize_fts5_query


from infinidev.tools.knowledge.read_findings_input import ReadFindingsInput
from infinidev.tools.knowledge.read_findings_tool import ReadFindingsTool
