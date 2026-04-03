"""Tool for searching findings by semantic similarity."""

import sqlite3
from typing import Type

import numpy as np
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, parse_query_or_terms


from infinidev.tools.knowledge.search_findings_input import SearchFindingsInput
from infinidev.tools.knowledge.search_findings_tool import SearchFindingsTool
