"""Tool for deleting research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


from infinidev.tools.knowledge.delete_finding_input import DeleteFindingInput
from infinidev.tools.knowledge.delete_finding_tool import DeleteFindingTool
