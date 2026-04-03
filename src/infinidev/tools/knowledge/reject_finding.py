"""Tool for rejecting/superseding research findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


from infinidev.tools.knowledge.reject_finding_input import RejectFindingInput
from infinidev.tools.knowledge.reject_finding_tool import RejectFindingTool
