"""Tool for updating existing research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry




from infinidev.tools.knowledge.update_finding_input import UpdateFindingInput
from infinidev.tools.knowledge.update_finding_tool import UpdateFindingTool
