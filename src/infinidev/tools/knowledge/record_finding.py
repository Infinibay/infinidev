"""Tool for recording research findings."""

import json
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry




from infinidev.tools.knowledge.record_finding_input import RecordFindingInput
from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
