"""Tool for getting a compact summary of findings."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


from infinidev.tools.knowledge.summarize_findings_input import SummarizeFindingsInput
from infinidev.tools.knowledge.summarize_findings_tool import SummarizeFindingsTool
