"""Tool for writing research reports."""

import os
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry, get_db_path


from infinidev.tools.knowledge.write_report_input import WriteReportInput
from infinidev.tools.knowledge.write_report_tool import WriteReportTool
