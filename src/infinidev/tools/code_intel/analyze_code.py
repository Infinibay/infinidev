"""Tool: run heuristic code analysis on indexed data."""

import json
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.code_intel.analyze_code_input import AnalyzeCodeInput
from infinidev.tools.code_intel.analyze_code_tool import AnalyzeCodeTool
