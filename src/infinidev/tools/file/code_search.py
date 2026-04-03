"""Tool for searching code patterns across a codebase."""

import json
import os
import subprocess
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.file.code_search_input import CodeSearchInput
from infinidev.tools.file.code_search_tool import CodeSearchTool
