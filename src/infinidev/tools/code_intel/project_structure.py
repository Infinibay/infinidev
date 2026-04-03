"""Tool: show project structure with semantic descriptions from the code index."""

import os
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.code_intel.project_structure_input import ProjectStructureInput
from infinidev.tools.code_intel.project_structure_tool import ProjectStructureTool
