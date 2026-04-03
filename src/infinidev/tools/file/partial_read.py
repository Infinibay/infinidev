"""Tool for reading a specific range of lines from a file."""

import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file.read_file import ReadFileTool


from infinidev.tools.file.partial_read_input import PartialReadInput
from infinidev.tools.file.partial_read_tool import PartialReadTool
