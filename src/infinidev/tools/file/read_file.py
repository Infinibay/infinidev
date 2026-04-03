"""Tool for reading file contents with optional line-range selection."""

import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.file.read_file_input import ReadFileInput
from infinidev.tools.file.read_file_tool import ReadFileTool

# Bytes considered "text-safe": printable ASCII, common whitespace, and high
# bytes (≥0x80) which appear in UTF-8 / Latin-1 text.  Control bytes 0x00-0x08,
# 0x0E-0x1F (excluding \t \n \r) and 0x7F are strong binary indicators.