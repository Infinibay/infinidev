"""Tool for replacing a range of lines in a file."""

import hashlib
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


from infinidev.tools.file.replace_lines_input import ReplaceLinesInput
from infinidev.tools.file.replace_lines_tool import ReplaceLinesTool
