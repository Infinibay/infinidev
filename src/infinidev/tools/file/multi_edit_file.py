"""Tool for applying multiple search-and-replace edits to a single file atomically."""

import hashlib
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


from infinidev.tools.file.edit_operation import EditOperation
from infinidev.tools.file.multi_edit_file_input import MultiEditFileInput
from infinidev.tools.file.multi_edit_file_tool import MultiEditFileTool
