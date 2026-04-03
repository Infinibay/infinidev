"""Tool for surgical file edits using search-and-replace."""

import difflib
import hashlib
import json
import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file._helpers import guard_file_access, atomic_write, record_artifact_change


from infinidev.tools.file.edit_file_input import EditFileInput
from infinidev.tools.file.edit_file_tool import EditFileTool
