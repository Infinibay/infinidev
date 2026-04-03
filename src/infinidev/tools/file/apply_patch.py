"""Tool for applying unified diff patches to files."""

import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


from infinidev.tools.file.apply_patch_input import ApplyPatchInput
from infinidev.tools.file.apply_patch_tool import (  # noqa: F401
    ApplyPatchTool,
    _extract_files_from_patch,
    _find_patch_binary,
    _parse_unified_diff,
)
