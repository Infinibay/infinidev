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


class ApplyPatchInput(BaseModel):
    patch: str = Field(
        ...,
        description=(
            "A unified diff string (like output of `git diff`). "
            "Can contain changes to one or multiple files. "
            "Must include diff headers (--- a/file, +++ b/file) and hunks (@@ ... @@)."
        ),
    )
    strip: int = Field(
        default=1,
        description="Number of leading path components to strip (like `patch -pN`). Default 1.",
    )


