"""Tool: rename a symbol and update all references across the project."""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.code_intel.rename_symbol_input import RenameSymbolInput
from infinidev.tools.code_intel.rename_symbol_tool import RenameSymbolTool
