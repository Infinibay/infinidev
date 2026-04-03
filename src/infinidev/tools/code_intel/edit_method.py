"""Tool: edit (replace) a method or function by symbol name.

Uses tree-sitter index to find the exact location of the symbol,
then replaces it with new code. No old_string matching needed.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.code_intel.edit_method_input import EditMethodInput
from infinidev.tools.code_intel.edit_symbol_tool import EditSymbolTool
