"""Tool: remove a method or function by symbol name.

Uses tree-sitter index to find and delete the symbol's source lines.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.code_intel.remove_method_input import RemoveMethodInput
from infinidev.tools.code_intel.remove_symbol_tool import RemoveSymbolTool
