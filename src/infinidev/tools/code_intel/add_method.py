"""Tool: add a method or function to a file or class.

Inserts code at the end of a class (if class_name given) or end of file.
Auto-detects indentation.
"""

import os
import tempfile
from typing import Type
from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.code_intel.add_method_input import AddMethodInput
from infinidev.tools.code_intel.add_symbol_tool import AddSymbolTool
