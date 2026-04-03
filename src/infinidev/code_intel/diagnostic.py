"""Heuristic code analyzer — detects errors using indexed data.

All checks run SQL queries against ci_symbols, ci_references, and ci_imports.
No re-parsing needed — works on data already extracted by tree-sitter.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from infinidev.tools.base.db import execute_with_retry

logger = logging.getLogger(__name__)

# Python builtins that should not be flagged as undefined
_PYTHON_BUILTINS = frozenset({
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "range", "type", "isinstance", "issubclass", "hasattr", "getattr",
    "setattr", "delattr", "super", "property", "staticmethod", "classmethod",
    "object", "id", "hash", "repr", "abs", "round", "min", "max", "sum",
    "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all",
    "iter", "next", "input", "open", "vars", "dir", "callable", "format",
    "chr", "ord", "hex", "oct", "bin", "pow", "divmod", "complex", "bytes",
    "bytearray", "memoryview", "frozenset", "slice", "breakpoint",
    "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
    "IndexError", "AttributeError", "ImportError", "ModuleNotFoundError",
    "FileNotFoundError", "OSError", "IOError", "RuntimeError", "StopIteration",
    "NotImplementedError", "PermissionError", "ConnectionError", "TimeoutError",
    "AssertionError", "NameError", "ZeroDivisionError", "OverflowError",
    "UnicodeDecodeError", "UnicodeEncodeError", "SystemExit", "RecursionError",
    "True", "False", "None", "NotImplemented", "Ellipsis",
    "__name__", "__file__", "__doc__", "__all__", "__init__",
    # Common decorators/functions from typing
    "Optional", "Union", "List", "Dict", "Set", "Tuple", "Any",
    "Callable", "Iterator", "Generator", "Sequence", "Mapping",
    "TYPE_CHECKING", "Protocol", "TypeVar", "Generic", "ClassVar",
    "Final", "Literal", "overload", "cast",
    # dataclasses
    "dataclass", "field",
})

ALL_CHECKS = ["broken_imports", "undefined_symbols", "unused_imports", "unused_definitions"]


@dataclass
class Diagnostic:
    """A single diagnostic finding."""
    file_path: str
    line: int
    severity: str  # "error", "warning", "hint"
    check: str  # check name
    message: str
    fix_suggestion: str = ""


