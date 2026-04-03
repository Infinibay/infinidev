"""Data models for the code intelligence system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from infinidev.code_intel.symbol_kind import SymbolKind


@dataclass
class Symbol:
    """A named entity defined in source code."""

    name: str
    qualified_name: str = ""
    kind: SymbolKind = SymbolKind.function
    file_path: str = ""
    line_start: int = 0
    line_end: int | None = None
    column_start: int = 0
    signature: str = ""
    type_annotation: str = ""
    docstring: str = ""
    parent_symbol: str = ""
    visibility: str = "public"
    is_async: bool = False
    is_static: bool = False
    is_abstract: bool = False
    language: str = ""


