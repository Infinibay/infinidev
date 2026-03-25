"""Data models for the code intelligence system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SymbolKind(str, Enum):
    function = "function"
    method = "method"
    class_ = "class"
    variable = "variable"
    constant = "constant"
    interface = "interface"
    enum = "enum"
    module = "module"
    property_ = "property"
    type_alias = "type_alias"
    decorator = "decorator"
    parameter = "parameter"


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


@dataclass
class Reference:
    """A usage of a symbol name in source code."""

    name: str
    file_path: str = ""
    line: int = 0
    column: int = 0
    context: str = ""
    ref_kind: str = "usage"  # "usage", "call", "import", "type_ref", "assignment"
    resolved_file: str = ""
    resolved_line: int | None = None
    language: str = ""


@dataclass
class Import:
    """An import statement in source code."""

    source: str  # "auth.service", "./utils", "std::io"
    name: str  # "verify_token", "HashMap"
    alias: str = ""
    file_path: str = ""
    line: int = 0
    is_wildcard: bool = False
    resolved_file: str = ""
    language: str = ""
