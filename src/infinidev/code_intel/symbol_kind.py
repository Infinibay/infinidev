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


