"""Data models for the code intelligence system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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


