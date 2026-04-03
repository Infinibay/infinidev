"""Data models for the code intelligence system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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

