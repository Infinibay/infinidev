"""Minimal build logger contract."""

from typing import Protocol


class BuildLogger(Protocol):
    def emit(self, node: str, status: str) -> None: ...
