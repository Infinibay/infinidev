"""Build action contract."""

from typing import Protocol


class BuildActions(Protocol):
    def build(self, name: str) -> None: ...
