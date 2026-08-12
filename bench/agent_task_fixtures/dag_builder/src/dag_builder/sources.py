"""Source fingerprint contract."""

from typing import Protocol


class SourceIndex(Protocol):
    def fingerprints(self) -> dict[str, str]: ...
