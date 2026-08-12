"""Build event model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BuildEvent:
    node: str
    status: str
