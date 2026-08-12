"""Build graph models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    name: str
    dependencies: tuple[str, ...] = ()
