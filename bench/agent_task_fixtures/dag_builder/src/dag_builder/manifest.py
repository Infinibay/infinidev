"""Manifest helper."""

from dag_builder.models import Node


def names(nodes: list[Node]) -> tuple[str, ...]:
    return tuple(node.name for node in nodes)
