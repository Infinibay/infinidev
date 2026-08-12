"""Dependency traversal and ordering."""

from __future__ import annotations

from dag_builder.errors import DependencyCycle
from dag_builder.models import Node


class BuildGraph:
    def __init__(self, nodes: list[Node]) -> None:
        self.nodes = {node.name: node for node in nodes}

    def affected_by(self, changed: set[str]) -> set[str]:
        """Return changed nodes and their direct dependants."""
        affected = set(changed)
        affected.update(
            node.name
            for node in self.nodes.values()
            if set(node.dependencies) & changed
        )
        return affected

    def topological(self, selected: set[str]) -> list[str]:
        result: list[str] = []
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(name: str) -> None:
            if name in visited or name not in selected:
                return
            if name in visiting:
                raise DependencyCycle(name)
            visiting.add(name)
            for dependency in self.nodes[name].dependencies:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)
            result.append(name)

        for name in sorted(selected):
            visit(name)
        return result
