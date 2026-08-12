import pytest

from dag_builder.errors import DependencyCycle
from dag_builder.graph import BuildGraph
from dag_builder.models import Node


def test_affected_nodes_are_transitive() -> None:
    graph = BuildGraph([
        Node("core"),
        Node("api", ("core",)),
        Node("app", ("api",)),
        Node("docs"),
    ])

    assert graph.affected_by({"core"}) == {"core", "api", "app"}


def test_topological_order_places_dependencies_first() -> None:
    graph = BuildGraph([Node("z-core"), Node("a-app", ("z-core",))])

    assert graph.topological({"a-app", "z-core"}) == ["z-core", "a-app"]


def test_cycle_is_rejected() -> None:
    graph = BuildGraph([Node("a", ("b",)), Node("b", ("a",))])

    with pytest.raises(DependencyCycle):
        graph.topological({"a", "b"})
