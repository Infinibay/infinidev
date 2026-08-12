import pytest

from dag_builder import BuildGraph, BuildState, IncrementalBuilder, Node


class Sources:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values

    def fingerprints(self) -> dict[str, str]:
        return dict(self.values)


class Actions:
    def __init__(self, fail: str | None = None) -> None:
        self.fail = fail
        self.calls: list[str] = []

    def build(self, name: str) -> None:
        self.calls.append(name)
        if name == self.fail:
            raise RuntimeError(name)


def graph() -> BuildGraph:
    return BuildGraph([
        Node("z-core"),
        Node("m-api", ("z-core",)),
        Node("a-app", ("m-api",)),
        Node("docs"),
    ])


def test_transitive_rebuild_uses_dependency_order() -> None:
    state = BuildState({"z-core": "old", "m-api": "same", "a-app": "same", "docs": "same"})
    sources = Sources({"z-core": "new", "m-api": "same", "a-app": "same", "docs": "same"})
    actions = Actions()

    built = IncrementalBuilder(graph(), sources, actions, state).build()

    assert built == ["z-core", "m-api", "a-app"]
    assert actions.calls == built
    assert state.fingerprints["z-core"] == "new"
    assert state.fingerprints["docs"] == "same"


def test_failed_node_and_dependants_remain_dirty_for_retry() -> None:
    initial = {"z-core": "old", "m-api": "old", "a-app": "old", "docs": "same"}
    current = {"z-core": "new", "m-api": "new", "a-app": "new", "docs": "same"}
    state = BuildState(dict(initial))
    actions = Actions(fail="m-api")

    with pytest.raises(RuntimeError):
        IncrementalBuilder(graph(), Sources(current), actions, state).build()

    assert actions.calls == ["z-core", "m-api"]
    assert state.fingerprints == {
        "z-core": "new",
        "m-api": "old",
        "a-app": "old",
        "docs": "same",
    }

    retry = Actions()
    built = IncrementalBuilder(graph(), Sources(current), retry, state).build()
    assert built == ["m-api", "a-app"]
    assert retry.calls == built
