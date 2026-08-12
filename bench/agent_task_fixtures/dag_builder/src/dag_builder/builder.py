"""Incremental build orchestration."""

from dag_builder.actions import BuildActions
from dag_builder.graph import BuildGraph
from dag_builder.sources import SourceIndex
from dag_builder.state import BuildState


class IncrementalBuilder:
    def __init__(
        self,
        graph: BuildGraph,
        sources: SourceIndex,
        actions: BuildActions,
        state: BuildState,
    ) -> None:
        self.graph = graph
        self.sources = sources
        self.actions = actions
        self.state = state

    def build(self) -> list[str]:
        fingerprints = self.sources.fingerprints()
        selected = self.graph.affected_by(self.state.changed(fingerprints))
        built: list[str] = []
        for name in sorted(selected):
            self.state.mark_attempt(name, fingerprints[name])
            self.actions.build(name)
            built.append(name)
        return built
