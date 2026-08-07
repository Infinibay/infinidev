"""Engine adapters and the coordinating selector.

Each adapter normalizes one execution strategy onto
:class:`EngineResult`; the coordinator picks among them and records the
event log. See docs/GRAPH_ENGINE_BETA_DESIGN.md §8, §12.
"""

from infinidev.engine.engines.base import (
    EngineAdapter,
    EngineResult,
    TransitionRequest,
)
from infinidev.engine.engines.coordinator import run_selected_engine
from infinidev.engine.engines.react import ReactAdapter
from infinidev.engine.engines.routing import EngineSelection, select_engine
from infinidev.engine.engines.staged_adapter import StagedAdapter
from infinidev.engine.engines.task import TaskAdapter

__all__ = [
    "EngineAdapter",
    "EngineResult",
    "EngineSelection",
    "ReactAdapter",
    "StagedAdapter",
    "TaskAdapter",
    "TransitionRequest",
    "run_selected_engine",
    "select_engine",
]
