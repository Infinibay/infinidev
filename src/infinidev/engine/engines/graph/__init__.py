"""Graph Engine (beta) — typed, versioned work-graph orchestration.

The coordinator dispatches explicit ``graph_beta`` selections here and
``auto`` may choose it when enabled. Leaf work runs through LoopEngine without
leaving the graph or entering the Stage Planner.
See docs/GRAPH_ENGINE_BETA_DESIGN.md.
"""

from infinidev.engine.engines.graph.completion import (
    GoalAssessment,
    NodeBudget,
    RunBudget,
    evaluate_goal,
    is_goal_complete,
)
from infinidev.engine.engines.graph.context import (
    NodeContextCapsule,
    build_capsule,
    render_capsule,
)
from infinidev.engine.engines.graph.engine import GraphEngineAdapter, LeafExecutor
from infinidev.engine.engines.graph.persistence import GraphPersistence
from infinidev.engine.engines.graph.reducer import GraphInvariantError, reduce
from infinidev.engine.engines.graph.scheduler import (
    SchedulerLimits,
    ready_frontier,
    select_next,
)

__all__ = [
    "GoalAssessment",
    "GraphEngineAdapter",
    "GraphInvariantError",
    "GraphPersistence",
    "LeafExecutor",
    "NodeBudget",
    "NodeContextCapsule",
    "RunBudget",
    "SchedulerLimits",
    "build_capsule",
    "evaluate_goal",
    "is_goal_complete",
    "ready_frontier",
    "reduce",
    "render_capsule",
    "select_next",
]
