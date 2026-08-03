"""Engine package — agent loop orchestration.

Convenience re-exports for the most commonly used classes:

    from infinidev.engine import LoopEngine, TreeEngine
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.engine.base import AgentEngine
    from infinidev.engine.loop import LoopEngine
    from infinidev.engine.tree import TreeEngine

__all__ = ["LoopEngine", "TreeEngine", "AgentEngine"]


def __getattr__(name: str) -> Any:
    """Load convenience exports without importing the complete engine eagerly."""
    if name == "LoopEngine":
        from infinidev.engine.loop import LoopEngine

        value = LoopEngine
    elif name == "TreeEngine":
        from infinidev.engine.tree import TreeEngine

        value = TreeEngine
    elif name == "AgentEngine":
        from infinidev.engine.base import AgentEngine

        value = AgentEngine
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
