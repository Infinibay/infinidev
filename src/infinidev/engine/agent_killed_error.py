"""Abstract base class for agent execution engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.agents.base import InfinibayAgent


class AgentKilledError(RuntimeError):
    """Raised when the agent process is killed (exit code 137/139/-9).

    This can happen intentionally during project shutdown (pods killed) or
    unexpectedly due to OOM / resource limits.  Callers should check whether
    a shutdown is in progress to decide the severity.
    """


