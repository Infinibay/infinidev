"""Abstract base class for agent execution engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.agents.base import InfinibayAgent


from infinidev.engine.agent_killed_error import AgentKilledError
from infinidev.engine.agent_engine import AgentEngine
