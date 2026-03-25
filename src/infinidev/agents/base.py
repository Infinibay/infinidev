"""Base agent wrapper for Infinidev agents."""

from __future__ import annotations
import logging
from typing import Any
from infinidev.tools.base.context import bind_tools_to_agent, set_context

logger = logging.getLogger(__name__)

class InfinidevAgent:
    """Simplified agent for Infinidev CLI.

    Responsibilities:
    - Hold role, backstory, goal
    - Provide tools for the role
    - Manage context for execution
    """

    def __init__(
        self,
        *,
        agent_id: str,
        role: str = "agent",
        name: str = "Infinidev",
        goal: str = "Assist the user with programming and research tasks.",
        backstory: str = "Expert software engineer and technical researcher.",
        project_id: int = 1,
        extra_tools: list | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.role = role
        self.name = name
        self.goal = goal
        self.backstory = backstory
        self.project_id = project_id
        self._tech_hints: list[str] | None = None
        self._session_summaries: list[str] | None = None
        self._session_id: str | None = None

        # Import tools dynamically to avoid circular imports
        from infinidev.tools import get_tools_for_role
        tools = get_tools_for_role(role)
        if extra_tools:
            instantiated_extras = [
                t() if isinstance(t, type) else t
                for t in extra_tools
            ]
            tools = tools + instantiated_extras

        # Stamp tools with agent context
        bind_tools_to_agent(tools, agent_id)
        self._tools = tools

    @property
    def tools(self) -> list:
        return self._tools

    def activate_context(self, *, session_id: str | None = None) -> None:
        """Set execution context for tools."""
        import os
        self._session_id = session_id
        # INFINIDEV_WORKSPACE env var takes priority (set by bench runner and external tools)
        workspace_path = os.environ.get("INFINIDEV_WORKSPACE") or os.getcwd()
        set_context(
            project_id=self.project_id,
            agent_id=self.agent_id,
            session_id=session_id,
            workspace_path=workspace_path,
        )

    def deactivate(self) -> None:
        """Clean up if needed (no-op in CLI)."""
        pass
