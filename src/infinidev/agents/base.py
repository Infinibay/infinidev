"""Base agent wrapper for Infinidev agents."""

from __future__ import annotations
import logging
from typing import Any
from infinidev.tools.base.context import bind_tools_to_agent, set_context


def _mcp_generation() -> int:
    """How many MCP tools discovery has produced so far.

    Used as a cheap change signal: it starts at 0 while the servers are still
    warming and settles once they answer, so a rebuilt toolset is triggered
    exactly once per session rather than polled.
    """
    try:
        from infinidev.tools.mcp_bridge import discover_mcp_tool_classes

        return len(discover_mcp_tool_classes())
    except Exception:
        return 0

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

        self._role = role
        self._extra_tools = extra_tools
        self._mcp_generation = -1
        self._tools: list = []
        self._resolve_tools()

    def _resolve_tools(self) -> None:
        """Build the toolset and remember which MCP generation it reflects."""
        # Import tools dynamically to avoid circular imports
        from infinidev.tools import get_tools_for_role

        tools = get_tools_for_role(self._role)
        if self._extra_tools:
            tools = tools + [
                t() if isinstance(t, type) else t for t in self._extra_tools
            ]

        # Stamp tools with agent context
        bind_tools_to_agent(tools, self.agent_id)
        self._tools = tools
        self._mcp_generation = _mcp_generation()

    @property
    def tools(self) -> list:
        """The agent's toolset, refreshed if MCP discovery has since filled.

        MCP servers are warmed on a background thread, so an agent built in
        the first moments of a session resolves its tools before any server
        has answered ``tools/list`` — and used to keep that empty result for
        its entire life, silently running without the project index. Checking
        a generation counter costs a dict lookup and lets the toolset heal on
        the next turn instead of the next process.
        """
        if _mcp_generation() != self._mcp_generation:
            self._resolve_tools()
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
