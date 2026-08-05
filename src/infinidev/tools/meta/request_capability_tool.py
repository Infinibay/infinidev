"""Controlled escape hatch for dynamically narrowed developer toolsets."""

from __future__ import annotations

from typing import Literal, Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


CapabilityName = Literal[
    "web",
    "knowledge",
    "docs",
    "git_mutation",
    "background",
    "advanced_refactor",
    "image_generation",
]


class RequestCapabilityInput(BaseModel):
    capability: CapabilityName
    rationale: str = Field(
        min_length=1,
        description="Why the active user objective now requires this capability.",
    )


class RequestCapabilityTool(InfinibayBaseTool):
    name: str = "request_capability"
    description: str = (
        "Expose one optional tool group omitted by dynamic routing when new evidence "
        "shows it is necessary. This changes availability only: it never grants "
        "permission for Git, destructive, paid, secret-sensitive, or external effects."
    )
    args_schema: Type[BaseModel] = RequestCapabilityInput

    def _run(self, capability: str, rationale: str) -> str:
        from infinidev.tools.base.context import get_context_for_agent

        agent_id = getattr(self, "_bound_agent_id", None) or self.agent_id
        context = get_context_for_agent(agent_id) if agent_id else None
        requester = context.capability_requester if context else None
        if requester is None:
            return self._error("Dynamic capability routing is not active for this run")
        try:
            return requester(capability, rationale)
        except (KeyError, ValueError) as exc:
            return self._error(str(exc))
