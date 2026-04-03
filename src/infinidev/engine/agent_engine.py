"""Abstract base class for agent execution engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.agents.base import InfinibayAgent


class AgentEngine(ABC):
    """Interface for agent execution engines (CrewAI, Claude Code, etc.)."""

    @abstractmethod
    def execute(
        self,
        agent: InfinibayAgent,
        task_prompt: tuple[str, str],
        *,
        verbose: bool = True,
        guardrail: Any | None = None,
        guardrail_max_retries: int = 5,
        output_pydantic: type | None = None,
        task_tools: list | None = None,
        event_id: int | None = None,
        resume_state: dict | None = None,
    ) -> str:
        """Execute a task using the given agent and return the string result.

        Args:
            agent: A InfinibayAgent instance with role, backstory, tools, etc.
            task_prompt: (description, expected_output) tuple.
            verbose: Whether to enable verbose logging.
            guardrail: Validation function for the output.
            guardrail_max_retries: Max retries when guardrail fails.
            output_pydantic: Pydantic model for structured output.
            task_tools: Override task-specific tools.
            event_id: Agent event row ID for crash recovery checkpointing.
            resume_state: LoopState dict to resume from after crash.
        """

