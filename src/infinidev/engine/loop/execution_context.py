"""Execution context dataclass — shared state for a single execute() invocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infinidev.engine.loop_models import LoopState
from infinidev.engine.file_change_tracker import FileChangeTracker


class ExecutionContext:
    """All shared state for a single execute() invocation.

    Replaces ~20 local variables that were threaded through the old
    monolithic execute() method. Components read config fields and
    mutate ``state`` / ``file_tracker`` as needed.
    """

    # Config (immutable after setup)
    llm_params: dict[str, Any]
    manual_tc: bool
    is_small: bool
    system_prompt: str
    tool_schemas: list[dict[str, Any]]
    tool_dispatch: dict[str, Any]
    planning_schemas: list[dict[str, Any]]
    tools: list[Any]
    max_iterations: int
    max_per_action: int
    max_total_calls: int
    history_window: int
    max_context_tokens: int
    verbose: bool
    guardrail: Any | None
    guardrail_max_retries: int
    output_pydantic: type | None

    # Agent identity
    agent: Any
    agent_name: str
    agent_role: str
    desc: str
    expected: str
    event_id: int | None

    # Mutable state
    state: LoopState
    file_tracker: FileChangeTracker
    start_iteration: int

    @property
    def project_id(self) -> int:
        return self.agent.project_id

    @property
    def agent_id(self) -> str:
        return self.agent.agent_id
