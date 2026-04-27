"""Execution context dataclass — shared state for a single execute() invocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from infinidev.engine.loop.models import LoopState
from infinidev.engine.file_change_tracker import FileChangeTracker

if TYPE_CHECKING:
    from infinidev.engine.orchestration.task_schema import Task


@dataclass
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

    # Behavior flags
    skip_plan: bool = False  # True for agents that don't use plan management (e.g. analyst)

    # Structured task spec — when set, the prompt builder renders the
    # task via ``render_task_xml`` instead of the legacy plain
    # ``<task>desc</task>`` block. Both the principal and (via shared
    # message history) the assistant critic see the same rendering.
    # ``None`` is the legacy path: the engine falls back to ``desc``.
    task: "Task | None" = None

    # Phase-specific over-budget warning. Used by the analyst (and any
    # future restricted-tools phase) to override the developer-oriented
    # default nudge with phase-appropriate language. Two ``{}``
    # placeholders are filled at injection time:
    #   {used}      → action_tool_calls so far this step
    #   {threshold} → the configured nudge_threshold
    # When ``None``, the engine uses its built-in developer message.
    nudge_message_template: str | None = None

    @property
    def project_id(self) -> int:
        return self.agent.project_id

    @property
    def agent_id(self) -> str:
        return self.agent.agent_id
