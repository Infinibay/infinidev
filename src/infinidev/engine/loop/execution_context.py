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
    max_prompt_tokens: int | None
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
    resumed: bool = False

    # Behavior flags
    skip_plan: bool = False  # True for agents that don't use plan management (e.g. analyst)
    # False when an outer scheduler owns topology. The local Step still
    # renders and closes normally, but the model cannot add sibling work.
    allow_plan_mutation: bool = True
    allow_explore: bool = True

    # Structured task spec — when set, the prompt builder renders the
    # task via ``render_task_xml`` instead of the legacy plain
    # ``<task>desc</task>`` block. Both the principal and (via shared
    # message history) the assistant critic see the same rendering.
    # ``None`` is the legacy path: the engine falls back to ``desc``.
    task: "Task | None" = None

    # Optional, immutable context corpus used by controlled context-delivery
    # evaluations. Production callers leave this unset. Keeping it separate
    # from the task and system prompt lets experiments vary only repository
    # evidence delivery while preserving agent instructions.
    context_corpus: str | None = None
    allow_llm_retries: bool = True

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

    @property
    def session_id(self) -> str:
        """Session this run belongs to — the key working memory is filed under.

        Falls back to ``agent_id`` so a run started outside a chat session
        (tests, one-shot CLI invocations) still gets a stable archive key
        instead of colliding with every other sessionless run.
        """
        from infinidev.tools.base.context import get_current_session_id

        return get_current_session_id() or self.agent_id
