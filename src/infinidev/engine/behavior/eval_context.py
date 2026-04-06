"""StepEvalContext — the input bundle passed to every StochasticChecker.

Built once per POST_STEP dispatch by :class:`BehaviorScorer` and handed
to each checker's :meth:`StochasticChecker.evaluate` so they don't each
re-parse the message buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from infinidev.engine.behavior.primitives.tool_inspect import (
    NormalizedCall,
    iterate_messages_tool_calls,
    normalize_tool_calls,
)


@dataclass
class StepEvalContext:
    """Immutable snapshot of everything a checker may inspect for one step."""

    # Identity — used by checkers that keep per-agent rolling state.
    project_id: int = 0
    agent_id: str = ""

    task: str = ""
    active_step_title: str = ""
    plan_steps: list[dict[str, Any]] = field(default_factory=list)

    # Current step
    iteration: int = 0
    step_status: str = "continue"
    step_summary: str = ""
    reasoning_content: str = ""
    latest_content: str = ""
    action_tool_calls: int = 0

    # Messages and tool calls scoped to THIS step
    step_messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[NormalizedCall] = field(default_factory=list)

    # Prior state
    prior_step_records: list[Any] = field(default_factory=list)
    prior_reasoning_blocks: list[str] = field(default_factory=list)

    # Escape hatch — raw state for exotic checkers
    state: Any = None

    @classmethod
    def from_post_step(
        cls,
        *,
        metadata: dict[str, Any],
        project_id: int = 0,
        agent_id: str = "",
    ) -> "StepEvalContext":
        state = metadata.get("state")
        step_result = metadata.get("step_result")
        messages = metadata.get("messages") or []
        start = int(metadata.get("step_messages_start") or 0)
        # Slice just this step's messages from the live buffer.
        step_messages = messages[start:] if messages else []

        # Latest assistant message in this step (raw content + reasoning)
        latest_content = ""
        reasoning_content = ""
        for m in reversed(step_messages):
            if m.get("role") == "assistant":
                latest_content = (
                    m.get("raw_content") or m.get("content") or ""
                )
                if isinstance(latest_content, list):
                    latest_content = " ".join(
                        b.get("text", "")
                        for b in latest_content
                        if isinstance(b, dict)
                    )
                reasoning_content = m.get("reasoning_content") or ""
                break

        # All tool calls made during this step
        tool_calls = list(iterate_messages_tool_calls(step_messages))

        # Plan snapshot
        task = ""
        active_title = ""
        plan_steps: list[dict[str, Any]] = []
        if state is not None:
            task = getattr(state, "task", "") or ""
            plan = getattr(state, "plan", None)
            if plan is not None:
                active = getattr(plan, "active_step", None)
                if active is not None:
                    active_title = getattr(active, "title", "") or ""
                raw_steps = getattr(plan, "steps", None) or []
                for s in raw_steps:
                    plan_steps.append(
                        {
                            "title": getattr(s, "title", "") or "",
                            "status": getattr(s, "status", "") or "",
                        }
                    )

        prior_records = []
        prior_reasoning: list[str] = []
        if state is not None:
            hist = getattr(state, "history", []) or []
            prior_records = list(hist[-10:])
            # Reasoning blocks from the prior few steps (for repetition check)
            # We don't have reasoning in ActionRecord — fall back to summary.
            prior_reasoning = [
                getattr(r, "summary", "") or "" for r in prior_records
            ]

        return cls(
            project_id=project_id,
            agent_id=agent_id,
            task=task,
            active_step_title=active_title,
            plan_steps=plan_steps,
            iteration=int(metadata.get("iteration", 0) or 0),
            step_status=getattr(step_result, "status", "continue"),
            step_summary=getattr(step_result, "summary", "") or "",
            reasoning_content=reasoning_content,
            latest_content=str(latest_content),
            action_tool_calls=int(metadata.get("action_tool_calls", 0) or 0),
            step_messages=step_messages,
            tool_calls=tool_calls,
            prior_step_records=prior_records,
            prior_reasoning_blocks=prior_reasoning,
            state=state,
        )

    @classmethod
    def from_single_message(
        cls,
        *,
        message: dict[str, Any],
        history: list[dict[str, Any]],
        task: str = "",
        plan_snapshot: dict[str, Any] | None = None,
        project_id: int = 0,
        agent_id: str = "",
    ) -> "StepEvalContext":
        """Build a context for per_message mode (legacy compat)."""
        latest_content = message.get("raw_content") or message.get("content") or ""
        if isinstance(latest_content, list):
            latest_content = " ".join(
                b.get("text", "") for b in latest_content if isinstance(b, dict)
            )
        reasoning_content = message.get("reasoning_content") or ""
        tool_calls = normalize_tool_calls(message.get("tool_calls"))
        plan_steps = (plan_snapshot or {}).get("steps", []) or []
        active_title = (plan_snapshot or {}).get("active_step_title", "") or ""
        return cls(
            project_id=project_id,
            agent_id=agent_id,
            task=task,
            active_step_title=active_title,
            plan_steps=plan_steps,
            iteration=0,
            step_status="continue",
            step_summary="",
            reasoning_content=reasoning_content,
            latest_content=str(latest_content),
            action_tool_calls=len(tool_calls),
            step_messages=list(history) + [message],
            tool_calls=tool_calls,
            prior_step_records=[],
            prior_reasoning_blocks=[],
            state=None,
        )
