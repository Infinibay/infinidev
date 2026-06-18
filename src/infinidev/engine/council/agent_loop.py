"""Shared terminating LLM loop for council agents.

Both council members and the moderator run the same shape: a short,
self-contained ``litellm.completion`` loop with read-only exploration
tools and one or more *terminator* tools whose args the caller reads
directly. This helper factors that out so member.py and moderator.py
stay thin — they just supply a system prompt, a user message, the tool
set, and which tool names terminate the turn.

Mirrors the structure of ``orchestration.chat_agent._run_llm_loop`` and
``analysis.planner._run_llm_loop`` (non-streaming, manual transcript
assembly, budget nudge), but is generic over the terminator set.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from infinidev.config.llm import get_litellm_params_for_behavior
from infinidev.config.settings import settings
from infinidev.engine.schema_sanitizer import tool_to_openai_schema
from infinidev.engine.tool_dispatch import build_tool_dispatch, execute_tool_call
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)

logger = logging.getLogger(__name__)

_MAX_RESULT_CHARS = 6000


@dataclass
class LoopResult:
    """Outcome of a terminating loop.

    ``terminator`` is the name of the terminator tool the agent called
    (e.g. ``"channel_post"``), or ``None`` if the agent exhausted its
    budget without terminating. ``args`` is that tool call's parsed
    arguments. ``explored`` lists the read-only tool names invoked, for
    audit/debug.
    """

    terminator: str | None
    args: dict[str, Any] = field(default_factory=dict)
    explored: list[str] = field(default_factory=list)


def run_terminating_loop(
    *,
    system_prompt: str,
    user_content: str,
    tools: list[Any],
    terminator_names: set[str],
    max_iterations: int,
    agent_id_prefix: str,
    project_id: int | None = None,
    session_id: str | None = None,
    workspace_path: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1500,
) -> LoopResult:
    """Run one agent turn until it calls a terminator (or runs out of budget).

    The agent gets an isolated context slot (its own ``agent_id``), so
    many of these can run concurrently without their tool contexts
    colliding — that is what makes parallel council members safe.
    """
    agent_id = f"{agent_id_prefix}-{uuid.uuid4().hex[:8]}"
    bind_tools_to_agent(tools, agent_id)
    set_context(
        agent_id=agent_id,
        project_id=project_id,
        session_id=session_id,
        workspace_path=workspace_path,
    )
    dispatch = build_tool_dispatch(tools)
    tool_schemas = [tool_to_openai_schema(t) for t in tools]

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    explored: list[str] = []

    try:
        import litellm

        base_kwargs = _council_base_kwargs()
        budget_nudged = False

        for iteration in range(max_iterations):
            if (
                not budget_nudged
                and max_iterations >= 3
                and iteration == max_iterations - 1
            ):
                messages.append({
                    "role": "user",
                    "content": (
                        "This is your last turn — stop exploring and call "
                        f"one of: {', '.join(sorted(terminator_names))}."
                    ),
                })
                budget_nudged = True

            call_kwargs = dict(base_kwargs)
            call_kwargs["messages"] = messages
            call_kwargs["tools"] = tool_schemas
            call_kwargs.setdefault("temperature", temperature)
            call_kwargs.setdefault("max_tokens", max_tokens)
            call_kwargs["stream"] = False

            response = litellm.completion(**call_kwargs)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:
                # No tool call — the agent emitted plain text. For a
                # terminating loop that's a dead end; log and stop.
                logger.debug(
                    "%s: no tool call at iter %d; ending", agent_id_prefix, iteration,
                )
                return LoopResult(terminator=None, explored=explored)

            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", None) or "",
                "tool_calls": [_tc_to_dict(tc) for tc in tool_calls],
            })

            # Terminator wins immediately if present in this batch.
            for tc in tool_calls:
                if tc.function.name in terminator_names:
                    return LoopResult(
                        terminator=tc.function.name,
                        args=_parse_args(tc),
                        explored=explored,
                    )

            # Otherwise dispatch read-only exploration tools and continue.
            for tc in tool_calls:
                explored.append(tc.function.name)
                result = execute_tool_call(
                    dispatch, tc.function.name, tc.function.arguments,
                )
                trimmed = result if len(result) <= _MAX_RESULT_CHARS else (
                    result[:_MAX_RESULT_CHARS] + "\n...[truncated]"
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": trimmed,
                })

        logger.info(
            "%s exhausted %d iterations without a terminator",
            agent_id_prefix, max_iterations,
        )
        return LoopResult(terminator=None, explored=explored)
    finally:
        clear_agent_context(agent_id)


def _council_base_kwargs() -> dict[str, Any]:
    """Base litellm kwargs for council agents.

    Reuses the behavior-judge param builder (same as the chat agent /
    planner auxiliary loops), then applies an optional ``COUNCIL_MODEL``
    override so a user can point the whole council at a different model
    (e.g. a cloud provider for real parallelism) without touching the
    main developer model.
    """
    kwargs = get_litellm_params_for_behavior()
    override = (getattr(settings, "COUNCIL_MODEL", "") or "").strip()
    if override:
        model = override
        if model.startswith("ollama/"):
            model = "ollama_chat/" + model[len("ollama/"):]
        kwargs["model"] = model
    return kwargs


def _parse_args(tc: Any) -> dict[str, Any]:
    raw = getattr(tc.function, "arguments", None) or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _tc_to_dict(tc: Any) -> dict[str, Any]:
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": (
                tc.function.arguments
                if isinstance(tc.function.arguments, str)
                else json.dumps(tc.function.arguments)
            ),
        },
    }


__all__ = ["run_terminating_loop", "LoopResult"]
