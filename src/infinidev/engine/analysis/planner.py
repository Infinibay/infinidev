"""Analyst planner — turns an EscalationPacket into an executable Plan.

Mirrors the self-contained loop shape of chat_agent.py (not a
LoopEngine invocation). Receives the chat agent's handoff packet,
runs up to N read-only exploration calls, and terminates when the
model calls ``emit_plan``. The parsed Plan is returned to the
pipeline which feeds it to LoopEngine via ``initial_plan=``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from infinidev.config.llm import get_litellm_params_for_behavior
from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from infinidev.engine.loop.schema_sanitizer import tool_to_openai_schema
from infinidev.engine.loop.tools import build_tool_dispatch, execute_tool_call
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.prompts.analyst.planner_prompt import ANALYST_PLANNER_SYSTEM_PROMPT
from infinidev.tools import get_tools_for_role
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)

logger = logging.getLogger(__name__)


_DEFAULT_MAX_EXPLORATION_CALLS = 4
_DEFAULT_MAX_ITERATIONS = 6  # exploration + emit turn — upper cap
_MAX_RESULT_CHARS = 8000


def run_planner(
    escalation: EscalationPacket,
    *,
    session_id: Optional[str] = None,
    project_id: Optional[int] = None,
    workspace_path: Optional[str] = None,
    max_exploration_calls: int = _DEFAULT_MAX_EXPLORATION_CALLS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
) -> Plan:
    """Produce a Plan from the chat agent's escalation packet.

    When the model exceeds the exploration budget, a nudge message is
    injected telling it to emit now. If it still does not emit, the
    planner returns a minimal single-step Plan derived from
    ``escalation.understanding`` — the pipeline NEVER gets back a null
    plan, because there is no recovery path downstream.
    """
    agent_id = f"planner-{uuid.uuid4().hex[:8]}"
    tools = get_tools_for_role("planner")
    bind_tools_to_agent(tools, agent_id)
    set_context(
        agent_id=agent_id,
        project_id=project_id,
        session_id=session_id,
        workspace_path=workspace_path,
    )

    dispatch = build_tool_dispatch(tools)
    tool_schemas = [tool_to_openai_schema(t) for t in tools]

    # If the user attached images in chat and the escalation forwarded
    # them, make them visible to the planner too (it may need to decide
    # scope based on what the image actually shows). Text-only fallback
    # for non-vision models so the paths are at least mentioned.
    _user_text = _render_handoff(escalation)
    _user_content: Any = _user_text
    if escalation.attachments:
        try:
            from infinidev.config.model_capabilities import _detect_vision_support
            _supports_vision = _detect_vision_support()
        except Exception:
            _supports_vision = False
        from infinidev.engine.multimodal import (
            build_user_content, mention_paths_as_text,
        )
        if _supports_vision:
            _user_content = build_user_content(_user_text, escalation.attachments)
        else:
            _user_content = mention_paths_as_text(_user_text, escalation.attachments)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": ANALYST_PLANNER_SYSTEM_PROMPT},
        {"role": "user", "content": _user_content},
    ]

    try:
        return _run_llm_loop(
            messages=messages,
            tool_schemas=tool_schemas,
            dispatch=dispatch,
            escalation=escalation,
            max_exploration_calls=max_exploration_calls,
            max_iterations=max_iterations,
        )
    except Exception as exc:
        logger.exception("Planner loop failed")
        return _fallback_plan(escalation, f"Planner error: {exc}")
    finally:
        clear_agent_context(agent_id)


# ─────────────────────────────────────────────────────────────────────────
# Loop driver
# ─────────────────────────────────────────────────────────────────────────


def _run_llm_loop(
    *,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    dispatch: dict[str, Any],
    escalation: EscalationPacket,
    max_exploration_calls: int,
    max_iterations: int,
) -> Plan:
    import litellm

    base_kwargs = get_litellm_params_for_behavior()
    exploration_calls = 0
    budget_nudged = False

    for iteration in range(max_iterations):
        call_kwargs = dict(base_kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["tools"] = tool_schemas
        call_kwargs.setdefault("temperature", 0.1)
        call_kwargs.setdefault("stream", False)
        call_kwargs.setdefault("max_tokens", 3000)

        response = litellm.completion(**call_kwargs)
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            content = (getattr(message, "content", None) or "").strip()
            logger.warning(
                "Planner returned text without tool call: %s", content[:200]
            )
            return _fallback_plan(
                escalation,
                "Planner did not emit via tool call.",
            )

        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
        })

        for tc in tool_calls:
            if tc.function.name == "emit_plan":
                return _parse_emitted_plan(tc, escalation)

        # Non-terminator calls — count toward exploration budget.
        for tc in tool_calls:
            exploration_calls += 1
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

        if exploration_calls >= max_exploration_calls and not budget_nudged:
            messages.append({
                "role": "user",
                "content": (
                    "You have used your exploration budget "
                    f"({exploration_calls} calls). Emit the plan NOW via "
                    "emit_plan with whatever you have. Do not make more "
                    "read calls."
                ),
            })
            budget_nudged = True

    logger.warning(
        "Planner hit max_iterations=%d without emit_plan", max_iterations
    )
    return _fallback_plan(
        escalation,
        "Planner exhausted iterations without emitting.",
    )


# ─────────────────────────────────────────────────────────────────────────
# Handoff + parsing
# ─────────────────────────────────────────────────────────────────────────


def _render_handoff(escalation: EscalationPacket) -> str:
    lines = [
        "HANDOFF FROM CHAT AGENT",
        "",
        f"user_request (verbatim):\n  {escalation.user_request}",
        "",
        f"understanding (chat agent's words):\n  {escalation.understanding}",
        "",
        f"suggested_flow: {escalation.suggested_flow}",
    ]
    if escalation.opened_files:
        lines.append("")
        lines.append("opened_files (already read by chat agent — do NOT re-open):")
        for path in escalation.opened_files:
            lines.append(f"  - {path}")
    if escalation.user_signal:
        lines.append("")
        lines.append(f"user_signal (text interpreted as approval): {escalation.user_signal}")
    lines.append("")
    lines.append(
        "Your turn. Explore at most 4 files if truly needed, "
        "then emit the plan via emit_plan."
    )
    return "\n".join(lines)


def _parse_emitted_plan(tc: Any, escalation: EscalationPacket) -> Plan:
    raw = getattr(tc.function, "arguments", None) or "{}"
    args = raw if isinstance(raw, dict) else _safe_json(raw)
    overview = (args.get("overview") or "").strip()
    raw_steps = args.get("steps") or []
    if not isinstance(raw_steps, list):
        raw_steps = []
    steps: list[PlanStepSpec] = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        title = (s.get("title") or "").strip()
        if not title:
            continue
        steps.append(PlanStepSpec(
            title=title,
            detail=(s.get("detail") or "").strip(),
            expected_output=(s.get("expected_output") or "").strip(),
        ))
    if not overview or not steps:
        logger.warning(
            "Planner emitted incomplete plan (overview=%r, steps=%d), "
            "falling back", overview[:60], len(steps),
        )
        return _fallback_plan(escalation, "emit_plan produced empty fields")
    return Plan(overview=overview, steps=steps)


def _fallback_plan(escalation: EscalationPacket, reason: str) -> Plan:
    """Last-resort plan: hand the request over to the developer with
    context, and let it bootstrap its own plan via ``add_step``.

    The overview carries the user request and chat-agent understanding
    so the developer has context in its first prompt. ``steps`` is
    empty — this signals the LoopEngine bootstrap branch, which tells
    the model "no plan yet, call add_step" rather than treating a
    single pre-seeded step as the whole plan. Previously we returned a
    one-step plan that the LLM couldn't modify (user_approved=True),
    which caused the developer to execute everything inside a single
    monolithic step instead of decomposing the work.

    The *reason* is LOGGED, not embedded in the overview, because the
    overview is rendered every iteration and repeating a debug string
    as context would confuse the model.
    """
    logger.warning("planner falling back: reason=%s", reason)
    return Plan(
        overview=(
            f"User request: {escalation.user_request}\n\n"
            f"Chat agent's understanding: {escalation.understanding}\n\n"
            "No structured plan was produced upstream. Decompose the "
            "request into steps via add_step before executing."
        ),
        steps=[],
    )


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
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


def _safe_json(s: str) -> dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


__all__ = ["run_planner"]
