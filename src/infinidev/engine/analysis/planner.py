"""Task planner — turns an EscalationPacket into an executable Plan.

Mirrors the self-contained loop shape of chat_agent.py (not a
LoopEngine invocation). Receives the chat agent's handoff packet,
runs up to N read-only exploration calls, and terminates when the
model calls ``emit_task_plan``. The parsed Plan is returned to the
pipeline which feeds it to LoopEngine via ``initial_plan=``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Optional

from infinidev.config.llm import get_litellm_params_for_behavior
from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from infinidev.engine.analysis.staged_planning import TaskPlanningHandoff
from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.formats._normalize import normalize_tool_arguments_json
from infinidev.engine.schema_sanitizer import tool_to_openai_schema
from infinidev.engine.tool_dispatch import build_tool_dispatch, execute_tool_call
from infinidev.engine.token_usage import report_prompt_tokens
from infinidev.engine.oversized_result import (
    DuplicateCallGuard,
    handle_oversized_result,
)
from infinidev.engine.orchestration.escalation_packet import EscalationPacket
from infinidev.prompts.analyst.task_planner_prompt import TASK_PLANNER_SYSTEM_PROMPT
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
    task_handoff: TaskPlanningHandoff | None = None,
    session_id: Optional[str] = None,
    project_id: Optional[int] = None,
    workspace_path: Optional[str] = None,
    max_exploration_calls: int = _DEFAULT_MAX_EXPLORATION_CALLS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    hooks: Any | None = None,
) -> Plan:
    """Produce a Plan from the chat agent's escalation packet.

    When the model exceeds the exploration budget, a nudge message is
    injected telling it to emit now. If it still does not emit, the
    planner returns a minimal single-step Plan derived from
    ``escalation.understanding`` — the pipeline NEVER gets back a null
    plan, because there is no recovery path downstream.
    """
    agent_id = f"planner-{uuid.uuid4().hex[:8]}"
    tools = get_tools_for_role("task_planner")
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
    _user_text = _render_handoff(
        escalation,
        max_exploration_calls,
        task_handoff=task_handoff,
    )
    _user_content: Any = _user_text
    if escalation.attachments:
        try:
            from infinidev.config.model_capabilities import get_capability_snapshot

            _supports_vision = get_capability_snapshot().supports_vision
        except Exception:
            _supports_vision = False
        from infinidev.engine.multimodal import (
            build_user_content, mention_paths_as_text,
        )
        if _supports_vision:
            _user_content = build_user_content(_user_text, escalation.attachments)
        else:
            _user_content = mention_paths_as_text(_user_text, escalation.attachments)

    from infinidev.engine.prompt_profile import apply_calibrated_guidance

    planner_prompt = apply_calibrated_guidance(
        TASK_PLANNER_SYSTEM_PROMPT, "planner"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": planner_prompt},
        {"role": "user", "content": _user_content},
    ]

    try:
        return _run_llm_loop(
            messages=messages,
            tool_schemas=tool_schemas,
            dispatch=dispatch,
            escalation=escalation,
            task_handoff=task_handoff,
            max_exploration_calls=max_exploration_calls,
            max_iterations=max_iterations,
            hooks=hooks,
        )
    except Exception as exc:
        logger.exception("Task Planner loop failed")
        return _fallback_plan(
            escalation,
            f"Task Planner error: {exc}",
            task_handoff=task_handoff,
        )
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
    task_handoff: TaskPlanningHandoff | None,
    max_exploration_calls: int,
    max_iterations: int,
    hooks: Any | None = None,
) -> Plan:
    import litellm

    base_kwargs = get_litellm_params_for_behavior()
    exploration_calls = 0
    budget_nudged = False

    # One guard per run: a repeat inside the same turn is the
    # livelock, a repeat in a later turn is a legitimate re-read.
    dup_guard = DuplicateCallGuard()

    for iteration in range(max_iterations):
        call_kwargs = dict(base_kwargs)
        call_kwargs["messages"] = messages
        call_kwargs["tools"] = tool_schemas
        call_kwargs["tool_choice"] = "required"
        call_kwargs.setdefault("temperature", 0.1)
        call_kwargs.setdefault("stream", False)
        call_kwargs.setdefault("max_tokens", 3000)

        response = litellm.completion(**call_kwargs)
        report_prompt_tokens(
            hooks, response, lane="chat",
            messages=messages, model=call_kwargs.get("model", ""),
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            content = (getattr(message, "content", None) or "").strip()
            # Small / non-FC models often emit the plan as prose JSON instead
            # of a native emit_task_plan call. Recover it before dropping to the
            # bootstrap fallback (the 7B default is exactly this tier).
            recovered = _plan_from_text(content)
            if recovered is not None:
                return recovered
            logger.warning(
                "Task Planner returned text without tool call: %s", content[:200]
            )
            return _fallback_plan(
                escalation,
                "Task Planner did not emit via tool call.",
                task_handoff=task_handoff,
            )

        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
        })

        for tc in tool_calls:
            if tc.function.name == "emit_task_plan":
                return _parse_emitted_plan(tc, escalation, task_handoff)

        # Non-terminator calls — count toward exploration budget. The budget is
        # a real execution ceiling: models may still *propose* more calls in a
        # batch, but those calls receive a refusal rather than performing I/O.
        for tc in tool_calls:
            if exploration_calls >= max_exploration_calls:
                result = (
                    "Exploration budget exhausted. Do not make more read calls; "
                    "emit_task_plan using the evidence already collected."
                )
            else:
                exploration_calls += 1
                result = dup_guard.refusal_for(
                    tc.function.name, tc.function.arguments,
                ) or execute_tool_call(
                    dispatch, tc.function.name, tc.function.arguments,
                )
            trimmed = handle_oversized_result(
                result,
                max_chars=_MAX_RESULT_CHARS,
                tool_name=tc.function.name,
                tool_args=tc.function.arguments,
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
                    f"({exploration_calls} calls). Emit the Task plan now via "
                    "emit_task_plan with the evidence already collected. "
                    "Do not make more read calls."
                ),
            })
            budget_nudged = True

    logger.warning(
        "Task Planner hit max_iterations=%d without emit_task_plan", max_iterations
    )
    return _fallback_plan(
        escalation,
        "Task Planner exhausted iterations without emitting.",
        task_handoff=task_handoff,
    )


# ─────────────────────────────────────────────────────────────────────────
# Handoff + parsing
# ─────────────────────────────────────────────────────────────────────────


def _render_handoff(
    escalation: EscalationPacket,
    max_exploration_calls: int = _DEFAULT_MAX_EXPLORATION_CALLS,
    *,
    task_handoff: TaskPlanningHandoff | None = None,
) -> str:
    if task_handoff is not None:
        return task_handoff.render(max_exploration_calls)
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
        lines.append(
            "opened_files (paths the chat agent judged worth opening; "
            "their contents are NOT included here):"
        )
        for path in escalation.opened_files:
            lines.append(f"  - {path}")
    if escalation.user_signal:
        lines.append("")
        lines.append(f"user_signal (text interpreted as approval): {escalation.user_signal}")
    # The spec-elaboration loop may have attached a GroundedSpec (scope,
    # resolved facts, assumptions, design direction). Render it so the
    # planner decomposes a COMPLETE spec rather than the raw request.
    spec = getattr(escalation, "grounded_spec", None)
    if spec is not None:
        try:
            lines.append("")
            lines.append(spec.render_for_planner())
        except Exception:
            logger.debug("grounded_spec render failed; skipping", exc_info=True)
    # A council may have deliberated upstream and attached a design
    # brief. Render it so the planner builds steps ON TOP of the agreed
    # design instead of re-deciding it. ``getattr`` keeps this robust if
    # an older packet without the field is ever passed in.
    brief = getattr(escalation, "design_brief", None)
    if brief is not None:
        try:
            lines.append("")
            lines.append(brief.render_for_planner())
        except Exception:
            logger.debug("design_brief render failed; skipping", exc_info=True)
    lines.append("")
    lines.append(
        f"Your turn. At most {max_exploration_calls} exploration calls, "
        "then emit the Task plan via emit_task_plan."
    )
    return "\n".join(lines)


def _build_plan_from_args(args: dict) -> Plan | None:
    """Build a validated Plan from Task Planner tool arguments.

    Returns None (rather than a fallback) when the overview or steps are
    empty/malformed, so callers can choose how to recover. Shared by the
    native tool-call path and the prose-JSON recovery path so the two
    cannot drift.
    """
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
            verify=StepVerification.from_loose(s),
        ))
    if not overview or not steps:
        return None

    # Planner-derived Task checks: keep only falsifiable ones (drop vague
    # quality phrases at the authoring boundary instead of trusting them).
    from infinidev.engine.orchestration.task_schema import is_falsifiable
    raw_criteria = (
        args.get("derived_verification_criteria")
        or args.get("acceptance_criteria")
        or []
    )
    if not isinstance(raw_criteria, list):
        raw_criteria = []
    criteria = [c.strip() for c in raw_criteria if isinstance(c, str) and is_falsifiable(c)]

    return Plan(overview=overview, steps=steps, acceptance_criteria=criteria)


def _parse_emitted_plan(
    tc: Any,
    escalation: EscalationPacket,
    task_handoff: TaskPlanningHandoff | None = None,
) -> Plan:
    raw = getattr(tc.function, "arguments", None) or "{}"
    args = raw if isinstance(raw, dict) else _safe_json(raw)
    if not isinstance(args, dict):
        args = {}
    plan = _build_plan_from_args(args)
    if plan is None:
        logger.warning("Task Planner emitted incomplete plan, falling back")
        return _fallback_plan(
            escalation,
            "emit_task_plan produced empty fields",
            task_handoff=task_handoff,
        )
    return plan


def _plan_from_text(content: str) -> Plan | None:
    """Recover a plan from a prose JSON response (no native tool call).

    Small / non-FC models frequently emit ``{"overview": ..., "steps": [...]}``
    as text instead of calling emit_task_plan. Returns a validated Plan when the
    content yields complete fields, else None so the caller falls back.
    """
    if not content or "{" not in content:
        return None
    start = content.find("{")
    end = content.rfind("}")
    if end <= start:
        return None
    args = _safe_json(content[start:end + 1])
    if not isinstance(args, dict):
        return None
    plan = _build_plan_from_args(args)
    if plan is not None:
        logger.info("Task Planner recovered a plan from prose JSON (no tool call)")
    return plan


def _fallback_plan(
    escalation: EscalationPacket,
    reason: str,
    *,
    task_handoff: TaskPlanningHandoff | None = None,
) -> Plan:
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
    logger.warning("task planner falling back: reason=%s", reason)
    if task_handoff is not None:
        overview = (
            f"Goal: {task_handoff.goal.user_request}\n\n"
            f"Active Stage: {task_handoff.stage.title}\n\n"
            f"Current Task: {task_handoff.task.title}\n"
            f"Outcome: {task_handoff.task.outcome}\n\n"
            "No structured Task plan was produced upstream. Decompose only "
            "this Task into steps via add_step before executing."
        )
    else:
        overview = (
            f"User request: {escalation.user_request}\n\n"
            f"Chat agent's understanding: {escalation.understanding}\n\n"
            "No structured plan was produced upstream. Decompose the "
            "request into steps via add_step before executing."
        )
    return Plan(overview=overview, steps=[])


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    return {
        "id": tc.id,
        "type": "function",
        "function": {
            "name": tc.function.name,
            "arguments": normalize_tool_arguments_json(tc.function.arguments),
        },
    }


def _safe_json(s: str) -> dict[str, Any]:
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


__all__ = ["run_planner"]
