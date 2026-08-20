"""Runtime for choosing the next Stage or a terminal Goal decision."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from infinidev.config.llm import get_litellm_params_for_behavior
from infinidev.engine.analysis.staged_planning import (
    BlockGoalDecision,
    CompleteGoalDecision,
    EmitStageDecision,
    EvidenceEntry,
    StageDecision,
    StageSpec,
    StagedPlanningState,
)
from infinidev.engine.formats._normalize import normalize_tool_arguments_json
from infinidev.engine.llm_client import call_llm
from infinidev.engine.oversized_result import DuplicateCallGuard, handle_oversized_result
from infinidev.engine.prompt_profile import apply_calibrated_guidance
from infinidev.engine.schema_sanitizer import tool_to_openai_schema
from infinidev.engine.token_usage import report_prompt_tokens
from infinidev.engine.tool_dispatch import build_tool_dispatch, execute_tool_call
from infinidev.prompts.analyst.stage_planner_prompt import (
    build_stage_planner_system_prompt,
)
from infinidev.prompts.profiles import EffectivePromptConfiguration
from infinidev.tools import get_tools_for_role
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_EXPLORATION_CALLS = 6
_DEFAULT_MAX_ITERATIONS = 8
_MAX_RESULT_CHARS = 8000
_MAX_STALL_NUDGES = 3
_TERMINALS = {"emit_stage", "complete_goal", "block_goal"}


def run_stage_planner(
    state: StagedPlanningState,
    *,
    session_id: str | None = None,
    project_id: int | None = None,
    workspace_path: str | None = None,
    max_exploration_calls: int = _DEFAULT_MAX_EXPLORATION_CALLS,
    max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    hooks: Any | None = None,
    prompt_configuration: EffectivePromptConfiguration | None = None,
) -> StageDecision:
    """Evaluate the Goal from durable history and return exactly one decision.

    Planner failures and exhausted resource budgets return an explicit blocked
    decision. They never masquerade as Goal completion or an empty successful
    queue.
    """
    if prompt_configuration is None:
        prompt_configuration = EffectivePromptConfiguration.compile()

    agent_id = f"stage-planner-{uuid.uuid4().hex[:8]}"
    tools = get_tools_for_role("stage_planner")
    bind_tools_to_agent(tools, agent_id)
    set_context(
        agent_id=agent_id,
        project_id=project_id,
        session_id=session_id,
        workspace_path=workspace_path,
    )
    dispatch = build_tool_dispatch(tools)
    tool_schemas = [tool_to_openai_schema(tool) for tool in tools]
    stage_planner_prompt = apply_calibrated_guidance(
        build_stage_planner_system_prompt(configuration=prompt_configuration),
        "planner",
    )
    from infinidev.config.settings import settings
    from infinidev.engine.task_policies.rendering import (
        compose_task_aware_system_prompt,
    )

    stage_planner_prompt = compose_task_aware_system_prompt(
        stage_planner_prompt,
        state.goal.task_profile,
        role="planner",
        phase="plan",
        max_utf8_bytes=settings.TASK_POLICIES_MAX_UTF8_BYTES,
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": stage_planner_prompt,
        },
        {
            "role": "user",
            "content": _render_handoff(state, max_exploration_calls),
        },
    ]

    try:
        return _run_llm_loop(
            state=state,
            messages=messages,
            tool_schemas=tool_schemas,
            dispatch=dispatch,
            max_exploration_calls=max_exploration_calls,
            max_iterations=max_iterations,
            hooks=hooks,
        )
    except Exception as exc:
        logger.exception("Stage Planner loop failed")
        return _failure_decision(f"Stage Planner error: {type(exc).__name__}")
    finally:
        clear_agent_context(agent_id)


def _run_llm_loop(
    *,
    state: StagedPlanningState,
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    dispatch: dict[str, Any],
    max_exploration_calls: int,
    max_iterations: int,
    hooks: Any | None,
) -> StageDecision:
    base_kwargs = get_litellm_params_for_behavior()
    exploration_calls = 0
    budget_nudged = False
    stall_nudges = 0
    duplicate_guard = DuplicateCallGuard()

    for _iteration in range(max_iterations):
        call_kwargs = dict(base_kwargs)
        call_kwargs.setdefault("temperature", 0.1)
        call_kwargs.setdefault("stream", False)
        call_kwargs.setdefault("max_tokens", 3500)

        response = call_llm(
            call_kwargs,
            messages,
            tools=tool_schemas,
            tool_choice="required",
        )
        report_prompt_tokens(
            hooks,
            response,
            lane="chat",
            messages=messages,
            model=call_kwargs.get("model", ""),
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []

        if not tool_calls:
            content = (getattr(message, "content", None) or "").strip()
            recovered = _decision_from_text(content)
            if recovered is not None:
                error = _decision_error(recovered, state)
                if not error:
                    return recovered
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": error})
                continue
            logger.warning("Stage Planner returned text without a decision")
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    "Your previous turn produced reasoning text but no terminal "
                    "decision. Call exactly one of emit_stage, complete_goal, or "
                    "block_goal now, with complete arguments matching its schema."
                ),
            })
            stall_nudges += 1
            if stall_nudges >= _MAX_STALL_NUDGES:
                logger.warning(
                    "Stage Planner stalled without a decision after %d nudges",
                    stall_nudges,
                )
                return _failure_decision(
                    "Stage Planner stalled without a terminal decision after "
                    f"{stall_nudges} nudges."
                )
            continue

        stall_nudges = 0
        messages.append({
            "role": "assistant",
            "content": getattr(message, "content", None) or "",
            "tool_calls": [_tool_call_to_dict(call) for call in tool_calls],
        })
        terminal_calls = [
            call for call in tool_calls if call.function.name in _TERMINALS
        ]
        if len(terminal_calls) > 1:
            for call in tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": (
                        "Invalid Stage Planner turn: call exactly one terminal tool."
                    ),
                })
            continue
        if terminal_calls:
            terminal = terminal_calls[0]
            decision = _parse_terminal(terminal.function.name, terminal.function.arguments)
            error = _decision_error(decision, state) if decision is not None else (
                "The terminal arguments were invalid. Emit one complete decision."
            )
            if not error and decision is not None:
                return decision
            for call in tool_calls:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": error or "Invalid terminal decision.",
                })
            continue

        for call in tool_calls:
            if exploration_calls >= max_exploration_calls:
                result = (
                    "Exploration budget exhausted. Do not make more read calls; "
                    "call emit_stage, complete_goal, or block_goal now."
                )
            else:
                exploration_calls += 1
                result = duplicate_guard.refusal_for(
                    call.function.name, call.function.arguments
                ) or execute_tool_call(
                    dispatch, call.function.name, call.function.arguments
                )
            trimmed = handle_oversized_result(
                result,
                max_chars=_MAX_RESULT_CHARS,
                tool_name=call.function.name,
                tool_args=call.function.arguments,
            )
            observation = EvidenceEntry(
                kind="stage_planner_observation",
                summary=trimmed[:2000],
                details={"tool": call.function.name},
            )
            if not state.add_evidence(observation):
                observation = next(
                    entry
                    for entry in state.evidence
                    if entry.fingerprint() == observation.fingerprint()
                )
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": f"[evidence-ledger-id: {observation.id}]\n{trimmed}",
            })

        if exploration_calls >= max_exploration_calls and not budget_nudged:
            messages.append({
                "role": "user",
                "content": (
                    "The read-only exploration budget is exhausted. Decide from "
                    "the evidence already collected and call exactly one of "
                    "emit_stage, complete_goal, or block_goal."
                ),
            })
            budget_nudged = True

    logger.warning("Stage Planner exhausted max_iterations=%d", max_iterations)
    return _failure_decision(
        "Stage Planner exhausted its iteration budget without a valid decision."
    )


def _render_handoff(state: StagedPlanningState, exploration_budget: int) -> str:
    history = [
        {
            "id": stage.id,
            "number": stage.number,
            "status": stage.status,
            "spec": stage.spec.model_dump(mode="json"),
            "tasks": [{
                "id": task.spec.id,
                "title": task.spec.title,
                "status": task.status,
                "result": task.result[:2000],
                "error": task.error[:1000],
                "evidence_ids": task.evidence_ids,
            } for task in stage.tasks],
            "outcome_summary": stage.outcome_summary,
        }
        for stage in state.stages
    ]
    payload = {
        "goal": state.goal.model_dump(mode="json"),
        "goal_status": state.status,
        "new_user_guidance": state.guidance,
        "stage_history": history,
        "evidence_ledger": [{
            "id": entry.id,
            "kind": entry.kind,
            "summary": entry.summary[:2000],
            "stage_id": entry.stage_id,
            "task_id": entry.task_id,
        } for entry in state.evidence],
    }
    return (
        "STAGE PLANNING HANDOFF\n"
        "Authority labels: goal.user_request and goal.acceptance_criteria are "
        "USER_LITERAL. planning_context, Stage checks and Task checks are DERIVED. "
        "Only evidence_ledger entries are OBSERVED_EVIDENCE.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        f"You may use at most {exploration_budget} read-only exploration calls. "
        "Then call exactly one terminal decision tool."
    )


def _parse_terminal(name: str, raw_arguments: Any) -> StageDecision | None:
    args = raw_arguments if isinstance(raw_arguments, dict) else _safe_json(raw_arguments)
    if not isinstance(args, dict):
        return None
    try:
        if name == "emit_stage":
            return EmitStageDecision(stage=StageSpec.model_validate(args))
        if name == "complete_goal":
            return CompleteGoalDecision.model_validate({"evidence": args.get("evidence")})
        if name == "block_goal":
            return BlockGoalDecision.model_validate(args)
    except Exception:
        logger.debug("invalid Stage Planner terminal arguments", exc_info=True)
    return None


def _decision_from_text(content: str) -> StageDecision | None:
    if not content or "{" not in content:
        return None
    obj = _safe_json(content[content.find("{"):content.rfind("}") + 1])
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("name") or obj.get("tool") or "")
    args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) else obj
    kind = str(obj.get("kind") or "")
    if not name:
        name = {
            "stage": "emit_stage",
            "goal_complete": "complete_goal",
            "goal_blocked": "block_goal",
        }.get(kind, "")
    return _parse_terminal(name, args)


def _decision_error(
    decision: StageDecision | None,
    state: StagedPlanningState,
) -> str:
    if decision is None:
        return "The Stage decision did not match a terminal tool schema."
    if isinstance(decision, EmitStageDecision):
        prior = state.active_stage
        if (
            state.goal.intent == "implementation"
            and decision.stage.purpose == "discovery"
            and prior is not None
            and prior.spec.purpose == "discovery"
        ):
            return (
                "emit_stage was rejected: an implementation Goal cannot run "
                "consecutive discovery Stages. Emit a delivery Stage using the "
                "observed evidence, or block on the concrete obstacle."
            )
    if isinstance(decision, CompleteGoalDecision):
        if not state.evidence:
            return (
                "complete_goal was rejected: the observed evidence ledger is empty. "
                "Emit a Stage that can establish the Goal or block on a real obstacle."
            )
        literal_count = len(state.goal.acceptance_criteria)
        if literal_count and len(decision.evidence) < literal_count:
            return (
                "complete_goal was rejected: provide one evidence statement for each "
                f"of the {literal_count} USER_LITERAL acceptance conditions."
            )
        known_ids = {entry.id for entry in state.evidence}
        if any(
            not any(evidence_id in statement for evidence_id in known_ids)
            for statement in decision.evidence
        ):
            return (
                "complete_goal was rejected: every evidence statement must cite "
                "an exact evidence-ledger ID from the handoff."
            )
    return ""


def _failure_decision(reason: str) -> BlockGoalDecision:
    return BlockGoalDecision(
        reason=reason,
        missing="A valid Stage Planner decision on a later retry.",
        evidence=[],
    )


def _tool_call_to_dict(call: Any) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": normalize_tool_arguments_json(call.function.arguments),
        },
    }


def _safe_json(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


__all__ = ["run_stage_planner"]
