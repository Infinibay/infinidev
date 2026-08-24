"""Guardrail validation runner for the loop engine.

Validation and correction run before LoopEngine emits its terminal event, so
corrective tool calls remain observable and file changes remain tracked. A
guardrail that raises is logged loudly and the unvalidated result is shipped
(fail-open), preserving the public fallback policy.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from infinidev.engine.llm_client import call_llm as _call_llm
from infinidev.engine.formats._normalize import normalize_tool_arguments_json
from infinidev.engine.formats.tool_call_parser import (
    parse_step_complete_args as _parse_step_complete_args,
)
from infinidev.engine.tool_executor import (
    capture_pre_content as _capture_pre_content,
    maybe_emit_file_change as _maybe_emit_file_change,
)
from infinidev.engine.tool_dispatch import execute_tool_call

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext
    from infinidev.engine.loop.models import LoopState

logger = logging.getLogger(__name__)

_MAX_GUARDRAIL_CORRECTION_ROUNDS = 8
_UNBOUNDED_GUARDRAIL_TOOL_FUSE = 16


def _tool_call_allowance(
    ctx: "ExecutionContext",
    state: "LoopState | None",
    max_per_action: int,
    unbounded_calls: int,
) -> int:
    """Return the remaining regular-tool allowance for one correction."""
    allowance = (
        max_per_action
        if max_per_action > 0
        else max(0, _UNBOUNDED_GUARDRAIL_TOOL_FUSE - unbounded_calls)
    )
    total_limit = getattr(ctx, "max_total_calls", None)
    total_used = getattr(state, "total_tool_calls", 0)
    if isinstance(total_limit, int):
        allowance = min(allowance, max(0, total_limit - int(total_used or 0)))
    return allowance


def _record_tool_call(state: "LoopState | None") -> None:
    """Account for a corrective tool call in the owning LoopState."""
    if state is None:
        return
    total = getattr(state, "total_tool_calls", None)
    if isinstance(total, int):
        state.total_tool_calls = total + 1


def _track_response_usage(ctx: "ExecutionContext", response: Any) -> None:
    """Reuse the normal loop accounting for corrective LLM calls."""
    if getattr(response, "usage", None) is None:
        return
    from infinidev.engine.loop.llm_caller import LLMCaller

    LLMCaller._track_usage(ctx, response)


def _prompt_budget_exhausted(ctx: "ExecutionContext") -> bool:
    """Return whether another corrective LLM request would cross the run fuse."""
    limit = getattr(ctx, "max_prompt_tokens", None)
    used = getattr(getattr(ctx, "state", None), "total_prompt_tokens", 0)
    return isinstance(limit, int) and int(used or 0) >= limit


def apply_guardrail(
    ctx: "ExecutionContext",
    result: str,
    guardrail: Any | None,
    max_retries: int,
    llm_params: dict[str, Any],
    system_prompt: str,
    desc: str,
    expected: str,
    state: "LoopState",
    tool_schemas: list[dict[str, Any]],
    tool_dispatch: dict[str, Any],
    max_per_action: int = 0,
    *,
    hooks: Any | None = None,
) -> str:
    """Validate result with guardrail; retry with feedback if it fails."""
    if guardrail is None:
        return result

    correction_limit = max(0, int(max_retries))
    unbounded_tool_calls = 0
    for attempt in range(correction_limit + 1):
        try:
            validation = guardrail(result)
            # Tuple guardrails return (success, result_or_feedback).
            if isinstance(validation, tuple):
                success, feedback = validation
                if success:
                    return result
                if attempt >= correction_limit:
                    return result
                # Retry with feedback
                logger.info(
                    "Guardrail failed (correction %d/%d): %s",
                    attempt + 1, correction_limit, str(feedback)[:200],
                )
                if _prompt_budget_exhausted(ctx):
                    logger.warning(
                        "Guardrail correction skipped: prompt-token budget exhausted"
                    )
                    return result
                feedback_prompt = (
                    f"Your previous output was rejected by validation.\n"
                    f"Feedback: {feedback}\n\n"
                    f"Please fix your output and try again.\n\n"
                    f"Previous output:\n{result}"
                )
                messages: list[dict[str, Any]] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": feedback_prompt},
                ]

                # One correction is a bounded micro-loop even when the main
                # Step has no configured tool-call boundary. The first model
                # call is always allowed because it may return corrected prose
                # or step_complete without using a regular tool.
                step_text = ""
                action_tool_calls = 0
                tool_call_allowance = _tool_call_allowance(
                    ctx,
                    state,
                    max_per_action,
                    unbounded_tool_calls,
                )
                tool_budget_exhausted = False
                for _round in range(_MAX_GUARDRAIL_CORRECTION_ROUNDS):
                    if _prompt_budget_exhausted(ctx):
                        return result
                    response = _call_llm(
                        llm_params,
                        messages,
                        tool_schemas if tool_schemas else None,
                    )
                    _track_response_usage(ctx, response)
                    choice = response.choices[0]
                    msg = choice.message
                    tc_list = getattr(msg, "tool_calls", None)
                    if tc_list:
                        assistant_msg: dict[str, Any] = {
                            "role": "assistant",
                            "content": msg.content or "",
                        }
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": normalize_tool_arguments_json(
                                        tc.function.arguments
                                    ),
                                },
                            }
                            for tc in tc_list
                        ]
                        messages.append(assistant_msg)
                        for tc in tc_list:
                            if tc.function.name == "step_complete":
                                sr = _parse_step_complete_args(
                                    tc.function.arguments
                                )
                                step_text = sr.final_answer or sr.summary
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": '{"status": "acknowledged"}',
                                })
                                break
                            if action_tool_calls >= tool_call_allowance:
                                tool_budget_exhausted = True
                                break
                            _pre_content_g = _capture_pre_content(
                                tc.function.name,
                                tc.function.arguments,
                                ctx.file_tracker,
                            )
                            action_tool_calls += 1
                            unbounded_tool_calls += 1
                            _record_tool_call(state)
                            tc_result = execute_tool_call(
                                tool_dispatch,
                                tc.function.name,
                                tc.function.arguments,
                            )
                            _maybe_emit_file_change(
                                tc.function.name,
                                tc.function.arguments,
                                tc_result,
                                _pre_content_g,
                                ctx.file_tracker,
                                ctx.project_id,
                                ctx.agent_id,
                                hooks,
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tc_result,
                            })
                        if step_text or tool_budget_exhausted:
                            break
                    else:
                        step_text = msg.content or ""
                        break

                result = step_text or result
            else:
                # A boolean guardrail provides no feedback for correction.
                return result
        except Exception as exc:
            # A guardrail is a correctness check; on a crash we fall through
            # to `return result`, shipping UNVALIDATED output (fail-open).
            # Make that loud (ERROR + traceback) so a broken guardrail is
            # never silent.
            logger.error(
                "Guardrail raised exception; result is UNVALIDATED: %s",
                exc, exc_info=True,
            )
            return result

    return result
