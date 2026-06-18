"""Guardrail validation runner for the loop engine.

Extracted verbatim from ``LoopEngine._apply_guardrail`` so the engine
module stays focused on the loop. Behavior is unchanged: validate the
final result with the guardrail and, on failure, re-prompt the model
(with feedback) to produce a corrected result. A guardrail that raises
is logged loudly (ERROR + traceback) and the *unvalidated* result is
shipped (fail-open) — same as before.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from infinidev.engine.llm_client import call_llm as _call_llm
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

    for attempt in range(max_retries):
        try:
            validation = guardrail(result)
            # CrewAI guardrail convention: returns (success, result_or_feedback)
            if isinstance(validation, tuple):
                success, feedback = validation
                if success:
                    return result
                # Retry with feedback
                logger.info(
                    "Guardrail failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, str(feedback)[:200],
                )
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

                # Run inner loop for retry
                step_text = ""
                action_tool_calls = 0
                while action_tool_calls < max_per_action:
                    response = _call_llm(
                        llm_params, messages,
                        tool_schemas if tool_schemas else None,
                    )
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
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in tc_list
                        ]
                        messages.append(assistant_msg)
                        for tc in tc_list:
                            if tc.function.name == "step_complete":
                                # Parse final answer from step_complete
                                sr = _parse_step_complete_args(tc.function.arguments)
                                step_text = sr.final_answer or sr.summary
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tc.id,
                                    "content": '{"status": "acknowledged"}',
                                })
                                break
                            _pre_content_g = _capture_pre_content(
                                tc.function.name, tc.function.arguments, ctx.file_tracker,
                            )
                            tc_result = execute_tool_call(
                                tool_dispatch,
                                tc.function.name,
                                tc.function.arguments,
                            )
                            _maybe_emit_file_change(
                                tc.function.name, tc.function.arguments, tc_result,
                                _pre_content_g, ctx.file_tracker,
                                ctx.project_id, ctx.agent_id, hooks,
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": tc_result,
                            })
                            action_tool_calls += 1
                        if step_text:
                            break
                    else:
                        step_text = msg.content or ""
                        break

                result = step_text or result
            else:
                # Simple bool guardrail
                if validation:
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
