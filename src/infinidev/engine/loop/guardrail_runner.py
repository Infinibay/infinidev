"""Guardrail validation runner for the loop engine.

Validation and correction run before LoopEngine emits its terminal event, so
corrective tool calls remain observable and file changes remain tracked. A
guardrail produces a typed outcome so runtime failure cannot become success.
The legacy string adapter still returns the latest text for compatibility.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from copy import copy
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Literal, TYPE_CHECKING

from infinidev.engine.engine_logging import extract_tool_error
from infinidev.engine.llm_client import call_llm as _call_llm
from infinidev.engine.formats._normalize import normalize_tool_arguments_json
from infinidev.engine.formats.tool_call_parser import (
    parse_step_complete_args as _parse_step_complete_args,
)
from infinidev.engine.tool_executor import (
    capture_pre_content as _capture_pre_content,
    maybe_emit_file_change as _maybe_emit_file_change,
    update_opened_files_cache,
)
from infinidev.engine.tool_dispatch import execute_tool_call

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext
    from infinidev.engine.loop.models import LoopState

logger = logging.getLogger(__name__)

_MAX_GUARDRAIL_CORRECTION_ROUNDS = 8
_UNBOUNDED_GUARDRAIL_TOOL_FUSE = 16


@dataclass(frozen=True)
class GuardrailOutcome:
    """Validation decision together with the latest, potentially partial text."""

    text: str
    status: Literal["done", "blocked", "exhausted", "failed", "cancelled"] = "done"
    reason: str = ""

    @classmethod
    def cancelled(cls, text: str) -> GuardrailOutcome:
        """Cancellation always carries a consistent status and explanation."""
        return cls(text, "cancelled", "Task cancelled during final validation.")

    @property
    def user_message(self) -> str:
        """Make incomplete output visibly agree with its terminal status."""
        if self.status == "done":
            return self.text
        return self.reason + (f"\n\nPartial result:\n{self.text}" if self.text else "")


@dataclass(frozen=True)
class _CorrectionResult:
    """A proposed answer or a stopped correction, never validated Task success."""

    text: str
    status: Literal[
        "candidate", "blocked", "exhausted", "cancelled", "not_started",
    ] = "candidate"
    reason: str = ""


class _CorrectionSession:
    """Own model/tool protocol and budgets for one guardrail evaluation."""

    def __init__(
        self,
        ctx: ExecutionContext,
        *,
        hooks: Any | None,
        build_context_messages: Callable[[], list[dict[str, Any]]] | None,
        run_tool: Callable[[str, str], str] | None,
        tool_scope: Callable[[], AbstractContextManager[Event]] | None,
        is_cancelled: Callable[[], bool],
    ) -> None:
        self._ctx = ctx
        self._hooks = hooks
        self._build_context_messages = build_context_messages
        self._run_tool = run_tool
        self._tool_scope = tool_scope or nullcontext
        self._is_cancelled = is_cancelled
        self._unbounded_tool_calls = 0
        self._action_tool_calls = 0
        self._tool_call_allowance = 0

    def run(self, result: str, feedback: Any, *, attempt: int) -> _CorrectionResult:
        """Propose one correction while retaining cumulative run accounting."""
        if self._prompt_budget_exhausted():
            logger.warning("Guardrail correction skipped: prompt-token budget exhausted")
            return _CorrectionResult(
                result, "not_started", "Final correction stopped: prompt-token budget exhausted.",
            )
        messages = self._build_messages(result, feedback)
        self._action_tool_calls = 0
        self._tool_call_allowance = self._remaining_tool_calls()
        for round_index in range(_MAX_GUARDRAIL_CORRECTION_ROUNDS):
            if self._is_cancelled():
                return _CorrectionResult(result, "cancelled")
            if self._prompt_budget_exhausted():
                return _CorrectionResult(
                    result, "exhausted", "Final correction stopped: prompt-token budget exhausted.",
                )
            ctx = self._ctx
            response = _call_llm(
                ctx.llm_params, messages,
                ctx.tool_schemas if ctx.tool_schemas and not ctx.manual_tc else None,
            )
            if getattr(response, "usage", None) is not None:
                from infinidev.engine.loop.llm_caller import LLMCaller

                LLMCaller._track_usage(ctx, response)
            if self._is_cancelled():
                return _CorrectionResult(result, "cancelled")
            from infinidev.engine.loop.llm_caller import normalize_message_text

            message = response.choices[0].message
            content = normalize_message_text(getattr(message, "content", None))
            calls = self._tool_calls(message, content, attempt, round_index)
            if not calls:
                return _CorrectionResult(content or result)
            completed = self._process_batch(messages, message, content, calls, result)
            if completed is not None:
                return completed
        return _CorrectionResult(
            result, "exhausted", "Final correction stopped: model-round limit reached.",
        )

    def _build_messages(self, result: str, feedback: Any) -> list[dict[str, Any]]:
        feedback_prompt = (
            "Your previous output was rejected by validation.\n"
            f"Feedback: {feedback}\n\n"
            "Please fix your output and try again.\n\n"
            f"Previous output:\n{result}"
        )
        if self._build_context_messages is not None:
            messages = list(self._build_context_messages())
        else:
            from infinidev.engine.loop.context import build_iteration_prompt
            from infinidev.engine.loop.loop_state import LoopState

            ctx = self._ctx
            state = ctx.state if isinstance(ctx.state, LoopState) else LoopState()
            context_prompt = build_iteration_prompt(
                ctx.desc, ctx.expected, state,
                task=getattr(ctx, "task", None),
                skip_plan=getattr(ctx, "skip_plan", False),
                small_model=getattr(ctx, "is_small", False),
                context_corpus=getattr(ctx, "context_corpus", None),
                prompt_configuration=getattr(ctx, "prompt_configuration", None),
            )
            messages = [
                {"role": "system", "content": ctx.system_prompt},
                {"role": "user", "content": context_prompt},
            ]
        messages.append({"role": "user", "content": feedback_prompt})
        return messages

    def _tool_calls(
        self, message: Any, content: str, attempt: int, round_index: int,
    ) -> list[Any]:
        calls = getattr(message, "tool_calls", None) or []
        if not self._ctx.manual_tc or calls:
            return calls
        from infinidev.engine.formats.tool_call_parser import ManualToolCall, parse_text_tool_calls

        return [ManualToolCall(
            f"correction-{attempt}-{round_index}-{index}",
            call["name"], normalize_tool_arguments_json(call.get("arguments", {})),
        ) for index, call in enumerate(parse_text_tool_calls(content) or [])]

    def _process_batch(
        self, messages: list[dict[str, Any]], message: Any, content: str,
        calls: list[Any], fallback: str,
    ) -> _CorrectionResult | None:
        from infinidev.engine.loop.classified_calls import ClassifiedCalls
        from infinidev.engine.loop.llm_call_result import LLMCallResult
        from infinidev.engine.loop.tool_runner import ToolRunner

        projected_message = copy(message)
        projected_message.content = content
        ToolRunner.append_assistant_message(
            self._ctx, ClassifiedCalls(regular=list(calls)), messages,
            LLMCallResult(message=projected_message, raw_content=content),
        )
        terminal: _CorrectionResult | None = None
        tool_budget_exhausted = False
        manual_results: list[str] = []

        def append_result(call: Any, body: str) -> None:
            if self._ctx.manual_tc:
                manual_results.append(f"[Tool: {call.function.name}] Result:\n{body}")
            else:
                messages.append({"role": "tool", "tool_call_id": call.id, "content": body})

        for call in calls:
            if self._is_cancelled():
                return _CorrectionResult(fallback, "cancelled")
            if terminal is not None:
                append_result(call, '{"error":"not_run: correction already completed"}')
                continue
            if call.function.name == "step_complete":
                step = _parse_step_complete_args(call.function.arguments)
                text = step.final_answer or step.summary or fallback
                blocker = {
                    "blocked": "Final correction reported a blocker.",
                    "explore": "Final correction requires further exploration.",
                }.get(step.status, "")
                terminal = _CorrectionResult(text, "blocked", blocker) if blocker else (
                    _CorrectionResult(text)
                )
                append_result(call, '{"status":"acknowledged"}')
                continue
            if self._action_tool_calls >= self._tool_call_allowance:
                tool_budget_exhausted = True
                append_result(call, '{"error":"not_run: correction tool budget exhausted"}')
                continue
            append_result(call, self._execute_tool(call))
            if self._is_cancelled():
                return _CorrectionResult(fallback, "cancelled")
        if manual_results:
            messages.append({"role": "user", "content": "\n".join(manual_results)})
        if terminal is not None and terminal.status == "blocked":
            return terminal
        if tool_budget_exhausted:
            return _CorrectionResult(
                terminal.text if terminal is not None else fallback,
                "exhausted", "Final correction stopped: tool-call budget exhausted.",
            )
        return terminal

    def _execute_tool(self, call: Any) -> str:
        ctx = self._ctx
        name = call.function.name
        arguments = normalize_tool_arguments_json(call.function.arguments)
        self._action_tool_calls += 1
        self._unbounded_tool_calls += 1
        state = ctx.state
        total_calls = getattr(state, "total_tool_calls", None)
        if isinstance(total_calls, int):
            state.total_tool_calls = total_calls + 1
        if self._run_tool is None:
            with self._tool_scope() as cancel_event:
                if cancel_event is not None and cancel_event.is_set():
                    return '{"error": "cancelled"}'
                result = execute_tool_call(
                    ctx.tool_dispatch, name, arguments,
                    hook_metadata={
                        "project_id": ctx.project_id, "agent_id": ctx.agent_id,
                        "cancel_event": cancel_event,
                    },
                    file_tracker=ctx.file_tracker,
                    file_change_hooks=self._hooks,
                )
        else:
            tool = ctx.tool_dispatch.get(name)
            before = _capture_pre_content(name, arguments, ctx.file_tracker, tool=tool)
            result = self._run_tool(name, arguments)
            _maybe_emit_file_change(
                name, arguments, result, before, ctx.file_tracker,
                ctx.project_id, ctx.agent_id, self._hooks, tool=tool,
            )
        if getattr(state, "opened_files", None) is not None:
            if not extract_tool_error(result):
                update_opened_files_cache(state, name, arguments, result)
            if name == "execute_command":
                from infinidev.engine.loop.tool_runner import ToolRunner

                ToolRunner._refresh_opened_files_after_shell(ctx, arguments)
        return result

    def _remaining_tool_calls(self) -> int:
        ctx = self._ctx
        allowance = (
            ctx.max_per_action if ctx.max_per_action > 0 else
            max(0, _UNBOUNDED_GUARDRAIL_TOOL_FUSE - self._unbounded_tool_calls)
        )
        total_limit = getattr(ctx, "max_total_calls", None)
        total_used = getattr(ctx.state, "total_tool_calls", 0)
        if isinstance(total_limit, int):
            allowance = min(allowance, max(0, total_limit - int(total_used or 0)))
        return allowance

    def _prompt_budget_exhausted(self) -> bool:
        limit = getattr(self._ctx, "max_prompt_tokens", None)
        used = getattr(self._ctx.state, "total_prompt_tokens", 0)
        return isinstance(limit, int) and int(used or 0) >= limit


def evaluate_guardrail(
    ctx: "ExecutionContext",
    result: str,
    *,
    hooks: Any | None = None,
    build_context_messages: Callable[[], list[dict[str, Any]]] | None = None,
    run_tool: Callable[[str, str], str] | None = None,
    tool_scope: Callable[[], AbstractContextManager[Event]] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> GuardrailOutcome:
    """Decide acceptance independently of the correction's model/tool mechanics."""
    cancelled = is_cancelled or (lambda: False)
    if cancelled():
        return GuardrailOutcome.cancelled(result)
    guardrail = ctx.guardrail
    if guardrail is None:
        return GuardrailOutcome(result)
    correction_limit = max(0, int(ctx.guardrail_max_retries))
    corrections = _CorrectionSession(
        ctx, hooks=hooks, build_context_messages=build_context_messages,
        run_tool=run_tool, tool_scope=tool_scope, is_cancelled=cancelled,
    )
    exhaustion_reason = ""
    for attempt in range(correction_limit + 1):
        try:
            if cancelled():
                return GuardrailOutcome.cancelled(result)
            validation = guardrail(result)
            if cancelled():
                return GuardrailOutcome.cancelled(result)
            if type(validation) is bool:
                return GuardrailOutcome(result) if validation else GuardrailOutcome(
                    result, "blocked", "Final validation rejected the result.",
                )
            if not (
                isinstance(validation, tuple) and len(validation) == 2
                and type(validation[0]) is bool
            ):
                return GuardrailOutcome(
                    result, "failed", "Final validation must return bool or (bool, feedback).",
                )
            accepted, feedback = validation
            if accepted:
                return GuardrailOutcome(result)
            if exhaustion_reason:
                return GuardrailOutcome(result, "exhausted", exhaustion_reason)
            if attempt >= correction_limit:
                return GuardrailOutcome(result, "blocked", f"Final validation rejected: {feedback}")
            logger.info(
                "Guardrail failed (correction %d/%d): %s",
                attempt + 1, correction_limit, str(feedback)[:200],
            )
            corrected = corrections.run(result, feedback, attempt=attempt)
            result = corrected.text
            if corrected.status == "cancelled":
                return GuardrailOutcome.cancelled(result)
            if corrected.status == "blocked":
                return GuardrailOutcome(result, "blocked", corrected.reason)
            if corrected.status == "not_started":
                return GuardrailOutcome(result, "exhausted", corrected.reason)
            exhaustion_reason = corrected.reason if corrected.status == "exhausted" else ""
        except Exception as exc:
            if cancelled():
                return GuardrailOutcome.cancelled(result)
            logger.error(
                "Guardrail raised exception; result is UNVALIDATED: %s", exc, exc_info=True,
            )
            return GuardrailOutcome(result, "failed", f"Final validation failed: {exc}")
    return GuardrailOutcome(result, "blocked", "Final validation did not accept the result.")


def apply_guardrail(
    ctx: ExecutionContext,
    result: str,
    guardrail: Any | None,
    max_retries: int,
    llm_params: dict[str, Any],
    system_prompt: str,
    desc: str,
    expected: str,
    state: LoopState,
    tool_schemas: list[dict[str, Any]],
    tool_dispatch: dict[str, Any],
    max_per_action: int = 0,
    *,
    hooks: Any | None = None,
    build_context_messages: Callable[[], list[dict[str, Any]]] | None = None,
) -> str:
    """Return the latest text with the legacy fallback and argument contract."""
    run_ctx = copy(ctx)
    for name, value in {
        "guardrail": guardrail, "guardrail_max_retries": max_retries,
        "llm_params": llm_params, "system_prompt": system_prompt, "desc": desc,
        "expected": expected, "state": state, "tool_schemas": tool_schemas,
        "tool_dispatch": tool_dispatch, "max_per_action": max_per_action,
        "manual_tc": getattr(ctx, "manual_tc", False),
    }.items():
        setattr(run_ctx, name, value)
    return evaluate_guardrail(
        run_ctx, result, hooks=hooks, build_context_messages=build_context_messages,
    ).text
