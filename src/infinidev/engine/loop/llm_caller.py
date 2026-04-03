"""LLM calling with manual-TC / FC-mode branching and retry."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from infinidev.engine.llm_client import (
    call_llm as _call_llm,
    is_malformed_tool_call as _is_malformed_tool_call,
    PERMANENT_ERRORS as _PERMANENT_ERRORS,
)
from infinidev.engine.engine_logging import (
    emit_log as _emit_log,
    YELLOW as _YELLOW,
    RED as _RED,
    RESET as _RESET,
)
from infinidev.engine.loop.context import build_system_prompt, build_tools_prompt_section
from infinidev.engine.loop.models import StepResult
from infinidev.engine.tool_call_parser import (
    ManualToolCall as _ManualToolCall,
    parse_text_tool_calls as _parse_text_tool_calls,
)

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext


from infinidev.engine.loop.llm_call_result import LLMCallResult
from infinidev.engine.loop.classified_calls import ClassifiedCalls

class LLMCaller:
    """Encapsulates LLM calling with manual-TC / FC-mode branching and retry."""

    def __init__(self) -> None:
        self._malformed_retries = 0
        self._MAX_MALFORMED_RETRIES = 4

    def reset(self) -> None:
        """Reset per-inner-loop counters."""
        self._malformed_retries = 0

    def call(
        self,
        ctx: ExecutionContext,
        messages: list[dict[str, Any]],
        is_planning: bool,
        action_tool_calls: int = 0,
    ) -> LLMCallResult:
        """Make one LLM call and return a parsed result.

        Handles manual-TC vs FC mode, parse-error retries, malformed
        tool-call retries, and FC→manual fallback on permanent errors.
        """
        if ctx.manual_tc:
            return self._call_manual(ctx, messages, action_tool_calls)
        return self._call_fc(ctx, messages, is_planning, action_tool_calls)

    # ── Manual TC mode ──────────────────────────────────────────────

    def _call_manual(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        action_tool_calls: int,
    ) -> LLMCallResult:
        _MANUAL_PARSE_RETRIES = 3
        response = None
        for attempt in range(1, _MANUAL_PARSE_RETRIES + 1):
            try:
                response = _call_llm(ctx.llm_params, messages)
                break
            except Exception as exc:
                msg = str(exc).lower()
                is_parse_error = (
                    "failed to parse" in msg or "internal server error" in msg
                )
                if is_parse_error and attempt < _MANUAL_PARSE_RETRIES:
                    _emit_log(
                        "warning",
                        f"{_YELLOW}⚠ Server parse error (attempt "
                        f"{attempt}/{_MANUAL_PARSE_RETRIES}), retrying...{_RESET}",
                        project_id=ctx.project_id, agent_id=ctx.agent_id,
                    )
                    time.sleep(1.0 * attempt)
                    continue
                raise

        self._track_usage(ctx, response)
        choice = response.choices[0]
        message = choice.message
        raw_content = (message.content or "").strip()
        reasoning_content = (getattr(message, "reasoning_content", None) or "").strip()

        # Parse tool calls from text
        parsed = _parse_text_tool_calls(raw_content)
        if not parsed and reasoning_content:
            parsed = _parse_text_tool_calls(reasoning_content)
        if not parsed and raw_content and reasoning_content:
            parsed = _parse_text_tool_calls(reasoning_content + "\n" + raw_content)

        if parsed:
            self._malformed_retries = 0
            tool_calls = [
                _ManualToolCall(
                    id=f"manual_{action_tool_calls + i}",
                    name=pc["name"],
                    arguments=(
                        json.dumps(pc["arguments"])
                        if isinstance(pc["arguments"], dict)
                        else str(pc["arguments"])
                    ),
                )
                for i, pc in enumerate(parsed)
            ]
            return LLMCallResult(
                tool_calls=tool_calls, message=message,
                raw_content=raw_content, reasoning_content=reasoning_content,
            )

        return LLMCallResult(
            message=message, raw_content=raw_content,
            reasoning_content=reasoning_content,
        )

    # ── FC mode ─────────────────────────────────────────────────────

    def _call_fc(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        is_planning: bool, action_tool_calls: int,
    ) -> LLMCallResult:
        iter_tools = ctx.planning_schemas if is_planning else ctx.tool_schemas
        try:
            response = _call_llm(ctx.llm_params, messages, iter_tools, tool_choice="required")
        except Exception as exc:
            return self._handle_fc_error(ctx, exc, messages)

        self._malformed_retries = 0
        self._track_usage(ctx, response)
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        # FC mode fallback: some models return tool calls as tags in content
        if not tool_calls:
            raw_content = (getattr(message, "content", None) or "").strip()
            if not raw_content:
                raw_content = (getattr(message, "reasoning_content", None) or "").strip()
            if raw_content:
                parsed = _parse_text_tool_calls(raw_content)
                if parsed:
                    tool_calls = [
                        _ManualToolCall(
                            id=f"fc_fallback_{action_tool_calls + i}",
                            name=pc["name"],
                            arguments=(
                                json.dumps(pc["arguments"])
                                if isinstance(pc["arguments"], dict)
                                else str(pc["arguments"])
                            ),
                        )
                        for i, pc in enumerate(parsed)
                    ]

        raw = (getattr(message, "content", None) or "").strip()
        return LLMCallResult(
            tool_calls=tool_calls, message=message, raw_content=raw,
            reasoning_content=(getattr(message, "reasoning_content", None) or "").strip(),
        )

    def _handle_fc_error(
        self, ctx: ExecutionContext, exc: Exception, messages: list[dict[str, Any]],
    ) -> LLMCallResult:
        """Handle FC mode exceptions: malformed retries, permanent error fallback."""
        if _is_malformed_tool_call(exc):
            self._malformed_retries += 1
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Malformed tool call from provider "
                f"(attempt {self._malformed_retries}/{self._MAX_MALFORMED_RETRIES}): "
                f"{str(exc)[:120]}{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            if self._malformed_retries < self._MAX_MALFORMED_RETRIES:
                return LLMCallResult(should_retry=True)
            _emit_log(
                "error",
                f"{_RED}⚠ Malformed tool calls persisted — forcing step completion{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            return LLMCallResult(forced_step_result=StepResult(
                summary=(
                    f"Step interrupted: LLM produced malformed tool calls "
                    f"({self._malformed_retries} attempts). Will retry on next step."
                ),
                status="continue",
            ))

        exc_msg = str(exc).lower()
        if any(p in exc_msg for p in _PERMANENT_ERRORS):
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Provider rejected function calling: "
                f"{str(exc)[:120]} — switching to manual tool calling{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            ctx.manual_tc = True
            tools_section = build_tools_prompt_section(ctx.tool_schemas)
            ctx.system_prompt = build_system_prompt(
                ctx.agent.backstory,
                tech_hints=getattr(ctx.agent, '_tech_hints', None),
                session_summaries=getattr(ctx.agent, '_session_summaries', None),
                identity_override=getattr(ctx.agent, '_system_prompt_identity', None),
            )
            ctx.system_prompt = f"{ctx.system_prompt}\n\n{tools_section}"
            messages[0] = {"role": "system", "content": ctx.system_prompt}
            return LLMCallResult(should_retry=True)

        raise exc  # Non-recoverable

    @staticmethod
    def _track_usage(ctx: ExecutionContext, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            ctx.state.total_tokens += getattr(usage, "total_tokens", 0)
            ctx.state.last_prompt_tokens = getattr(usage, "prompt_tokens", 0)
            ctx.state.last_completion_tokens = getattr(usage, "completion_tokens", 0)
