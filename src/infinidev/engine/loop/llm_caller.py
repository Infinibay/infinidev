"""LLM calling with manual-TC / FC-mode branching and retry."""

from __future__ import annotations

import json
import time
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
from infinidev.engine.formats.tool_call_parser import (
    ManualToolCall as _ManualToolCall,
    parse_text_tool_calls as _parse_text_tool_calls,
)

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext


from infinidev.engine.loop.llm_call_result import LLMCallResult
from infinidev.engine.loop.classified_calls import ClassifiedCalls


import re as _re

# Matches the tool-call text fragments that FC-fallback may rescue.
# When we recover a tool call from message.content, we scrub these
# fragments so the TUI/chat layer doesn't render the raw JSON/XML.
# Order matters — strip wrapper tags first, then bare JSON blobs.
_STRIP_TAG_RES = [
    _re.compile(r"<tool_call>.*?</tool_call>", _re.DOTALL),
    _re.compile(r"<\|tool_call\|>.*?<\|/tool_call\|>", _re.DOTALL),
    _re.compile(r"<function_call>.*?</function_call>", _re.DOTALL),
    _re.compile(r"<function=[a-zA-Z_]\w*>.*?</function>", _re.DOTALL),
]
# Bare OpenAI-style JSON blob emitted inside content by some Qwen
# templates: `{"tool_calls": [{"name": "...", ...}]}` — match a
# balanced outer object via lazy-but-greedy fallback.
_STRIP_JSON_RES = [
    _re.compile(r'\{\s*"tool_calls"\s*:\s*\[.*', _re.DOTALL),
    _re.compile(r'\{\s*"name"\s*:\s*"[a-zA-Z_]\w*"\s*,\s*"arguments"\s*:.*', _re.DOTALL),
]


def _strip_tool_call_markup(text: str) -> str:
    """Remove tool-call XML/JSON fragments so content can be shown safely."""
    for r in _STRIP_TAG_RES:
        text = r.sub("", text)
    for r in _STRIP_JSON_RES:
        text = r.sub("", text)
    return text.rstrip()


_THINK_BLOCK_RE = _re.compile(
    r"<(?:think|thinking|thoughts)>(.*?)</(?:think|thinking|thoughts)>",
    _re.DOTALL | _re.IGNORECASE,
)


def strip_think_blocks(text: str) -> str:
    """Remove any <think>...</think> blocks from free-form text.

    Useful for the streaming path where content is assembled from
    deltas and no message-like object is available to mutate.
    """
    if not text:
        return text
    return _THINK_BLOCK_RE.sub("", text).strip()


class ThinkStreamFilter:
    """Stateful streaming filter that suppresses content inside
    ``<think>...</think>`` blocks as deltas arrive.

    The chat agent emits every content delta to the TUI live via
    ``hooks.notify_stream_chunk``. If the model opens a think block
    mid-stream, we must not forward those characters until the block
    closes — otherwise the TUI renders ``<think>`` as a chat bubble.

    The filter is a small state machine that:
    - buffers up to ``len("</think>")`` characters at the tail to
      catch an opening/closing tag split across chunk boundaries,
    - emits a chunk only once it is safely outside any think block,
    - discards characters that fall inside an open block.

    Callers feed one ``delta`` at a time and emit the return value to
    the user. At end-of-stream, call ``flush()`` to drain any
    held-back tail.
    """

    OPEN = "<think>"
    CLOSE = "</think>"
    _MAX_HOLD = max(len(OPEN), len(CLOSE))

    __slots__ = ("_inside", "_pending")

    def __init__(self) -> None:
        self._inside = False
        self._pending = ""

    def feed(self, delta: str) -> str:
        if not delta:
            return ""
        self._pending += delta
        emit_parts: list[str] = []
        while self._pending:
            if self._inside:
                idx = self._pending.find(self.CLOSE)
                if idx < 0:
                    # Still inside, nothing to emit yet. Keep only the
                    # last few chars in case CLOSE spans chunks.
                    if len(self._pending) > self._MAX_HOLD:
                        self._pending = self._pending[-self._MAX_HOLD:]
                    break
                # Close found — discard up to and including it.
                self._pending = self._pending[idx + len(self.CLOSE):]
                self._inside = False
                continue

            idx = self._pending.find(self.OPEN)
            if idx >= 0:
                if idx > 0:
                    emit_parts.append(self._pending[:idx])
                self._pending = self._pending[idx + len(self.OPEN):]
                self._inside = True
                continue

            # No OPEN seen yet, but a partial OPEN could still be
            # forming at the tail. Hold back that much.
            hold = 0
            for i in range(1, min(len(self.OPEN), len(self._pending) + 1)):
                if self._pending.endswith(self.OPEN[:i]):
                    hold = i
            if hold == 0:
                emit_parts.append(self._pending)
                self._pending = ""
            else:
                emit_parts.append(self._pending[:-hold])
                self._pending = self._pending[-hold:]
            break
        return "".join(emit_parts)

    def flush(self) -> str:
        """End-of-stream: drain any held-back tail.

        If we are still inside an unclosed think block, the tail is
        part of the block and must be dropped.
        """
        if self._inside:
            self._pending = ""
            self._inside = False
            return ""
        tail = self._pending
        self._pending = ""
        return tail


def promote_embedded_think(message: Any) -> None:
    """Move <think>...</think> from content → reasoning_content.

    MiniMax without reasoning_split, DeepSeek-compat backends in "none"
    reasoning mode, and a few other providers send the think block
    embedded in message.content. The TUI would then render those tags
    as a chat bubble. Extract them always when seen in content; if
    reasoning_content was already populated, merge the stray blocks
    into it so no think content is lost.
    """
    existing = (getattr(message, "reasoning_content", None) or "").strip()
    content = getattr(message, "content", None) or ""
    if not content or not _THINK_BLOCK_RE.search(content):
        return
    blocks = _THINK_BLOCK_RE.findall(content)
    if not blocks:
        return
    cleaned = _THINK_BLOCK_RE.sub("", content).strip()
    promoted = "\n\n".join(b.strip() for b in blocks if b.strip())
    try:
        if existing:
            message.reasoning_content = f"{existing}\n\n{promoted}"
        else:
            message.reasoning_content = promoted
        message.content = cleaned
    except Exception:
        pass


class LLMCaller:
    """Encapsulates LLM calling with manual-TC / FC-mode branching and retry."""

    def __init__(
        self,
        on_thinking_chunk: "Callable[[str], None] | None" = None,
        on_stream_status: "Callable[[str, int, str | None], None] | None" = None,
    ) -> None:
        self._malformed_retries = 0
        self._MAX_MALFORMED_RETRIES = 4
        self._on_thinking_chunk = on_thinking_chunk
        self._on_stream_status = on_stream_status

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
            result = self._call_manual(ctx, messages, action_tool_calls)
        else:
            result = self._call_fc(ctx, messages, is_planning, action_tool_calls)
        self._dispatch_post_model_message(ctx, result, messages)
        return result

    @staticmethod
    def _dispatch_post_model_message(
        ctx: "ExecutionContext",
        result: LLMCallResult,
        messages: list[dict[str, Any]],
    ) -> None:
        """Fire POST_MODEL_MESSAGE hook so behavior checkers can score the response."""
        # Skip retry/forced placeholders — nothing to score yet
        if result.should_retry or result.forced_step_result is not None:
            return
        if result.message is None and not result.tool_calls and not result.raw_content:
            return
        try:
            from infinidev.engine.hooks.hooks import hook_manager, HookContext, HookEvent
            if not hook_manager.has_hooks_for(HookEvent.POST_MODEL_MESSAGE):
                return
            # Extract original task (first user message after the system prompt)
            task_text = ""
            for m in messages:
                if m.get("role") == "user":
                    content = m.get("content", "")
                    if isinstance(content, str):
                        task_text = content[:1500]
                    break

            # Snapshot of the current plan, if any
            plan_snapshot: dict[str, Any] = {}
            try:
                plan = getattr(getattr(ctx, "state", None), "plan", None)
                if plan is not None and getattr(plan, "steps", None):
                    active = getattr(plan, "active_step", None)
                    plan_snapshot = {
                        "active_step_index": getattr(active, "index", None),
                        "active_step_title": getattr(active, "title", None),
                        "steps": [
                            {
                                "index": s.index,
                                "title": s.title,
                                "explanation": (s.explanation or "")[:160],
                                "status": s.status,
                            }
                            for s in plan.steps
                        ],
                    }
            except Exception:
                plan_snapshot = {}

            hook_manager.dispatch(HookContext(
                event=HookEvent.POST_MODEL_MESSAGE,
                project_id=ctx.project_id,
                agent_id=ctx.agent_id,
                metadata={
                    "raw_content": result.raw_content,
                    "reasoning_content": result.reasoning_content,
                    "tool_calls": result.tool_calls,
                    "messages": messages,
                    "agent_name": getattr(ctx.agent, "role", ctx.agent_id),
                    "task": task_text,
                    "plan_snapshot": plan_snapshot,
                },
            ))
        except Exception:
            # Never break the loop if scoring fails
            import logging
            logging.getLogger(__name__).debug(
                "POST_MODEL_MESSAGE dispatch failed", exc_info=True,
            )

    # ── Manual TC mode ──────────────────────────────────────────────

    def _call_manual(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        action_tool_calls: int,
    ) -> LLMCallResult:
        _MANUAL_PARSE_RETRIES = 3
        response = None
        for attempt in range(1, _MANUAL_PARSE_RETRIES + 1):
            try:
                response = _call_llm(ctx.llm_params, messages,
                                     on_thinking_chunk=self._on_thinking_chunk,
                                     on_stream_status=self._on_stream_status)
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
                    # Tight exponential backoff: 0.3s, 0.6s, 1.2s — Ollama
                    # parse errors are usually transient and a 1s+ wall-clock
                    # waste per retry was a noticeable chunk of step latency.
                    time.sleep(0.3 * (2 ** (attempt - 1)))
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
            response = _call_llm(ctx.llm_params, messages, iter_tools, tool_choice="required",
                                 on_thinking_chunk=self._on_thinking_chunk,
                                 on_stream_status=self._on_stream_status)
        except Exception as exc:
            return self._handle_fc_error(ctx, exc, messages)

        self._malformed_retries = 0
        self._track_usage(ctx, response)
        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, "tool_calls", None)

        # Some FC-capable providers occasionally return tool calls as
        # text (XML tags / raw JSON) inside ``content`` or ``reasoning_content``
        # instead of populating the ``tool_calls`` API field. This
        # happens with qwen/glm family on LiteLLM when the router picks
        # a non-FC model mid-session. Recover by parsing text as manual
        # tool calls — the result is indistinguishable downstream.
        if not tool_calls:
            tool_calls = self._fc_fallback_parse_text(message, action_tool_calls)

        # Defense-in-depth: some providers (MiniMax without reasoning_split,
        # or any backend that doesn't forward a "extract reasoning" flag)
        # return <think>...</think> embedded in content instead of in the
        # reasoning_content field. The TUI would then render the think
        # block as a chat bubble. Detect + split here so reasoning ends up
        # where it belongs regardless of server-side config.
        promote_embedded_think(message)

        raw = (getattr(message, "content", None) or "").strip()
        return LLMCallResult(
            tool_calls=tool_calls, message=message, raw_content=raw,
            reasoning_content=(getattr(message, "reasoning_content", None) or "").strip(),
        )

    def _fc_fallback_parse_text(
        self, message: Any, action_tool_calls: int,
    ) -> list[Any] | None:
        """Parse text content as tool calls when FC mode returned none.

        See the caller for the "why". Returns ``None`` if there's no
        parseable content at all (caller will treat as a text-only
        response and hit the guardrail path).
        """
        raw_content = (getattr(message, "content", None) or "").strip()
        reasoning_content = (getattr(message, "reasoning_content", None) or "").strip()
        if not raw_content and not reasoning_content:
            return None

        # Try content first, then reasoning (models like Qwen3.6 sometimes
        # emit the tool-call XML inside the <think> block, which llama-server
        # extracts into reasoning_content), then the concatenation as a
        # last resort. Mirrors the chain used by the manual-mode caller.
        parsed = _parse_text_tool_calls(raw_content) if raw_content else None
        parsed_src = "content" if parsed else None
        if not parsed and reasoning_content:
            parsed = _parse_text_tool_calls(reasoning_content)
            parsed_src = "reasoning" if parsed else None
        if not parsed and raw_content and reasoning_content:
            parsed = _parse_text_tool_calls(reasoning_content + "\n" + raw_content)
            parsed_src = "both" if parsed else None
        if not parsed:
            return None

        # Strip the tool-call text from content so the TUI/chat layer
        # doesn't display the raw JSON/XML as a message bubble. We only
        # touch message.content (not reasoning_content) because
        # reasoning is already routed to the thinking panel, not chat.
        if parsed_src in ("content", "both") and getattr(message, "content", None):
            try:
                message.content = _strip_tool_call_markup(message.content)
            except Exception:
                pass

        return [
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
            tools_section = build_tools_prompt_section(ctx.tool_schemas, small_model=ctx.is_small)
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

            # Cache metrics — Anthropic/DashScope/MiniMax format
            cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            ctx.state.cache_creation_tokens += cache_creation
            ctx.state.cache_read_tokens += cache_read

            # Cache metrics — OpenAI/ZAI format (prompt_tokens_details)
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                cached = getattr(details, "cached_tokens", 0) or 0
                ctx.state.cached_tokens += cached

            # Cache metrics — DeepSeek-specific format
            ds_hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
            if ds_hit:
                ctx.state.cached_tokens += ds_hit
