"""Step manager — plan management, summarization, and termination."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from infinidev.engine.engine_logging import (
    emit_loop_event as _emit_loop_event,
    emit_log as _emit_log,
    log as _log,
    log_finish as _log_finish,
    DIM as _DIM,
    YELLOW as _YELLOW,
    RESET as _RESET,
)
from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.loop.models import ActionRecord, StepResult

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext


def _get_settings():
    """Lazy import to avoid circular import at module load time."""
    from infinidev.config.settings import settings
    return settings


def _log_cache_summary(state: Any) -> None:
    """Log a one-line cache summary if any cache metrics are non-zero."""
    cache_read = state.cache_read_tokens
    cache_write = state.cache_creation_tokens
    cached_prefix = state.cached_tokens

    if not (cache_read or cache_write or cached_prefix):
        return

    parts: list[str] = []
    if cache_read:
        parts.append(f"{cache_read:,} read from cache")
    if cache_write:
        parts.append(f"{cache_write:,} written to cache")
    if cached_prefix:
        parts.append(f"{cached_prefix:,} prefix-cached")

    _log(f"   {_DIM}💾 Cache: {' · '.join(parts)}{_RESET}")


class StepManager:
    """Post-step processing: plan management, summarization, termination."""

    def __init__(self, engine: "LoopEngine") -> None:
        self._engine = engine

    def auto_split(self, ctx: ExecutionContext, step_result: StepResult) -> StepResult:
        """Prevent premature 'done' when plan steps are still pending."""
        if step_result.status == "done" and not step_result.final_answer:
            pending = sum(1 for s in ctx.state.plan.steps if s.status == "pending")
            if pending > 0:
                step_result.status = "continue"
                _emit_log(
                    "warning",
                    f"{_YELLOW}⚠ Override: status='done' but {pending} steps pending → continue{_RESET}",
                    project_id=ctx.project_id, agent_id=ctx.agent_id,
                )
        return step_result

    def advance_plan(self, ctx: ExecutionContext, step_result: StepResult) -> None:
        """Create or update plan from step_result, activate next step."""
        if not ctx.state.plan.steps:
            if step_result.next_steps:
                ctx.state.plan.apply_operations(step_result.next_steps)
            if ctx.state.plan.steps:
                for s in ctx.state.plan.steps:
                    if s.status == "pending":
                        s.status = "active"
                        break
        else:
            ctx.state.plan.mark_active_done()
            if step_result.next_steps:
                ctx.state.plan.apply_operations(step_result.next_steps)
            ctx.state.plan.activate_next()

    def summarize_and_record(
        self, ctx: ExecutionContext, step_result: StepResult,
        messages: list[dict[str, Any]], action_tool_calls: int,
        iteration: int,
    ) -> None:
        """Run summarizer, build ActionRecord, append to history, preload files."""
        step_index = ctx.state.plan.active_step.index if ctx.state.plan.active_step else iteration + 1
        done_steps = [s for s in ctx.state.plan.steps if s.status == "done"]
        if done_steps:
            step_index = done_steps[-1].index

        _summarizer_on = (
            self._engine._summarizer_override
            if self._engine._summarizer_override is not None
            else _get_settings().LOOP_SUMMARIZER_ENABLED
        )
        if _summarizer_on:
            try:
                structured = _summarize_step(messages, ctx.desc, ctx.state, step_result, ctx.llm_params)
                record = ActionRecord(
                    step_index=step_index,
                    summary=structured.get("summary", step_result.summary),
                    tool_calls_count=action_tool_calls,
                    files_to_preload=structured.get("files_to_preload", []),
                    changes_made=structured.get("changes_made", ""),
                    discovered_context=structured.get("discovered", ""),
                    pending_items=structured.get("pending", ""),
                    anti_patterns=structured.get("anti_patterns", ""),
                )
            except Exception:
                record = ActionRecord(step_index=step_index, summary=step_result.summary, tool_calls_count=action_tool_calls)
        else:
            record = ActionRecord(step_index=step_index, summary=step_result.summary, tool_calls_count=action_tool_calls)

        ctx.state.history.append(record)

        # Pre-load files recommended by summarizer
        for fpath in record.files_to_preload:
            if fpath not in ctx.state.opened_files and os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        ctx.state.cache_file(fpath, f.read())
                except Exception:
                    pass

        ctx.state.current_step_index = step_index

        # Keep _last_state up-to-date for live introspection (e.g. /debug panel)
        self._engine._last_state = ctx.state

    def finish(
        self, ctx: ExecutionContext, status: str,
        iteration: int, result: str | None = None,
    ) -> str:
        """Common finish logic: deactivate tracker, log, emit events, store stats."""
        ctx.file_tracker.deactivate()
        if ctx.verbose:
            _log_finish(ctx.agent_name, status, iteration + 1, ctx.state.total_tool_calls, ctx.state.total_tokens)
            _log_cache_summary(ctx.state)
        _emit_loop_event("loop_finished", ctx.project_id, ctx.agent_id, {
            "agent_id": ctx.agent_id, "agent_name": ctx.agent_name,
            "status": status, "iterations": iteration + 1,
            "tool_calls_total": ctx.state.total_tool_calls,
            "tokens_total": ctx.state.total_tokens,
        })
        _hook_manager.dispatch(_HookContext(
            event=_HookEvent.LOOP_END,
            metadata={
                "state": ctx.state, "result": result, "status": status,
                "cache_stats": {
                    "cache_creation_tokens": ctx.state.cache_creation_tokens,
                    "cache_read_tokens": ctx.state.cache_read_tokens,
                    "cached_tokens": ctx.state.cached_tokens,
                },
            },
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        ))
        self._engine._store_stats(ctx.state)
        if result is None:
            return _synthesize_final(ctx.state)
        return result
