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
from infinidev.engine.loop.behavior_rules import _EDIT_TOOLS, _READ_TOOLS
from infinidev.engine.loop.step_summarizer import _synthesize_final

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext


def _auto_enhance_record(record: ActionRecord, messages: list[dict]) -> ActionRecord:
    """Extract key facts from tool calls for small models that produce poor summaries.

    Scans the step's messages for read/write tool calls and auto-populates
    the ActionRecord's discovered_context and changes_made fields.
    """
    import json as _json
    files_read: list[str] = []
    files_changed: list[str] = []
    errors: list[str] = []

    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                args_str = fn.get("arguments", "{}")
                try:
                    args = _json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
                except (_json.JSONDecodeError, TypeError):
                    args = {}
                path = args.get("file_path", args.get("path", ""))
                if fn_name in _READ_TOOLS and path:
                    if path not in files_read:
                        files_read.append(path)
                elif fn_name in _EDIT_TOOLS and path:
                    if path not in files_changed:
                        files_changed.append(path)
        elif msg.get("role") == "tool":
            content = msg.get("content", "")
            if '"error"' in content and len(errors) < 3:
                # Extract first 80 chars of error
                errors.append(content[:80])

    if files_read and not record.discovered_context:
        record.discovered_context = f"Read: {', '.join(files_read[:5])}"
    if files_changed and not record.changes_made:
        record.changes_made = f"Modified: {', '.join(files_changed[:5])}"
    if errors and not record.pending_items:
        record.pending_items = f"Errors: {len(errors)} tool failures"

    return record


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
        # Notify a UI hook (if any) that a new step is now active. Best
        # effort — never let a hook error interrupt the engine loop.
        self._emit_step_start(ctx)

    def _emit_step_start(self, ctx: ExecutionContext) -> None:
        hooks = getattr(self._engine, "_hooks", None)
        if hooks is None:
            return
        cb = getattr(hooks, "on_step_start", None)
        if not callable(cb):
            return
        try:
            steps = list(ctx.state.plan.steps)
            active = ctx.state.plan.active_step
            if active is None:
                return
            all_steps = [
                {
                    "index": s.index,
                    "title": s.title,
                    "status": s.status,
                    "user_approved": s.user_approved,
                }
                for s in steps
            ]
            completed = [s.index for s in steps if s.status == "done"]
            cb(active.index, len(steps), all_steps, completed)
        except Exception:
            pass

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
                from infinidev.engine.static_analysis_timer import measure as _sa_measure
                with _sa_measure("summarizer_llm"):
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

        # For small models: auto-enhance record with extracted facts
        if ctx.is_small:
            record = _auto_enhance_record(record, messages)

        # Merge behavior tracker data if available
        bt = step_result.behavior_tracker
        if bt:
            bsum = bt.summary()
            record.behavior_score = bsum["behavior_score"]
            record.behavior_good = bsum["good_patterns"]
            record.behavior_bad = bsum["bad_patterns"]

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
            # Static-analysis latency block — printed only when the
            # accumulator was enabled for this run via the
            # INFINIDEV_ENABLE_SA_TIMER env var. The accumulator is
            # off by default and the print path short-circuits when
            # it's off, so a normal user run stays clean and pays
            # zero overhead.
            try:
                from infinidev.engine.static_analysis_timer import (
                    is_enabled as _sa_enabled,
                    render as _sa_render,
                )
                if _sa_enabled():
                    from infinidev.engine.engine_logging import log as _log
                    _log(_sa_render())
            except Exception:
                pass
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
        # ContextRank: snapshot session scores for cross-session ranking
        try:
            self._engine._cr_hooks.finish()
        except Exception:
            pass
        if result is None:
            return _synthesize_final(ctx.state)
        return result
