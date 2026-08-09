"""Step manager — plan management, summarization, and termination."""

from __future__ import annotations

import os
from typing import Any, TYPE_CHECKING

from infinidev.engine._best_effort import best_effort
from infinidev.engine.engine_logging import (
    emit_loop_event as _emit_loop_event,
    emit_log as _emit_log,
    log as _log,
    log_finish as _log_finish,
    DIM as _DIM,
    RESET as _RESET,
)
from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.loop.models import ActionRecord, StepResult
from infinidev.engine.loop.behavior_rules import _READ_TOOLS, is_workspace_edit_tool
from infinidev.engine.loop.step_summarizer import _summarize_step, _synthesize_final

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
                elif is_workspace_edit_tool(fn_name) and path:
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

    @staticmethod
    def reconcile_task_completion(
        ctx: ExecutionContext, step_result: StepResult,
    ) -> StepResult:
        """Turn a premature Task close into the documented Step transition.

        ``done`` means the whole Task is complete.  If another planned Step is
        still open, the only valid interpretation is that the current Step is
        complete and execution should continue.  Reconcile that mismatch in
        the state machine instead of spending extra LLM calls asking the model
        to repeat the same tool call with a different enum value.
        """
        if step_result.status != "done":
            return step_result

        plan = getattr(ctx.state, "plan", None)
        if plan is None or not plan.steps:
            return step_result

        active = plan.active_step
        pending = plan.undischarged(
            exclude_index=active.index if active is not None else None,
        )
        if not pending:
            return step_result

        titles = ", ".join(f"{step.index}. {step.title}" for step in pending)
        _emit_log(
            "info",
            "↪ Task completion advanced to the next planned Step; still open: "
            f"{titles}",
            project_id=ctx.project_id,
            agent_id=ctx.agent_id,
        )
        return step_result.model_copy(
            update={
                "status": "continue",
                "final_answer": None,
                "summary": (
                    f"{step_result.summary.rstrip()} "
                    f"Continuing because planned work remains: {titles}."
                ).strip(),
            },
        )

    def advance_plan(self, ctx: ExecutionContext, step_result: StepResult) -> None:
        """Create or update plan from step_result, activate next step.

        Task-level completion is reconciled before this method. Both an
        ordinary model-authored ``continue`` and a reconciled premature
        ``done`` close the active Step; engine-forced interruptions do not
        call this method.
        """
        if not ctx.state.plan.steps:
            if step_result.next_steps:
                ctx.state.plan.apply_operations(step_result.next_steps)
            if ctx.state.plan.steps:
                for s in ctx.state.plan.steps:
                    if s.status == "pending":
                        s.status = "active"
                        break
        else:
            # A step the model gave up on is recorded as given up on. Filing it
            # as ``done`` told run_report, and through it the reviewer, that the
            # work had succeeded.
            ctx.state.plan.mark_active(
                "blocked" if step_result.status == "blocked" else "done"
            )
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
        with best_effort("on_step_start hook dispatch failed"):
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

    def summarize_and_record(
        self, ctx: ExecutionContext, step_result: StepResult,
        messages: list[dict[str, Any]], action_tool_calls: int,
        iteration: int,
    ) -> None:
        """Run summarizer, build ActionRecord, append to history, preload files."""
        step_index = ctx.state.plan.active_step.index if ctx.state.plan.active_step else iteration + 1
        # An ``explore`` step did not advance the plan, so the newest closed
        # step is the *previous* one and would misfile this record. ``blocked``
        # counts as closed here: the step is over, and its archived context is
        # worth as much as a successful step's — more, since the next attempt
        # needs to know what failed.
        if step_result.status != "explore" and not step_result.interrupted:
            closed = [
                s for s in ctx.state.plan.steps if s.status in ("done", "blocked")
            ]
            if closed:
                step_index = closed[-1].index

        _summarizer_on = (
            self._engine._summarizer_override
            if self._engine._summarizer_override is not None
            else _get_settings().LOOP_SUMMARIZER_ENABLED
        )
        if _summarizer_on and not step_result.interrupted:
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

        # Archive everything that is about to leave the model's context.
        # The prompt is rebuilt from summaries only, so without this the
        # model-visible tool output would be unrecoverable; private full command
        # output is never copied here and remains behind its validated handle.
        titles = self._archive_evicted_context(ctx, step_index, messages, record.summary)
        # Written after archiving, not before: the evidence labels only exist
        # once the rows do, and a label pointing at a row that failed to store
        # would be a query returning nothing.
        self._record_outcome(ctx, step_index, record.summary, titles)

        # The user's end-of-step hook, if any. Runs after the summariser so
        # it can be handed the summary the step actually produced, and its
        # output lands on the record rather than in it — see
        # ActionRecord.hook_notes for why that distinction matters.
        record.hook_notes = self._step_end_summary_hook(ctx, step_index, record.summary)

        # Closure notes are independently opt-in and run only after the existing
        # archive → outcome → hook sequence. Failures cannot rewrite the summary,
        # reorder hooks, or disturb legacy archiving. Their path-free descriptor
        # view is the only command-output state carried into the rebuilt prompt.
        handle_view = self._record_command_output_notes(
            ctx, step_index, record.summary
        )
        if handle_view:
            record.discovered_context = "\n".join(
                part for part in (record.discovered_context, handle_view) if part
            )

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
                with best_effort("preload file read failed"):
                    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                        ctx.state.cache_file(fpath, f.read())

        ctx.state.current_step_index = step_index

        # Keep _last_state up-to-date for live introspection (e.g. /debug panel)
        self._engine._last_state = ctx.state

    @staticmethod
    def _step_end_summary_hook(
        ctx: ExecutionContext, step_index: int, summary: str,
    ) -> str:
        """Run the user's ``step_end_summary`` hook for the step just closed.

        Returns the empty string for every uneventful case — no hook, no
        output, a failure — so the caller can assign the result
        unconditionally.

        The step index is passed in rather than read from the plan: by the
        time this runs the plan has already advanced, so ``active_step`` is
        the *next* step and would mislabel every payload by one.
        """
        from infinidev.engine.user_hooks import (
            UserHookEvent, run_hooks, step_payload,
        )

        output = None
        with best_effort("step_end_summary hook failed"):
            payload = step_payload(ctx)
            payload["step_index"] = step_index
            payload["summary"] = summary
            output = run_hooks(
                UserHookEvent.STEP_END_SUMMARY,
                payload,
                workspace_path=getattr(ctx, "workspace_path", None),
            )
        return output.text.strip() if output else ""

    @staticmethod
    def _archive_evicted_context(
        ctx: ExecutionContext, step_index: int,
        messages: list[dict[str, Any]], summary: str,
    ) -> list[str]:
        """Persist the step's raw exchanges into searchable working memory.

        Returns the titles the archive filed them under. Each is a working
        query for ``recall_context``, which is what lets the plan block point
        back at the evidence instead of merely asserting a step happened.

        Best-effort by design: the archive is an aid, and a storage hiccup
        must never fail a step that already did its work — hence the empty
        list on every failure path rather than a raise.
        """
        # Drained before the settings check on purpose: the queue is filled
        # on every tool call, so returning early without emptying it would
        # accumulate every result of the whole task in memory.
        pending, ctx.state.pending_archive = list(ctx.state.pending_archive), []
        if not _get_settings().WORKING_MEMORY_ENABLED:
            return []
        titles: list[str] = []
        with best_effort("working-memory archive failed"):
            from infinidev.engine.working_memory import get_working_memory

            memory = get_working_memory(ctx.session_id)
            titles = memory.archive_step(step_index, messages, summary)
            # The transcript is one source; the bodies captured as the tools
            # returned them are the other. Neither is redundant: the first
            # carries the step summary, the second survives manual mode and
            # small-model compaction. Content-hash dedup keeps the overlap
            # from being stored twice.
            titles += memory.archive_calls(step_index, pending)
            if titles:
                _log(
                    f"   {_DIM}🗄  Archived {len(titles)} excerpt(s) "
                    f"— recall with recall_context{_RESET}"
                )
        return titles

    @staticmethod
    def _record_command_output_notes(
        ctx: ExecutionContext, step_index: int, summary: str,
    ) -> str:
        """Persist short traceable notes for command-output handles, if enabled.

        Descriptors are drained even when disabled so a later settings change
        cannot turn handles from earlier steps into notes with the wrong step
        identity. The note contains no command, output, path, or secret-bearing
        metadata — only the closed-step summary and an opaque artifact identity.
        """
        handles = list(getattr(ctx, "pending_command_output_handles", ()) or ())
        if hasattr(ctx, "pending_command_output_handles"):
            ctx.pending_command_output_handles = []
        settings = _get_settings()
        if not handles or not settings.COMMAND_OUTPUT_AUTO_NOTES_ENABLED:
            return ""

        rendered: list[str] = []
        with best_effort("command-output closure note failed"):
            from infinidev.engine.working_memory import (
                create_traceable_note,
                get_working_memory,
            )

            memory = get_working_memory(ctx.session_id)
            stored = []
            note_summary = summary.strip() or f"Completed step {step_index}"
            for handle in handles:
                artifact_id = handle.get("artifact_id")
                tool_call_id = handle.get("tool_call_id")
                stream = handle.get("stream")
                if (
                    type(artifact_id) is not int
                    or artifact_id <= 0
                    or not isinstance(tool_call_id, str)
                    or not tool_call_id
                    or stream not in ("stdout", "stderr")
                    or handle.get("type") != "command_output"
                ):
                    continue
                note = create_traceable_note(
                    "auto_note",
                    note_summary,
                    source_artifact_id=artifact_id,
                    step_index=step_index,
                    tool_call_id=tool_call_id,
                    occurrence_id=f"command-output:{artifact_id}",
                )
                if memory.remember_traceable(note):
                    stored.append(note)
                    rendered.append(
                        "Command output: "
                        f"artifact_id={artifact_id}, type=command_output, "
                        f"stream={stream}, char_count={handle['char_count']}, "
                        f"byte_count={handle['byte_count']}"
                    )

            if (
                settings.COMMAND_OUTPUT_NOTE_COMPACTION_ENABLED
                and len(stored) > 1
            ):
                memory.compact_traceable_notes(
                    stored, note_summary, step_index=step_index,
                )
        return "\n".join(rendered)

    @staticmethod
    def _record_outcome(
        ctx: ExecutionContext, step_index: int, summary: str, titles: list[str],
    ) -> None:
        """Write what the step established, and how to get the evidence back.

        Two renderings of a finished step used to sit in the prompt saying
        almost the same thing: its line in ``<plan>`` and its collapsed line in
        ``<previous-actions>``. This makes the first one carry something the
        second cannot — the archive labels — so the model can go from "step 2
        established X" to the raw tool output behind X in one call.

        The step-summary record is filtered out of the evidence: recalling it
        returns the sentence already on the line, which is the one thing the
        model does not need to ask for.
        """
        step = next(
            (s for s in ctx.state.plan.steps if s.index == step_index), None
        )
        if step is None:
            return
        if summary and not step.conclusion:
            head = summary.strip().split(". ")[0].strip()
            step.conclusion = (head if 0 < len(head) <= 160 else summary.strip()[:160])
        evidence = [t for t in titles if not t.startswith("Summary of step ")]
        if evidence and not step.evidence:
            step.evidence = evidence[:4]

    @staticmethod
    def _record_undischarged(ctx: ExecutionContext) -> None:
        """Note any plan step the run never reached, on every way out.

        Normal completion is reconciled before the plan advances, but a run can
        still end by exhausting its iterations, tripping the loop guard, or
        being cancelled. Recording stranded work here — one place all exits
        pass through — keeps it visible to the reviewer and the next turn.
        """
        plan = getattr(ctx.state, "plan", None)
        if plan is None or not plan.steps:
            return
        stranded = plan.undischarged()
        if not stranded:
            return
        approved = sum(1 for s in stranded if s.user_approved)
        note = (
            f"⚠ Run ended with {len(stranded)} plan step(s) not executed "
            f"({approved} from the approved plan):\n"
            + "\n".join(f"  {s.index}. {s.title}" for s in stranded)
        )
        # Only once: the gate may already have written the same finding when it
        # ran out of attempts, and two copies in the prompt read as two facts.
        if any(n.startswith("⚠ Run ended with") for n in ctx.state.notes):
            return
        with best_effort("undischarged-step note append failed"):
            ctx.state.notes.append(note)

    def finish(
        self, ctx: ExecutionContext, status: str,
        iteration: int, result: str | None = None,
    ) -> str:
        """Common finish logic: deactivate tracker, log, emit events, store stats."""
        self._record_undischarged(ctx)
        ctx.file_tracker.deactivate()
        # Retain the terminal status + state so the pipeline can build the
        # hidden end-of-task work summary after execute() returns.
        self._engine._last_state = ctx.state
        self._engine._last_status = status
        if ctx.verbose:
            _log_finish(ctx.agent_name, status, iteration + 1, ctx.state.total_tool_calls, ctx.state.total_tokens)
            _log_cache_summary(ctx.state)
            # Static-analysis latency block — printed only when the
            # accumulator was enabled for this run via the
            # INFINIDEV_ENABLE_SA_TIMER env var. The accumulator is
            # off by default and the print path short-circuits when
            # it's off, so a normal user run stays clean and pays
            # zero overhead.
            with best_effort("static-analysis timer render failed"):
                from infinidev.engine.static_analysis_timer import (
                    is_enabled as _sa_enabled,
                    render as _sa_render,
                )
                if _sa_enabled():
                    from infinidev.engine.engine_logging import log as _log
                    _log(_sa_render())
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
        with best_effort("ContextRank session finish failed"):
            self._engine._cr_hooks.finish()
        if result is None:
            return _synthesize_final(ctx.state, status)
        return result
