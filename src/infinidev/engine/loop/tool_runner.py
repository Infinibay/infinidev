"""Turning a batch of tool calls into conversation the model can read.

Between "the model asked for these tools" and "the model sees what
happened" sits a surprising amount of protocol. Tool calls have to be
batched (writes serially, reads in parallel), results have to be annotated
with a budget counter and behavioural feedback, images have to travel as
their own user turn because most providers reject content blocks inside a
tool message, and the whole assistant→tool block has to stay contiguous
because some providers reject anything wedged between a tool call and its
result.

None of that is loop logic. Collecting it here is what lets the inner loop
read as the decision it actually is.

The engine is held as a collaborator rather than having its state passed in
piecemeal: cancellation, the UI hooks and the ContextRank recorder are all
consulted mid-batch, and threading four attributes through six call sites
would say less than one back-reference does. This mirrors ``StepManager``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine._best_effort import best_effort
from infinidev.engine.engine_logging import extract_tool_error
from infinidev.engine.formats._normalize import normalize_tool_arguments_json
from infinidev.engine.loop.context_manager import ContextManager
from infinidev.engine.loop.execution_context import ExecutionContext
from infinidev.engine.loop.llm_caller import ClassifiedCalls, LLMCallResult
from infinidev.engine.loop.step_manager import _get_settings
from infinidev.engine.loop.tool_processor import ToolProcessor
from infinidev.engine.tool_dispatch import execute_tool_call
from infinidev.engine.tool_executor import (
    WRITE_TOOLS,
    batch_tool_calls,
    capture_pre_content,
    execute_tool_calls_parallel,
    maybe_emit_file_change,
    update_opened_files_cache,
)

logger = logging.getLogger(__name__)

# How many parsed test failures are appended to a test command's output.
# Enough to see the shape of the breakage without pasting a whole suite.
_MAX_STRUCTURED_FAILURES = 8

_CANCELLED_RESULT = '{"error": "cancelled"}'
_BUDGET_EXHAUSTED_RESULT = (
    '{"error": "not_run: tool budget exhausted; inspect completed tool results '
    'and continue in the next step"}'
)


class ToolRunner:
    """Executes a step's tool calls and writes the result into *messages*."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    # ── entry points ─────────────────────────────────────────────────

    def run_regular(
        self,
        ctx: ExecutionContext,
        classified: ClassifiedCalls,
        messages: list[dict[str, Any]],
        llm_result: LLMCallResult,
        action_tool_calls: int,
        iteration: int,
        guard: Any,
        tracker: Any,
    ) -> int:
        """Execute the real tools and append everything they produced."""
        self.append_assistant_message(ctx, classified, messages, llm_result)

        tool_results_text: list[str] = []
        deferred: list[dict[str, Any]] = []
        remaining = min(
            max(0, ctx.max_per_action - action_tool_calls),
            max(0, ctx.max_total_calls - ctx.state.total_tool_calls),
        )
        executable = classified.regular[:remaining]
        skipped = classified.regular[remaining:]
        action_tool_calls = self._run_batches(
            ctx, executable, messages, action_tool_calls, iteration,
            guard, tracker, tool_results_text, deferred,
        )
        self._answer_budget_limited(ctx, messages, skipped, tool_results_text)

        if ctx.is_small:
            ContextManager.compact_for_small(messages)

        self.append_pseudo_results(ctx, classified, messages, tool_results_text)
        # Only now is the assistant→tool block closed. Everything the model
        # asked for has an answer — the real tools from the batches above,
        # the pseudo-tools from the line before — so a ``user`` turn can
        # finally follow without wedging itself between a tool call and its
        # result. Anything appended before this point would split the block.
        messages.extend(deferred)
        return action_tool_calls

    def run_pseudo_only(
        self,
        ctx: ExecutionContext,
        classified: ClassifiedCalls,
        messages: list[dict[str, Any]],
        llm_result: LLMCallResult,
    ) -> None:
        """The turn asked only for pseudo-tools — no execution, just replies."""
        self.append_assistant_message(ctx, classified, messages, llm_result)
        self.append_pseudo_results(ctx, classified, messages)

    # ── the assistant turn ───────────────────────────────────────────

    @staticmethod
    def append_assistant_message(
        ctx: ExecutionContext,
        classified: ClassifiedCalls,
        messages: list[dict[str, Any]],
        llm_result: LLMCallResult,
    ) -> None:
        """Record what the model just asked for.

        In manual mode the tool calls were parsed out of prose, so the
        assistant turn is that prose. In function-calling mode every call —
        real and pseudo — has to be listed, because a provider that sees a
        ``tool`` result for an id it was never told about will reject the
        conversation.
        """
        message = llm_result.message
        if ctx.manual_tc:
            messages.append({
                "role": "assistant",
                "content": getattr(message, "content", "") or llm_result.raw_content,
            })
            return

        all_calls = list(classified.regular)
        all_calls += classified.thinks + classified.notes + classified.session_notes
        if classified.step_complete:
            all_calls.append(classified.step_complete)
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
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
                for tc in all_calls
            ],
        })

    @staticmethod
    def append_pseudo_results(
        ctx: ExecutionContext,
        classified: ClassifiedCalls,
        messages: list[dict[str, Any]],
        tool_results_text: list[str] | None = None,
    ) -> None:
        """Acknowledge the pseudo-tools (think, notes, step_complete).

        They have no implementation — they exist so the model can signal
        intent through the tool channel — but each one still needs a result,
        or the provider sees an unanswered tool call.
        """
        def note_result(call: Any) -> str:
            return classified.note_results.get(call.id, '{"status": "noted"}')

        if ctx.manual_tc:
            texts = tool_results_text if tool_results_text is not None else []
            texts += [f'[Tool: add_note] Result:\n{note_result(n)}'
                      for n in classified.notes]
            texts += ['[Tool: add_session_note] Result:\n{"status": "noted"}'
                      for _ in classified.session_notes]
            texts += ['[Tool: think] Result:\n{"status": "acknowledged"}'
                      for _ in classified.thinks]
            if texts:
                messages.append({"role": "user", "content": "\n\n".join(texts)})
            return

        def ack(call: Any, body: str) -> None:
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": body}
            )

        for think in classified.thinks:
            ack(think, '{"status": "acknowledged"}')
        for note in classified.notes:
            ack(note, note_result(note))
        for note in classified.session_notes:
            ack(note, '{"status": "noted"}')
        if classified.step_complete:
            ack(classified.step_complete, '{"status": "acknowledged"}')

    # ── execution ────────────────────────────────────────────────────

    def _run_batches(
        self,
        ctx: ExecutionContext,
        calls: list[Any],
        messages: list[dict[str, Any]],
        action_tool_calls: int,
        iteration: int,
        guard: Any,
        tracker: Any,
        tool_results_text: list[str],
        deferred: list[dict[str, Any]],
    ) -> int:
        """Execute every batch, stopping early if the user cancels.

        *deferred* collects the ``user`` turns each batch produced (images,
        the budget nudge) so the caller can write them once the whole
        assistant→tool block is closed. Batches must not write them
        themselves: a second batch appends its results behind the first
        batch's ``user`` turn, and the block is split.
        """
        hook_meta = {
            "agent_name": ctx.agent_name,
            "iteration": iteration,
            "verbose": ctx.verbose,
            "tokens_total": ctx.state.total_tokens,
            "prompt_tokens": ctx.state.last_prompt_tokens,
            "completion_tokens": ctx.state.last_completion_tokens,
            "project_id": ctx.project_id,
            "agent_id": ctx.agent_id,
            "cancel_event": self._engine._tool_cancel_event,
        }
        # tc.id → the images that call produced. Flushed as their own user
        # turn later; see ``_append_results``.
        attachments_by_tc: dict[str, list] = {}

        for batch in batch_tool_calls(calls):
            # Effectful calls stay serial. The legacy name set remains a
            # compatibility fallback, while ToolEffects covers dynamically
            # discovered MCP writers that the static list cannot know about.
            def _writes_or_mutates(tc: Any) -> bool:
                if tc.function.name in WRITE_TOOLS:
                    return True
                bound = ctx.tool_dispatch.get(tc.function.name)
                effects = getattr(bound, "effects", None)
                return bool(
                    effects
                    and (
                        effects.writes_workspace
                        or effects.mutates_git
                        or effects.mutates_internal_state
                        or effects.mutates_external_state
                        or effects.runs_process
                        or effects.destructive
                    )
                )

            is_parallel = len(batch) > 1 and not any(
                _writes_or_mutates(tc) for tc in batch
            )

            self._engine._begin_tool_batch()
            try:
                if is_parallel:
                    hook_meta["call_num"] = action_tool_calls + 1
                    hook_meta["total_calls"] = ctx.state.total_tool_calls + 1
                    batch_results = execute_tool_calls_parallel(
                        batch, ctx.tool_dispatch,
                        hook_metadata=hook_meta,
                        attachments_by_tc=attachments_by_tc,
                    )
                else:
                    batch_results = self._run_serial(
                        ctx, batch, hook_meta, action_tool_calls, attachments_by_tc,
                    )
            finally:
                self._engine._finish_tool_batch()

            if self._engine._cancel_event.is_set():
                self._answer_unreached(ctx, messages, batch,
                                       {tc.id for tc, _ in batch_results})
                break

            action_tool_calls = self._append_results(
                ctx, batch_results, messages, action_tool_calls, iteration,
                is_parallel, guard, tracker, tool_results_text, deferred,
                attachments_by_tc,
            )
            ctx.state.tick_opened_files(1)

        if self._engine._cancel_event.is_set():
            answered = {
                msg["tool_call_id"]
                for msg in messages
                if msg.get("role") == "tool" and "tool_call_id" in msg
            }
            self._answer_unreached(ctx, messages, calls, answered)

        return action_tool_calls

    @staticmethod
    def _answer_budget_limited(
        ctx: ExecutionContext,
        messages: list[dict[str, Any]],
        calls: list[Any],
        tool_results_text: list[str],
    ) -> None:
        """Acknowledge calls refused by a hard execution budget.

        Function-calling providers require one result for every call in the
        assistant message.  Refusing the overflow explicitly preserves that
        protocol while ensuring a parallel batch cannot silently spend more
        than either the per-step or global budget.
        """
        for tc in calls:
            if ctx.manual_tc:
                tool_results_text.append(
                    f"[Tool: {tc.function.name}] Result:\n{_BUDGET_EXHAUSTED_RESULT}"
                )
            else:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _BUDGET_EXHAUSTED_RESULT,
                })

    def _run_serial(
        self,
        ctx: ExecutionContext,
        batch: list[Any],
        hook_meta: dict[str, Any],
        action_tool_calls: int,
        attachments_by_tc: dict[str, list],
    ) -> list[tuple]:
        """Run one batch call-by-call, capturing before/after for diffs."""
        results: list[tuple] = []
        for offset, tc in enumerate(batch):
            pre = capture_pre_content(
                tc.function.name, tc.function.arguments, ctx.file_tracker,
            )
            hook_meta["call_num"] = action_tool_calls + offset + 1
            hook_meta["total_calls"] = ctx.state.total_tool_calls + offset + 1
            attachments: list = []
            attachments_by_tc[tc.id] = attachments
            result = execute_tool_call(
                ctx.tool_dispatch, tc.function.name, tc.function.arguments,
                hook_metadata=hook_meta,
                attachments_out=attachments,
            )
            maybe_emit_file_change(
                tc.function.name, tc.function.arguments, result, pre,
                ctx.file_tracker, ctx.project_id, ctx.agent_id,
                self._engine._hooks,
            )
            results.append((tc, result))
        return results

    @staticmethod
    def _answer_unreached(
        ctx: ExecutionContext,
        messages: list[dict[str, Any]],
        calls: list[Any],
        answered: set[str],
    ) -> None:
        """Give every unrun call a result so the conversation stays valid.

        A cancelled run still has to leave a well-formed transcript: a
        provider that finds a tool call with no matching result rejects the
        next request outright, which would turn "user pressed stop" into
        "the session is broken".
        """
        if ctx.manual_tc:
            return
        for tc in calls:
            if tc.id not in answered:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _CANCELLED_RESULT,
                })

    # ── results → conversation ───────────────────────────────────────

    def _append_results(
        self,
        ctx: ExecutionContext,
        batch_results: list[tuple],
        messages: list[dict[str, Any]],
        action_tool_calls: int,
        iteration: int,
        is_parallel: bool,
        guard: Any,
        tracker: Any,
        tool_results_text: list[str],
        deferred: list[dict[str, Any]],
        attachments_by_tc: dict[str, list] | None = None,
    ) -> int:
        """Write one batch's results into the conversation.

        Two things never go into *messages* here. Image messages, because
        some providers (Minimax among them) reject a ``user`` turn wedged
        between an assistant's tool calls and their results, so the
        assistant→tool block has to stay contiguous. And the budget nudge,
        because firing it mid-batch would interleave a warning with results
        the model has not read yet.

        Both go to *deferred* instead of straight into the conversation.
        Deferring them to the end of this batch is not enough: a write tool
        gets a batch of its own, and pseudo-tool acks are appended after
        every batch has run, so either one would land behind the ``user``
        turn and split the block anyway. ``run_regular`` owns the flush.
        """
        pending_images: list[dict[str, Any]] = []
        pending_nudge: str | None = None

        for tc, result in batch_results:
            tool_error = extract_tool_error(result)
            guard.on_tool_result(tc.function.name, tc.function.arguments, bool(tool_error))
            tracker.on_tool_call(tc.function.name, tc.function.arguments, bool(tool_error))

            if not tool_error:
                update_opened_files_cache(
                    ctx.state, tc.function.name, tc.function.arguments, result,
                )
                ToolProcessor.auto_note_for_small(
                    ctx, tc.function.name, tc.function.arguments, result,
                )

            if tc.function.name == "execute_command":
                # The tool result already contains only the legacy truncated
                # streams plus path-free descriptors. Capture descriptors before
                # test parsing can append prose and make the JSON non-parseable;
                # never alter the result the model receives.
                self._propagate_command_output_handles(ctx, tc.id, result)
                result = self.capture_test_output(ctx, tc.function.arguments, result)

            if not tool_error:
                with best_effort("memory annotation failed for %s", tc.function.name):
                    from infinidev.engine.tool_executor import annotate_with_memory

                    result = annotate_with_memory(
                        tc.function.name, tc.function.arguments, result,
                        project_id=ctx.project_id,
                    )

            body = result + self._counter_tag(ctx, action_tool_calls, is_parallel)
            if feedback := tracker.drain_feedback():
                body += f"\n{feedback}"

            # Queue the raw exchange for working memory before anything
            # downstream gets to shorten it. Both branches: the archiver
            # cannot recover this from the transcript in manual mode (no
            # tool messages) nor on small models (compacted in place).
            ctx.state.pending_archive.append(
                (tc.function.name, str(tc.function.arguments or ""), body)
            )

            if ctx.manual_tc:
                tool_results_text.append(
                    f"[Tool: {tc.function.name}] Result:\n{body}"
                )
            else:
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": body}
                )
                if image_msg := self._image_message(ctx, tc, attachments_by_tc):
                    pending_images.append(image_msg)

            action_tool_calls += 1
            ctx.state.total_tool_calls += 1
            ctx.state.tool_calls_since_last_note += 1

            with best_effort("ContextRank tool call log failed"):
                self._engine._cr_hooks.on_tool_call(
                    tc.function.name, tc.function.arguments, iteration,
                    was_error=bool(tool_error),
                )

            # Ken's reactive channel is computed from exactly this: which
            # files the agent touched, in what way, how many turns ago. Both
            # events are emitted from here rather than straddling execution,
            # because the pair is what Ken stores — pre records the
            # interaction, post retracts it when the tool failed, so a
            # broken read never pushes a file up the ranking.
            with best_effort("ken session tool report failed"):
                self._report_to_ken(tc.function.name, tc.function.arguments, tool_error)

            pending_nudge = self._budget_nudge(ctx, action_tool_calls) or pending_nudge

        deferred.extend(pending_images)

        if pending_nudge is not None:
            if ctx.manual_tc:
                # Manual mode has no tool channel: results are prose in a
                # single ``user`` turn, so the nudge rides along with them
                # and there is no block to split.
                tool_results_text.append(f"\n⚠ STEP BUDGET: {pending_nudge}")
            else:
                deferred.append({"role": "user", "content": pending_nudge})

        return action_tool_calls

    @staticmethod
    def _propagate_command_output_handles(
        ctx: ExecutionContext, tool_call_id: str, result: str,
    ) -> None:
        """Retain only path-free, internally consistent output descriptors.

        ``ExecuteCommandTool`` has already durably verified these handles before
        publishing them. ToolRunner performs a second structural check because
        this boundary is where untrusted tool text becomes loop state. Invalid or
        partial descriptors are ignored; the model-visible result is untouched.
        """
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(payload, dict):
            return
        raw_handles = payload.get("command_output_handles")
        if not isinstance(raw_handles, dict):
            return

        propagated: list[dict[str, int | str]] = []
        for stream, raw in raw_handles.items():
            if stream not in ("stdout", "stderr") or not isinstance(raw, dict):
                return
            if set(raw) != {
                "artifact_id", "type", "stream", "char_count", "byte_count",
            }:
                return
            artifact_id = raw.get("artifact_id")
            char_count = raw.get("char_count")
            byte_count = raw.get("byte_count")
            if (
                type(artifact_id) is not int
                or artifact_id <= 0
                or raw.get("type") != "command_output"
                or raw.get("stream") != stream
                or type(char_count) is not int
                or char_count < 0
                or type(byte_count) is not int
                or byte_count < 0
                or not isinstance(tool_call_id, str)
                or not tool_call_id
            ):
                return
            propagated.append({
                "artifact_id": artifact_id,
                "type": "command_output",
                "stream": stream,
                "char_count": char_count,
                "byte_count": byte_count,
                "tool_call_id": tool_call_id,
            })

        if not propagated:
            return
        pending = getattr(ctx, "pending_command_output_handles", None)
        if pending is None:
            pending = []
            setattr(ctx, "pending_command_output_handles", pending)
        pending.extend(propagated)

    @staticmethod
    def _report_to_ken(tool_name: str, arguments: Any, tool_error: Any) -> None:
        """Feed one tool call into Ken's reactive channel."""
        from infinidev.engine.ken_session import get_ken_session

        session = get_ken_session()
        if session is None or not session.available:
            return
        session.tool_pre(tool_name, arguments)
        session.tool_post(tool_name, success=not tool_error, arguments=arguments)

    @staticmethod
    def _counter_tag(
        ctx: ExecutionContext, action_tool_calls: int, is_parallel: bool
    ) -> str:
        """Tell the model how much of its per-step budget it has spent."""
        tag = (
            f"\n[Tool call {action_tool_calls + 1}/{ctx.max_per_action} "
            f"for this step]"
        )
        return tag + " (parallel)" if is_parallel else tag

    @staticmethod
    def _image_message(
        ctx: ExecutionContext,
        tc: Any,
        attachments_by_tc: dict[str, list] | None,
    ) -> dict[str, Any] | None:
        """Package a tool's images as their own multimodal user turn.

        Most providers reject content blocks inside a ``tool`` message, so a
        separate user turn is the portable path. Returns ``None`` when the
        tool produced no images or the model cannot see them anyway.
        """
        if attachments_by_tc is None:
            return None
        attachments = attachments_by_tc.get(tc.id) or []
        if not attachments:
            return None
        try:
            from infinidev.config.model_capabilities import get_capability_snapshot

            if not get_capability_snapshot().supports_vision:
                return None
        except Exception:
            return None

        blocks: list[dict[str, Any]] = [{
            "type": "text",
            "text": (
                f"[Images attached by tool `{tc.function.name}` — "
                f"{len(attachments)} image(s)]"
            ),
        }]
        blocks += [
            {"type": "image_url", "image_url": {"url": att.data_url}}
            for att in attachments
        ]
        return {"role": "user", "content": blocks}

    def _budget_nudge(
        self, ctx: ExecutionContext, action_tool_calls: int
    ) -> str | None:
        """Warn once, at the threshold, that the step's budget is running out.

        Fires on equality rather than ``>=`` so the model is told once and
        not on every remaining call of the step.
        """
        default = 4 if ctx.is_small else _get_settings().LOOP_STEP_NUDGE_THRESHOLD
        override = self._engine._nudge_threshold_override
        threshold = override if override is not None else default
        if threshold <= 0 or action_tool_calls != threshold:
            return None

        if ctx.nudge_message_template:
            return ctx.nudge_message_template.format(
                used=action_tool_calls, threshold=threshold,
            )
        active = ctx.state.plan.active_step.title if ctx.state.plan.active_step else ""
        return (
            f"You have used {action_tool_calls}/{ctx.max_per_action} tool calls "
            f"for this step. Step scope: \"{active}\". "
            f"Call step_complete now. If the step is not finished, set "
            f"status='continue' and add/modify next_steps to capture the "
            f"remaining work."
        )

    # ── test-runner special case ─────────────────────────────────────

    @staticmethod
    def capture_test_output(
        ctx: ExecutionContext, arguments: str, result: str
    ) -> str:
        """Side-effects and annotation for an ``execute_command`` that ran tests.

        Three jobs, all best-effort — this is an optimisation, and a failure
        here must not turn a passing test run into a broken tool result:

        1. Cache the raw stdout on the state so ``tail_test_output`` can
           serve a filtered view without re-running the suite.
        2. Fingerprint the outcome per *normalised* command, so
           ``regression_after_edit`` compares like with like. Only the last
           two entries per command are kept, to bound the state.
        3. Parse the failures out and append them to the result, so the
           model reads them beside the stdout rather than having to find
           them in it.
        """
        with best_effort("test command capture failed for %s", arguments[:80]):
            from infinidev.engine.guidance import (
                is_test_command,
                normalize_test_command,
                test_outcome_fingerprint,
            )

            if not is_test_command(arguments, ctx.state):
                return result

            ctx.state.last_test_output = result
            try:
                parsed = json.loads(arguments) if arguments else {}
                command = str(parsed.get("command", ""))
            except Exception:
                command = arguments
            ctx.state.last_test_command = command[:300]

            if fingerprint := test_outcome_fingerprint(result):
                key = normalize_test_command(command)
                history = ctx.state.test_outcome_history.get(key, [])
                if not history or history[-1] != fingerprint:
                    history.append(fingerprint)
                    ctx.state.test_outcome_history[key] = history[-2:]

            try:
                from infinidev.engine.test_parsers import parse_test_failures

                failures = parse_test_failures(result)
            except Exception:
                failures = []
            if failures:
                shown = failures[:_MAX_STRUCTURED_FAILURES]
                more = (
                    f", showing first {_MAX_STRUCTURED_FAILURES}"
                    if len(failures) > _MAX_STRUCTURED_FAILURES
                    else ""
                )
                result += (
                    f"\n\n[auto-extracted structured_failures "
                    f"({len(failures)} total{more}):]\n"
                    + json.dumps([f.to_dict() for f in shown], indent=2)
                )

        return result
