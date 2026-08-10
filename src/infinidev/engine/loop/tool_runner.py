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
import os
import re
import shlex
from typing import Any, TYPE_CHECKING

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
from infinidev.tools.base.context import get_current_workspace_path

if TYPE_CHECKING:
    from infinidev.engine.loop.behavior_tracker import BehaviorTracker

logger = logging.getLogger(__name__)

# How many parsed test failures are appended to a test command's output.
# Enough to see the shape of the breakage without pasting a whole suite.
_MAX_STRUCTURED_FAILURES = 8

_CANCELLED_RESULT = '{"error": "cancelled"}'
_BUDGET_EXHAUSTED_RESULT = (
    '{"error": "not_run: tool budget exhausted; inspect completed tool results '
    'and continue in the next step"}'
)
_REPEATED_READ_STATUS = "already_delivered"
_DISCOVERY_SUPPRESSED_STATUS = "discovery_suppressed"
_MAX_READ_DELIVERY_KEYS = 256


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
        step_limit = getattr(ctx, "step_tool_limit", ctx.max_per_action)
        remaining = max(0, step_limit - action_tool_calls)
        if ctx.max_total_calls is not None:
            remaining = min(
                remaining,
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
        else:
            ContextManager.compact_old_tool_results(messages)

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

            plan_results, executable_batch = self._partition_plan_mutations(
                ctx, batch,
            )
            suppressed_results, executable_batch = self._partition_suppressed_discovery(
                ctx, executable_batch,
            )
            cached_results, executable_batch = self._partition_repeated_reads(
                ctx, executable_batch,
            )
            is_parallel = len(executable_batch) > 1 and not any(
                _writes_or_mutates(tc) for tc in batch
            )

            self._engine._begin_tool_batch()
            try:
                if is_parallel:
                    hook_meta["call_num"] = action_tool_calls + 1
                    hook_meta["total_calls"] = ctx.state.total_tool_calls + 1
                    batch_results = execute_tool_calls_parallel(
                        executable_batch, ctx.tool_dispatch,
                        hook_metadata=hook_meta,
                        attachments_by_tc=attachments_by_tc,
                    )
                else:
                    batch_results = self._run_serial(
                        ctx, executable_batch, hook_meta, action_tool_calls,
                        attachments_by_tc,
                    )
            finally:
                self._engine._finish_tool_batch()

            synthetic_results = [
                *plan_results, *suppressed_results, *cached_results,
            ]
            if synthetic_results:
                by_id = {
                    tc.id: (tc, result)
                    for tc, result in [*batch_results, *synthetic_results]
                }
                batch_results = [by_id[tc.id] for tc in batch]

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
    def _partition_plan_mutations(
        ctx: ExecutionContext,
        batch: list[Any],
    ) -> tuple[list[tuple[Any, str]], list[Any]]:
        """Acknowledge out-of-authority plan calls without treating them as errors.

        Some providers emit a familiar plan tool even when its schema is not
        advertised. In fixed-plan and plan-free runs, dispatching that call
        produces ``Unknown tool`` and can spend the whole window retrying an
        operation the outer scheduler deliberately owns. A structured no-op
        preserves protocol ordering and budget accounting while leaving the
        plan immutable.
        """
        if getattr(ctx, "allow_plan_mutation", True) and not getattr(
            ctx, "skip_plan", False
        ):
            return [], batch

        controlled = {"add_step", "modify_step", "remove_step"}
        synthetic: list[tuple[Any, str]] = []
        executable: list[Any] = []
        active = getattr(getattr(ctx.state, "plan", None), "active_step", None)
        mode = (
            "plan_free"
            if getattr(ctx, "skip_plan", False)
            else "scheduler_owned"
        )
        for tc in batch:
            if tc.function.name not in controlled:
                executable.append(tc)
                continue
            synthetic.append((tc, json.dumps({
                "status": "ignored",
                "reason": mode,
                "active_step": getattr(active, "title", None),
                "next_action": (
                    "Use execution tools for the active work, or close it with "
                    "step_complete. The plan was not changed."
                ),
            })))
        return synthetic, executable

    @staticmethod
    def _read_delivery_identity(
        ctx: ExecutionContext,
        tc: Any,
    ) -> tuple[str, str, str] | None:
        """Return ``(key, revision, path)`` for a concrete read_file call."""
        if tc.function.name not in {"read_file", "partial_read"}:
            return None
        try:
            args = (
                json.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else (tc.function.arguments or {})
            )
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(args, dict):
            return None
        raw_path = next(
            (
                args[name]
                for name in ("file_path", "path", "filepath", "file", "filename")
                if isinstance(args.get(name), str) and args[name].strip()
            ),
            None,
        )
        if raw_path is None:
            return None
        workspace = (
            getattr(ctx, "workspace_path", None)
            or get_current_workspace_path()
            or os.getcwd()
        )
        path = os.path.realpath(os.path.join(workspace, os.path.expanduser(raw_path)))
        try:
            stat = os.stat(path)
        except OSError:
            return None
        valid_range, canonical_range = ToolRunner._canonical_read_range(args)
        if not valid_range:
            # ``read_file(limit=0)`` returns an empty-range warning, not file
            # evidence. Caching it would let that warning claim future reads.
            return None
        key = json.dumps(
            [path, canonical_range],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        revision = f"{stat.st_mtime_ns}:{stat.st_size}"
        return key, revision, path

    @staticmethod
    def _partition_suppressed_discovery(
        ctx: ExecutionContext,
        batch: list[Any],
    ) -> tuple[list[tuple[Any, str]], list[Any]]:
        """Decline read-only discovery during one evidence-triggered Step."""
        if not getattr(ctx, "suppress_discovery_this_step", False):
            return [], batch

        from infinidev.engine.loop.behavior_rules import is_workspace_edit_tool
        from infinidev.engine.loop.llm_caller import _COMPLETION_TOOL_NAMES
        from infinidev.engine.loop.semantic_stagnation import (
            SEMANTIC_RECOVERY_CONTEXT_TOOL_NAMES,
        )

        synthetic: list[tuple[Any, str]] = []
        executable: list[Any] = []
        for tc in batch:
            name = tc.function.name
            context_read = name in SEMANTIC_RECOVERY_CONTEXT_TOOL_NAMES
            allow_action = (
                name in _COMPLETION_TOOL_NAMES
                or is_workspace_edit_tool(name)
            )
            if name == "execute_command":
                try:
                    args = (
                        json.loads(tc.function.arguments)
                        if isinstance(tc.function.arguments, str)
                        else (tc.function.arguments or {})
                    )
                except (json.JSONDecodeError, TypeError):
                    args = {}
                command = str(args.get("command", "")) if isinstance(args, dict) else ""
                from infinidev.engine.guidance.test_runners import is_test_command
                from infinidev.tools.base.command_risk import classify_command

                safe, _ = classify_command(command)
                is_test = is_test_command(tc.function.arguments, ctx.state)
                # A non-read-only classification is not evidence that a shell
                # command advances the Step.  In particular, ``python -c`` is
                # often used as another inspection surface and previously let
                # a stagnant model bypass this recovery gate indefinitely.
                # Workspace mutations have first-class tools; the only shell
                # action admitted here is a recognised test runner.
                allow_action = is_test
                context_read = safe and not is_test
            allowance = int(
                getattr(ctx, "semantic_recovery_context_calls", 0) or 0
            )
            if context_read and allowance > 0:
                ctx.semantic_recovery_context_calls = allowance - 1
                executable.append(tc)
                continue
            suppress = not allow_action
            if not suppress:
                executable.append(tc)
                continue
            synthetic.append((tc, json.dumps({
                "status": _DISCOVERY_SUPPRESSED_STATUS,
                "reason": (
                    "semantically repeated Step summaries produced no "
                    "successful edit or new test outcome"
                ),
                "available_actions": (
                    "edit the target named by the active Step, run the test "
                    "whose target covers that edited file, update the plan, "
                    "or complete the Step"
                ),
            })))
        return synthetic, executable

    @staticmethod
    def _identity_for_file_evidence(
        path: str,
        evidence: dict[str, Any],
    ) -> tuple[str, str, str] | None:
        """Build a delivery identity tied to the current file revision."""
        path = os.path.realpath(os.path.expanduser(path))
        try:
            stat = os.stat(path)
        except OSError:
            return None
        if not os.path.isfile(path):
            return None
        key = json.dumps(
            [path, evidence],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return key, f"{stat.st_mtime_ns}:{stat.st_size}", path

    @classmethod
    def _shell_read_identities(
        cls,
        ctx: ExecutionContext,
        tc: Any,
    ) -> list[tuple[str, str, str]] | None:
        """Recognise simple ``cat``/``head`` calls as file evidence.

        This deliberately abstains on pipelines, transforms, stdin and tools
        such as ``awk`` whose program can execute or write. The permission
        classifier answers whether a command is safe; this narrower parser
        answers the different question of exactly which unchanged lines it
        delivered, so only commands with an unambiguous interval qualify.
        """
        if tc.function.name != "execute_command":
            return None
        try:
            args = (
                json.loads(tc.function.arguments)
                if isinstance(tc.function.arguments, str)
                else (tc.function.arguments or {})
            )
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(args, dict) or not isinstance(args.get("command"), str):
            return None

        command = args["command"]
        cwd = args.get("cwd") if isinstance(args.get("cwd"), str) else None
        if cwd is None:
            from infinidev.tools.shell.shell_invocation import split_leading_cd

            split = split_leading_cd(command)
            if split is not None:
                command, cwd = split

        from infinidev.tools.base.command_risk import split_shell_segments

        segments, _ = split_shell_segments(command)
        live_segments = [segment.strip() for segment in segments or [] if segment.strip()]
        if len(live_segments) != 1:
            return None
        try:
            tokens = shlex.split(live_segments[0])
        except ValueError:
            return None
        if not tokens:
            return None

        workspace = (
            getattr(ctx, "workspace_path", None)
            or get_current_workspace_path()
            or os.getcwd()
        )
        base_dir = cwd or workspace
        if not os.path.isabs(base_dir):
            base_dir = os.path.join(workspace, base_dir)

        base = os.path.basename(tokens[0])
        paths: list[str]
        interval: dict[str, int | None]
        if base == "cat":
            # Options may change numbering/visibility, and '-' reads stdin.
            paths = tokens[1:]
            if not paths or any(path.startswith("-") for path in paths):
                return None
            interval = {"start": 1, "end": None}
        elif base == "head":
            count = 10
            paths = []
            index = 1
            while index < len(tokens):
                token = tokens[index]
                if token in ("-n", "--lines"):
                    if index + 1 >= len(tokens):
                        return None
                    try:
                        count = int(tokens[index + 1])
                    except ValueError:
                        return None
                    index += 2
                    continue
                if token.startswith("--lines="):
                    try:
                        count = int(token.split("=", 1)[1])
                    except ValueError:
                        return None
                    index += 1
                    continue
                if re.fullmatch(r"-\d+", token):
                    count = int(token[1:])
                    index += 1
                    continue
                if token.startswith("-"):
                    return None
                paths.append(token)
                index += 1
            if count <= 0 or not paths:
                return None
            interval = {"start": 1, "end": count}
        elif base in {"grep", "egrep", "fgrep", "rg", "wc"}:
            # Exact query cache: these commands select or summarise rather
            # than deliver a contiguous interval. Bind the normalised query
            # to every explicit regular-file argument, and abstain when the
            # command only searches stdin/a directory/the implicit workspace.
            paths = []
            for token in tokens[1:]:
                candidate = token if os.path.isabs(token) else os.path.join(base_dir, token)
                if os.path.isfile(candidate):
                    paths.append(token)
            if not paths:
                return None
            signature = {
                "command": shlex.join(tokens),
                "cwd": os.path.realpath(base_dir),
            }
            identities = []
            for raw_path in paths:
                path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
                identity = cls._identity_for_file_evidence(path, signature)
                if identity is None:
                    return None
                identities.append(identity)
            return identities
        else:
            return None

        identities: list[tuple[str, str, str]] = []
        for raw_path in paths:
            if raw_path == "-":
                return None
            path = raw_path if os.path.isabs(raw_path) else os.path.join(base_dir, raw_path)
            identity = cls._identity_for_file_evidence(path, interval)
            if identity is None:
                return None
            identities.append(identity)
        return identities

    @classmethod
    def _read_delivery_identities(
        cls,
        ctx: ExecutionContext,
        tc: Any,
    ) -> list[tuple[str, str, str]] | None:
        identity = cls._read_delivery_identity(ctx, tc)
        if identity is not None:
            return [identity]
        return cls._shell_read_identities(ctx, tc)

    @staticmethod
    def _canonical_read_range(
        args: dict[str, Any],
    ) -> tuple[bool, dict[str, int | None] | None]:
        """Mirror ``ReadFileTool`` aliases as one delivered line interval.

        ``None`` is the tool's unbounded/skeleton request. A finite mapping is
        inclusive, so syntactically different calls for the same lines share
        an identity and overlapping deliveries can be unioned later.
        """

        def as_int(value: Any) -> int | None:
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        offset = as_int(args.get("offset"))
        limit = as_int(args.get("limit"))
        start_line = as_int(args.get("start_line"))
        end_line = as_int(args.get("end_line"))

        line_range = args.get("line_range")
        if line_range is not None and offset is None:
            match = re.match(r"(\d+)\s*[-:,]\s*(\d+)", str(line_range))
            if match:
                start_line = int(match.group(1))
                end_line = int(match.group(2))
            else:
                start_line = as_int(line_range)

        if start_line is not None and offset is None:
            offset = start_line
        if end_line is not None and limit is None:
            if offset is None:
                offset = 1
            limit = max(1, end_line - (offset or 1) + 1)

        if limit is not None and limit <= 0:
            return False, None
        if offset is None and limit is None:
            return True, None

        start = max(offset or 1, 1)
        end = None if limit is None else start + limit - 1
        return True, {"start": start, "end": end}

    @staticmethod
    def _read_range_from_key(
        key: str,
    ) -> tuple[str, dict[str, int | None] | None] | None:
        """Decode a current delivery key, ignoring older incompatible shapes."""
        try:
            path, interval = json.loads(key)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None
        if not isinstance(path, str):
            return None
        if interval is None:
            return path, None
        if not isinstance(interval, dict) or set(interval) != {"start", "end"}:
            return None
        start = interval.get("start")
        end = interval.get("end")
        if not isinstance(start, int) or (end is not None and not isinstance(end, int)):
            return None
        return path, {"start": start, "end": end}

    @classmethod
    def _read_range_is_covered(
        cls,
        delivered: dict[str, str],
        *,
        path: str,
        revision: str,
        target: dict[str, int | None],
    ) -> bool:
        """Whether unchanged prior deliveries cover every target line."""
        intervals: list[tuple[int, int | None]] = []
        for key, stored_revision in delivered.items():
            if stored_revision != revision:
                continue
            decoded = cls._read_range_from_key(key)
            if decoded is None or decoded[0] != path or decoded[1] is None:
                continue
            intervals.append((decoded[1]["start"], decoded[1]["end"]))

        target_start = target["start"]
        target_end = target["end"]
        cursor = target_start
        # Sort only by the lower bound. Two deliveries may start on the same
        # line while one is open-ended; tuple ordering would then compare
        # ``None`` with an integer and crash instead of answering the read.
        for start, end in sorted(intervals, key=lambda item: item[0]):
            if end is not None and end < cursor:
                continue
            if start > cursor:
                break
            if end is None:
                return True
            cursor = max(cursor, end + 1)
            if target_end is not None and cursor > target_end:
                return True
        return False

    @classmethod
    def _partition_repeated_reads(
        cls,
        ctx: ExecutionContext,
        batch: list[Any],
    ) -> tuple[list[tuple[Any, str]], list[Any]]:
        """Replace unchanged exact re-reads with a bounded cache notice."""
        cached: list[tuple[Any, str]] = []
        executable: list[Any] = []
        delivered = ctx.state.read_delivery_revisions
        for tc in batch:
            identities = cls._read_delivery_identities(ctx, tc)
            if not identities:
                executable.append(tc)
                continue
            statuses: list[str] = []
            paths: list[str] = []
            for key, revision, path in identities:
                decoded = cls._read_range_from_key(key)
                interval = decoded[1] if decoded is not None else None
                exact = delivered.get(key) == revision
                covered = bool(
                    not exact
                    and interval is not None
                    and cls._read_range_is_covered(
                        delivered,
                        path=path,
                        revision=revision,
                        target=interval,
                    )
                )
                statuses.append("exact" if exact else "contained" if covered else "")
                paths.append(path)
            if not all(statuses):
                executable.append(tc)
                continue
            payload: dict[str, Any] = {
                "status": _REPEATED_READ_STATUS,
                "coverage": "exact" if all(s == "exact" for s in statuses) else "contained",
                "message": (
                    "This unchanged read is already fully covered by prior "
                    "results. Use the existing opened-files/prior evidence; "
                    "running it again would add no information."
                ),
            }
            payload["path" if len(paths) == 1 else "paths"] = (
                paths[0] if len(paths) == 1 else paths
            )
            cached.append((tc, json.dumps(payload)))
        return cached, executable

    @classmethod
    def _record_read_delivery(
        cls,
        ctx: ExecutionContext,
        tc: Any,
    ) -> None:
        identities = cls._read_delivery_identities(ctx, tc)
        if not identities:
            return
        delivered = ctx.state.read_delivery_revisions
        for key, revision, _ in identities:
            delivered[key] = revision
        while len(delivered) > _MAX_READ_DELIVERY_KEYS:
            delivered.pop(next(iter(delivered)))

    @staticmethod
    def _is_synthetic_no_evidence_result(result: str) -> bool:
        try:
            payload = json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return False
        return (
            isinstance(payload, dict)
            and payload.get("status") in {
                _REPEATED_READ_STATUS,
                _DISCOVERY_SUPPRESSED_STATUS,
            }
        )

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
            no_new_evidence = self._is_synthetic_no_evidence_result(result)
            guard.on_tool_result(tc.function.name, tc.function.arguments, bool(tool_error))
            tracker.on_tool_call(tc.function.name, tc.function.arguments, bool(tool_error))

            if not tool_error and not no_new_evidence:
                update_opened_files_cache(
                    ctx.state, tc.function.name, tc.function.arguments, result,
                )
                self._record_read_delivery(ctx, tc)
                ToolProcessor.auto_note_for_small(
                    ctx, tc.function.name, tc.function.arguments, result,
                )

            if tc.function.name == "execute_command":
                # The tool result already contains only the legacy truncated
                # streams plus path-free descriptors. Capture descriptors before
                # test parsing can append prose and make the JSON non-parseable;
                # never alter the result the model receives.
                self._propagate_command_output_handles(ctx, tc.id, result)
                if not tool_error:
                    # A denied or malformed command did not run. Recording it as
                    # the latest test command makes the deterministic reviewer
                    # replay a known denial and reject otherwise-green work.
                    self._record_successful_step_test(
                        ctx, tc.function.arguments, result, tracker,
                    )
                    result = self.capture_test_output(
                        ctx, tc.function.arguments, result,
                    )
                    self._refresh_opened_files_after_shell(
                        ctx, tc.function.arguments,
                    )

            if not tool_error and not no_new_evidence:
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
    def _refresh_opened_files_after_shell(
        ctx: ExecutionContext, arguments: str | dict[str, Any],
    ) -> None:
        """Reconcile model-visible file state after a possibly mutating shell.

        Shell commands can move, delete, generate, or rewrite files without
        passing through the first-class file tools.  Keeping the old
        ``opened_files`` entries after such a command tells the next model turn
        that deleted content is current.  Read-only commands are proven by the
        same allow-list used by the permission broker; everything else gets a
        bounded refresh of the at-most-ten cached files.
        """
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            args = {}
        command = str(args.get("command", "")) if isinstance(args, dict) else ""

        from infinidev.tools.base.command_risk import classify_command

        safe, _ = classify_command(command)
        if safe:
            return

        workspace = getattr(ctx, "workspace_path", None) or os.getcwd()
        for key, opened in list(ctx.state.opened_files.items()):
            if key.startswith("[symbol] "):
                ctx.state.opened_files.pop(key, None)
                continue
            path = key if os.path.isabs(key) else os.path.join(workspace, key)
            if not os.path.isfile(path):
                ctx.state.opened_files.pop(key, None)
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    content = handle.read()
            except OSError:
                ctx.state.opened_files.pop(key, None)
                continue
            if len(content) > 32_000:
                ctx.state.opened_files.pop(key, None)
                continue
            opened.content = "\n".join(
                f"{index + 1:>6}\t{line}"
                for index, line in enumerate(content.split("\n"))
            )

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
            f"\n[Tool call {action_tool_calls + 1}/"
            f"{getattr(ctx, 'step_tool_limit', ctx.max_per_action)} "
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
        step_limit = getattr(ctx, "step_tool_limit", ctx.max_per_action)
        remaining = max(0, step_limit - action_tool_calls)
        return (
            f"You have used {action_tool_calls}/{step_limit} tool calls "
            f"for this step ({remaining} remain). Step scope: \"{active}\". "
            "If the success criterion is already verified, call step_complete. "
            "Otherwise spend the remaining calls on the highest-value concrete edit "
            "or verification; if work still remains at the limit, close with "
            "status='continue' and record the exact next action."
        )

    # ── test-runner special case ─────────────────────────────────────

    @staticmethod
    def _record_successful_step_test(
        ctx: ExecutionContext,
        arguments: str,
        result: str,
        tracker: BehaviorTracker,
    ) -> None:
        """Attach a successful recognised test command to Step-local evidence."""
        with best_effort("step test evidence capture failed for %s", arguments[:80]):
            from infinidev.engine.guidance import is_test_command

            if not is_test_command(arguments, ctx.state):
                return
            parsed_args = json.loads(arguments) if arguments else {}
            command = str(parsed_args.get("command", ""))
            payload = json.loads(result)
            if (
                isinstance(payload, dict)
                and payload.get("exit_code") == 0
                and payload.get("success", True) is not False
            ):
                tracker.on_successful_test(command)

    @staticmethod
    def capture_test_output(
        ctx: ExecutionContext, arguments: str, result: str,
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
            try:
                payload = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                payload = {}
            exit_code = payload.get("exit_code") if isinstance(payload, dict) else None
            if isinstance(exit_code, int) and not isinstance(exit_code, bool):
                ctx.state.last_test_exit_code = exit_code

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
