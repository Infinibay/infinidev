"""Loop guard — detects repetition loops, error cascades, and budget exhaustion."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from infinidev.engine.engine_logging import (
    emit_log as _emit_log,
    YELLOW as _YELLOW,
    RESET as _RESET,
)
from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.loop.models import StepResult

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext

_MAX_SAME_TOOL_CONSECUTIVE = 3
_MAX_TEXT_RETRIES = 3  # Hard limit per inner loop — all retries are errors
# Turns of pseudo-tools only (think / add_note, no step_complete) allowed
# per step. Generous, because thinking then acting is legitimate; it exists
# to bound the case where the model never gets to the acting part.
_MAX_PSEUDO_ONLY_ROUNDS = 4
_MAX_NON_PROGRESS_TOOL_CALLS = 12
_MAX_EVIDENCE_RESETS_PER_WORKSPACE = 1
_EDIT_REQUIRING_TASK_KINDS = frozenset({
    "feature", "bugfix", "refactor", "performance", "docs", "test", "chore",
    "config", "migration", "security",
})


class LoopGuard:
    """Detects repetition loops, error cascades, and budget exhaustion."""

    def __init__(self, is_small: bool = False) -> None:
        self._is_small = is_small
        # Cross-iteration state (NOT reset by reset())
        self.text_only_iterations = 0
        self.reset()

    def reset(self) -> None:
        self.text_retries = 0
        self.consecutive_tool_errors = 0
        self.last_tool_sig: str | None = None
        self.last_tool_had_error = False
        self.same_tool_streak = 0
        self.repetition_nudged = False
        self.repetition_recovery_emitted = False
        self.reads_since_last_note = 0
        self._note_nudged = False
        self.pseudo_only_rounds = 0
        self.non_progress_tool_calls = 0
        self.workspace_stagnation_tool_calls = 0
        self.evidence_progress_resets = 0
        self._progress_since_check = False
        self._workspace_progress_since_check = False
        self._last_workspace_fingerprint: Any | None = None
        self._seen_workspace_fingerprints: set[Any] = set()

    def seed_workspace_fingerprint(self, fingerprint: Any) -> None:
        """Start net-progress tracking from the Step's current workspace."""
        self._last_workspace_fingerprint = fingerprint
        self._seen_workspace_fingerprints = {fingerprint}

    def _observe_workspace_fingerprint(self, fingerprint: Any) -> bool:
        """Return true only for a workspace state not seen in this Step."""
        changed = fingerprint != self._last_workspace_fingerprint
        novel = changed and fingerprint not in self._seen_workspace_fingerprints
        self._last_workspace_fingerprint = fingerprint
        self._seen_workspace_fingerprints.add(fingerprint)
        return novel


    def mark_text_only_iteration(self) -> None:
        """Called when an inner loop produced zero tool calls."""
        self.text_only_iterations += 1

    def mark_productive_iteration(self) -> None:
        """Called when an inner loop produced at least one tool call."""
        self.text_only_iterations = 0

    def on_tool_result(
        self,
        tool_name: str,
        args: str,
        had_error: bool,
        *,
        made_progress: bool = False,
        workspace_fingerprint: Any | None = None,
    ) -> None:
        """Track repetition, errors, and evidence-free tool drift."""
        workspace_progress = False
        if workspace_fingerprint is not None:
            workspace_progress = self._observe_workspace_fingerprint(
                workspace_fingerprint
            )
            made_progress = workspace_progress or made_progress
        if made_progress:
            self.non_progress_tool_calls = 0
            self._progress_since_check = True
            if workspace_progress:
                self._workspace_progress_since_check = True
        else:
            self.non_progress_tool_calls += 1
        if workspace_progress:
            self.workspace_stagnation_tool_calls = 0
            self.evidence_progress_resets = 0
        elif (
            made_progress
            and self.evidence_progress_resets < _MAX_EVIDENCE_RESETS_PER_WORKSPACE
        ):
            self.workspace_stagnation_tool_calls = 0
            self.evidence_progress_resets += 1
        else:
            self.workspace_stagnation_tool_calls += 1
        if had_error:
            self.consecutive_tool_errors += 1
        else:
            self.consecutive_tool_errors = 0

        # Track reads without notes (for small model nudging)
        if tool_name in ("read_file", "partial_read"):
            self.reads_since_last_note += 1

        sig = f"{tool_name}:{args}"
        if sig == self.last_tool_sig:
            self.same_tool_streak += 1
        else:
            self.last_tool_sig = sig
            self.same_tool_streak = 1
            self.repetition_nudged = False
            self.repetition_recovery_emitted = False
        self.last_tool_had_error = had_error

    def reset_read_counter(self) -> None:
        """Reset the read-without-note counter (called when a note is recorded)."""
        self.reads_since_last_note = 0
        self._note_nudged = False

    def check_repetition(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> StepResult | None:
        """Redirect an identical-call loop without ending the active Step."""
        threshold = (
            2
            if self._is_small or self.last_tool_had_error
            else _MAX_SAME_TOOL_CONSECUTIVE
        )
        tool_name = (self.last_tool_sig or "").split(":", 1)[0]

        if self.same_tool_streak >= threshold and not self.repetition_nudged:
            self.repetition_nudged = True
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Identical '{tool_name}' call repeated "
                f"{self.same_tool_streak}x — requiring a changed approach{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            messages.append({
                "role": "user",
                "content": (
                    f"REPETITION: You have made the exact same '{tool_name}' call "
                    f"{self.same_tool_streak} times in a row with identical arguments. "
                    "This is a loop. Change the arguments, tool, or strategy now. "
                    "Complete the Step only if its success criterion is actually "
                    "satisfied or a concrete external blocker remains."
                ),
            })
            return None  # nudged, not forced — caller should continue

        if (
            self.same_tool_streak >= threshold + 2
            and not self.repetition_recovery_emitted
        ):
            self.repetition_recovery_emitted = True
            ctx.suppress_discovery_this_step = True
            ctx.semantic_recovery_context_calls = 0
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Identical '{tool_name}' call repeated "
                f"{self.same_tool_streak}x — narrowing tools without ending the Step{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            messages.append({
                "role": "user",
                "content": (
                    "PROGRESS RECOVERY: the repeated call is no longer useful. "
                    f"Do not issue the same '{tool_name}' call again. "
                    "Use a materially different action: change its arguments, edit "
                    "the grounded target, run a focused test, or state the exact "
                    "external blocker. The Step remains active and has no call budget."
                ),
            })
        return None

    def check_error_circuit_breaker(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> None:
        """Append nudge if too many consecutive tool errors."""
        _MAX = 4
        if self.consecutive_tool_errors >= _MAX:
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ {self.consecutive_tool_errors} consecutive tool errors "
                f"— nudging model to try a different approach{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            self.consecutive_tool_errors = 0
            messages.append({
                "role": "user",
                "content": (
                    f"WARNING: Your last {_MAX} tool calls all failed. "
                    "Do not repeat the failed call unchanged. Diagnose the returned "
                    "error and make the next call differ in command, path, working "
                    "directory, arguments, or strategy. A known local correction must "
                    "be tried before you call the step blocked. Examples:\n"
                    "- If edit_file keeps failing, read the file again — old_string must match "
                    "the current bytes exactly, and it must be unique.\n"
                    "- If read_file keeps failing on a path, use glob or list_directory to find the correct path.\n"
                    "- If a shell command ran in the wrong directory, call execute_command "
                    "again with its cwd argument set to the intended directory.\n"
                    "Call step_complete(status='blocked') only when the changed approach also "
                    "cannot proceed, or the remaining blocker needs user action."
                ),
            })

    def check_progress_drift(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> None:
        """Narrow discovery after evidence-free work without ending the Step."""
        task = getattr(ctx, "task", None)
        task_kind = str(getattr(task, "kind", "") or "").casefold()
        if task is not None and task_kind not in _EDIT_REQUIRING_TASK_KINDS:
            return

        if self._progress_since_check:
            requires_workspace_change = getattr(
                ctx, "recovery_requires_workspace_change", False,
            )
            if (
                requires_workspace_change
                and ctx.suppress_discovery_this_step
                and not self._workspace_progress_since_check
            ):
                self._progress_since_check = False
                return
            if not requires_workspace_change or self._workspace_progress_since_check:
                ctx.suppress_discovery_this_step = False
                ctx.semantic_recovery_context_calls = 0
                self._progress_since_check = False
                self._workspace_progress_since_check = False
                return
            self._progress_since_check = False
            self._workspace_progress_since_check = False
        if (
            not getattr(ctx, "semantic_stagnation_control", False)
            or ctx.suppress_discovery_this_step
            or self.workspace_stagnation_tool_calls < _MAX_NON_PROGRESS_TOOL_CALLS
        ):
            return

        from infinidev.engine.loop.semantic_stagnation import (
            SEMANTIC_RECOVERY_CONTEXT_CALLS,
        )

        ctx.suppress_discovery_this_step = True
        ctx.semantic_recovery_context_calls = SEMANTIC_RECOVERY_CONTEXT_CALLS
        _emit_log(
            "warning",
            f"{_YELLOW}⚠ {self.workspace_stagnation_tool_calls} tool calls produced "
            f"no net workspace change; narrowing discovery without ending "
            f"the Step{_RESET}",
            project_id=ctx.project_id,
            agent_id=ctx.agent_id,
        )
        notice = (
            "PROGRESS RECOVERY: recent tool calls produced no net workspace change. "
            "Read and test evidence remains available, but it no longer counts as "
            "implementation progress. This is not a tool-call budget and the Step "
            "remains active. Broad discovery is now unavailable. Recovery is an "
            "experiment phase, not a certainty gate: use the small direct read_file "
            "allowance only for the most plausible edit target, make one reversible "
            "edit now, and let its focused test accept or reject the hypothesis. "
            "Low confidence or multiple plausible fixes is not an external blocker. "
            "Report blocked only after a concrete external constraint prevents both "
            "the edit and its check."
        )
        if messages and messages[-1].get("role") == "user":
            content = messages[-1].get("content", "")
            if isinstance(content, str):
                messages[-1]["content"] = f"{content}\n\n{notice}"
                return
        messages.append({"role": "user", "content": notice})

    def check_note_discipline(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> None:
        """Nudge small models to save notes after multiple reads without noting."""
        if not self._is_small:
            return
        if self.reads_since_last_note >= 2 and not self._note_nudged:
            self._note_nudged = True
            messages.append({
                "role": "user",
                "content": (
                    "You read files but saved no notes. Call add_note NOW with what you found. "
                    "Example: add_note(note='verify_token at auth.py line 42, uses JWT')"
                ),
            })

    def handle_pseudo_only(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> StepResult | None:
        """Bound a turn that asked for nothing but ``think`` / ``add_note``.

        The inner ``while`` advances on ``action_tool_calls``, which only
        moves when a *regular* tool runs. A turn of pure pseudo-tools is
        therefore free: it spends no budget, trips no guard, and can repeat
        for as long as the model keeps doing it — an unbounded spin that
        ``max_per_action`` was assumed to cover and does not.

        Returns a ``StepResult`` once the allowance is spent, ``None`` to
        keep going.
        """
        self.pseudo_only_rounds += 1
        if self.pseudo_only_rounds <= _MAX_PSEUDO_ONLY_ROUNDS:
            return None

        _emit_log(
            "warning",
            f"{_YELLOW}⚠ {self.pseudo_only_rounds} consecutive turns without a "
            f"real tool call — pausing the step{_RESET}",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        return StepResult(
            summary=(
                "Step interrupted: the model kept thinking and taking notes "
                "without calling a real tool or completing the step."
            ),
            status="continue",
            interrupted=True,
        )

    def handle_text_only(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        content: str,
    ) -> StepResult | None:
        """Handle LLM text response without tool calls.

        Every text-only response is an error — the model MUST produce function
        calls, never plain text.  After _MAX_TEXT_RETRIES error messages the
        step is force-completed.  If the model has been text-only for multiple
        consecutive iterations (tracked via text_only_iterations), the budget
        is slashed so we bail out faster.

        Returns StepResult if retries exhausted, None to continue inner loop.
        """
        self.text_retries += 1

        # Slash budget on repeat text-only iterations: 3 → 2 → 1
        budget = max(1, _MAX_TEXT_RETRIES - self.text_only_iterations)

        if self.text_retries > budget:
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ LLM returned text {self.text_retries}x without "
                f"calling a tool — pausing the step{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            summary = content[:197] + "..." if len(content) > 200 else content
            return StepResult(
                summary=summary or "Step completed (model failed to produce tool calls).",
                status="continue",
                interrupted=True,
            )

        # Dispatch thinking hook
        if content:
            _hook_manager.dispatch(_HookContext(
                event=_HookEvent.POST_TOOL,
                tool_name="think",
                arguments={"reasoning": content},
                result=content,
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            ))
        messages.append({"role": "assistant", "content": content})

        # Hard error — no gentle nudging
        _emit_log(
            "warning",
            f"{_YELLOW}⚠ No function call detected (retry "
            f"{self.text_retries}/{budget}){_RESET}",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        if ctx.manual_tc:
            nudge = (
                f"ERROR ({self.text_retries}/{budget}): Text responses are NOT "
                f"allowed. You MUST respond with a JSON function call. "
                f"Do NOT explain, do NOT think, just call a tool. Format:\n"
                f'{{"tool_calls": [{{"name": "tool_name", '
                f'"arguments": {{"param": "value"}}}}]}}'
            )
        else:
            nudge = (
                f"ERROR ({self.text_retries}/{budget}): Text responses are NOT "
                f"allowed. You MUST respond with a function call. "
                f"Do NOT output plain text. Call a tool now, or call "
                f"step_complete if you have nothing to do."
            )

        messages.append({"role": "user", "content": nudge})
        return None  # continue inner loop
