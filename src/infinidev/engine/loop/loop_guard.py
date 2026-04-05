"""Loop guard — detects repetition loops, error cascades, and budget exhaustion."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from infinidev.engine.engine_logging import (
    emit_log as _emit_log,
    YELLOW as _YELLOW,
    RED as _RED,
    RESET as _RESET,
)
from infinidev.engine.hooks.hooks import hook_manager as _hook_manager, HookContext as _HookContext, HookEvent as _HookEvent
from infinidev.engine.loop.models import StepResult

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext

_MAX_SAME_TOOL_CONSECUTIVE = 3
_MAX_TEXT_RETRIES = 5


class LoopGuard:
    """Detects repetition loops, error cascades, and budget exhaustion."""

    def __init__(self, is_small: bool = False) -> None:
        self._is_small = is_small
        self.reset()

    def reset(self) -> None:
        self.text_retries = 0
        self.consecutive_tool_errors = 0
        self.last_tool_sig: str | None = None
        self.same_tool_streak = 0
        self.repetition_nudged = False
        self.reads_since_last_note = 0
        self._note_nudged = False

    def on_tool_result(self, tool_name: str, args: str, had_error: bool) -> None:
        """Track a tool call for repetition/error detection."""
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

    def reset_read_counter(self) -> None:
        """Reset the read-without-note counter (called when a note is recorded)."""
        self.reads_since_last_note = 0
        self._note_nudged = False

    def check_repetition(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
    ) -> StepResult | None:
        """Returns StepResult if loop detected and must force-break, else None."""
        threshold = 2 if self._is_small else _MAX_SAME_TOOL_CONSECUTIVE
        tool_name = (self.last_tool_sig or "").split(":", 1)[0]

        if self.same_tool_streak >= threshold and not self.repetition_nudged:
            self.repetition_nudged = True
            _emit_log(
                "warning",
                f"{_YELLOW}⚠ Identical '{tool_name}' call repeated "
                f"{self.same_tool_streak}x — nudging step_complete{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            messages.append({
                "role": "user",
                "content": (
                    f"STOP: You have made the exact same '{tool_name}' call "
                    f"{self.same_tool_streak} times in a row with identical arguments. "
                    f"This is a loop. You MUST now call the step_complete "
                    f"tool to summarize what you've accomplished and move on."
                ),
            })
            return None  # nudged, not forced — caller should continue

        if self.same_tool_streak >= threshold + 2:
            _emit_log(
                "error",
                f"{_RED}⚠ Tool loop detected: identical '{tool_name}' call "
                f"{self.same_tool_streak}x — forcing step completion{_RESET}",
                project_id=ctx.project_id, agent_id=ctx.agent_id,
            )
            return StepResult(
                summary=f"Step interrupted: identical {tool_name} calls ({self.same_tool_streak}x) without progress.",
                status="continue",
            )
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
                    "You are stuck in a failing pattern. Change your approach:\n"
                    "- If edit_symbol or replace_lines keeps failing, try a different edit tool.\n"
                    "- If read_file keeps failing on a path, use glob or list_directory to find the correct path.\n"
                    "- If nothing works, call step_complete(status='blocked') to move on."
                ),
            })

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

    def handle_text_only(
        self, ctx: ExecutionContext, messages: list[dict[str, Any]],
        content: str,
    ) -> StepResult | None:
        """Handle LLM text response without tool calls.

        Returns StepResult if retries exhausted, None to continue inner loop.
        """
        self.text_retries += 1

        if self.text_retries < _MAX_TEXT_RETRIES:
            if content:
                _hook_manager.dispatch(_HookContext(
                    event=_HookEvent.POST_TOOL,
                    tool_name="think",
                    arguments={"reasoning": content},
                    result=content,
                    project_id=ctx.project_id, agent_id=ctx.agent_id,
                ))
            messages.append({"role": "assistant", "content": content})
            if ctx.manual_tc:
                # For small models: include available tool names in the nudge
                available = sorted(list(ctx.tool_dispatch.keys())[:12])
                tools_str = ", ".join(available) if available else "read_file, step_complete"
                nudge = (
                    "You must respond with a JSON tool call. Exact format:\n"
                    '{"tool_calls": [{"name": "read_file", "arguments": {"file_path": "src/main.py"}}]}\n\n'
                    f"Available tools: {tools_str}"
                )
            else:
                nudge = (
                    "Good reasoning. Now call the appropriate tool to "
                    "execute your plan, or call step_complete if done."
                )
            messages.append({"role": "user", "content": nudge})
            return None  # continue inner loop

        _emit_log(
            "warning",
            f"{_YELLOW}⚠ LLM returned text {self.text_retries}x without "
            f"calling a tool — moving to next step{_RESET}",
            project_id=ctx.project_id, agent_id=ctx.agent_id,
        )
        summary = content[:197] + "..." if len(content) > 200 else content
        return StepResult(
            summary=summary or "Step completed (model reasoned but did not call tools).",
            status="continue",
        )
