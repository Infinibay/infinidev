"""Loop engine adapter for the reactive-guidance subsystem.

Wraps ``infinidev.engine.guidance.maybe_queue_guidance`` so the loop
engine doesn't repeat the same "check settings → call detector → log"
boilerplate in two places (end-of-step and mid-step). Keeping it here
means guidance policy changes touch one file, and the engine stays
focused on the plan-execute-summarize control flow.

Stateless on purpose: reads config on every call because
``LOOP_GUIDANCE_ENABLED`` can flip at runtime via the settings reloader.
"""

from __future__ import annotations

from typing import Any


class GuidanceHandler:
    """Gate + dispatch for the reactive guidance detector."""

    def try_queue(
        self,
        ctx: Any,
        messages: list[dict[str, Any]],
        step_messages_start: int,
        *,
        mid_step: bool,
    ) -> None:
        """Run the guidance detector if conditions allow; log on success.

        Mirrors the original inline logic: small models only, feature
        flag respected, mid-step calls additionally require that no
        guidance is already pending (so we don't stack two hints in
        the same step).  Any exception is swallowed — guidance is a
        best-effort nicety, never a blocker.
        """
        try:
            from infinidev.engine.loop.step_manager import _get_settings
            from infinidev.engine.engine_logging import log as _log, YELLOW, RESET

            _settings = _get_settings()
            if not ctx.is_small:
                return
            if not getattr(_settings, "LOOP_GUIDANCE_ENABLED", True):
                return
            if mid_step and ctx.state.pending_guidance:
                return

            from infinidev.engine.guidance import maybe_queue_guidance
            queued = maybe_queue_guidance(
                ctx.state,
                messages[step_messages_start:],
                is_small=True,
                max_per_task=int(getattr(_settings, "LOOP_GUIDANCE_MAX_PER_TASK", 3)),
            )
            if queued and ctx.verbose:
                suffix = " mid-step" if mid_step else ""
                _log(f"  {YELLOW}↪ guidance queued{suffix}: {queued}{RESET}")
        except Exception:
            pass
