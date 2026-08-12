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

from infinidev.engine._best_effort import best_effort


class GuidanceHandler:
    """Gate + dispatch for the reactive guidance detector."""

    def observe_reasoning(
        self,
        ctx: Any,
        messages: list[dict[str, Any]],
        llm_result: Any,
    ) -> None:
        """Classify provider-visible reasoning before its tool round executes."""
        with best_effort("reasoning mini-model observation failed"):
            from infinidev.engine.behavior.reasoning_content import extract_reasoning
            from infinidev.engine.behavior.runtime_policy import (
                observe_reasoning_behavior,
            )
            from infinidev.engine.loop.step_manager import _get_settings

            current_settings = _get_settings()
            if not getattr(
                current_settings, "ADAPTIVE_RUNTIME_REASONING_ENABLED", True
            ):
                return
            message = getattr(llm_result, "message", None)
            envelope = extract_reasoning(message) if message is not None else None
            text = getattr(llm_result, "reasoning_content", "") or (
                envelope.text if envelope is not None else ""
            )
            observe_reasoning_behavior(
                ctx.state,
                text,
                messages,
                task=getattr(ctx, "task", None),
                current_tool_calls=list(
                    getattr(llm_result, "tool_calls", None) or []
                ),
                sources=envelope.sources if envelope is not None else (),
                shadow_mode=bool(
                    getattr(
                        current_settings,
                        "ADAPTIVE_RUNTIME_REASONING_SHADOW_MODE",
                        True,
                    )
                ),
                max_interventions=int(
                    getattr(current_settings, "ADAPTIVE_RUNTIME_MAX_INTERVENTIONS", 2)
                ),
            )

    @staticmethod
    def inject_pending(ctx: Any, messages: list[dict[str, Any]]) -> bool:
        """Deliver an intervention after assistant/tool ordering is closed."""
        from infinidev.engine.behavior.runtime_policy import (
            drain_runtime_intervention,
        )

        intervention = drain_runtime_intervention(ctx.state)
        if not intervention:
            return False
        if messages and messages[-1].get("role") == "user":
            messages[-1]["content"] = (
                f"{messages[-1].get('content', '')}\n\n{intervention}"
            )
        else:
            messages.append({"role": "user", "content": intervention})
        events = list(getattr(ctx.state, "runtime_behavior_events", ()) or ())
        for event in reversed(events):
            if event.get("intervention_queued") and not event.get(
                "intervention_delivered"
            ):
                event["intervention_delivered"] = True
                event["delivery_channel"] = "next-user-turn"
                break
        ctx.state.runtime_behavior_events = events[-64:]
        return True

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
        with best_effort("guidance queue check failed"):
            from infinidev.engine.loop.step_manager import _get_settings
            from infinidev.engine.engine_logging import log as _log, YELLOW, RESET

            _settings = _get_settings()
            if getattr(_settings, "ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED", True):
                from infinidev.engine.behavior.runtime_policy import (
                    observe_runtime_behavior,
                )

                active_step = getattr(ctx.state.plan, "active_step", None)
                if mid_step:
                    runtime_step_index = getattr(active_step, "index", None)
                else:
                    history = list(getattr(ctx.state, "history", ()) or ())
                    runtime_step_index = history[-1].step_index if history else None

                runtime_queued = observe_runtime_behavior(
                    ctx.state,
                    messages[step_messages_start:],
                    task=getattr(ctx, "task", None),
                    shadow_mode=bool(
                        getattr(_settings, "ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE", True)
                    ),
                    max_interventions=int(
                        getattr(_settings, "ADAPTIVE_RUNTIME_MAX_INTERVENTIONS", 2)
                    ),
                    opened_files_budget_chars=int(
                        getattr(_settings, "ADAPTIVE_RUNTIME_OPENED_FILES_MAX_CHARS", 16_000)
                    ),
                    step_index=runtime_step_index,
                )
                if runtime_queued and ctx.verbose:
                    suffix = " mid-step" if mid_step else ""
                    _log(
                        f"  {YELLOW}↪ runtime intervention queued{suffix}: "
                        f"{runtime_queued}{RESET}"
                    )
                if (
                    not mid_step
                    and getattr(
                        _settings,
                        "ADAPTIVE_RUNTIME_SEMANTIC_SHADOW_ENABLED",
                        False,
                    )
                ):
                    from infinidev.engine.behavior.runtime_policy import (
                        observe_semantic_behavior,
                    )

                    observe_semantic_behavior(ctx.state)
            if mid_step:
                self.inject_pending(ctx, messages)
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
