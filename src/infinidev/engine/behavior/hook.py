"""Wire BehaviorScorer into the right hook event based on settings.

``BEHAVIOR_CHECK_MODE`` picks the firing frequency:
  - "per_step"    (default) → one evaluation per completed step (POST_STEP)
  - "per_message" (legacy)  → evaluate after every model message
"""

from __future__ import annotations

from infinidev.engine.behavior.scorer import BehaviorScorer
from infinidev.engine.hooks.hooks import hook_manager, HookEvent, HookContext

_registered_events: list[tuple[HookEvent, object]] = []


def _on_post_model_message(ctx: HookContext) -> None:
    BehaviorScorer.instance().on_model_message(ctx)


def _on_post_step(ctx: HookContext) -> None:
    BehaviorScorer.instance().on_step(ctx)


def _current_mode() -> str:
    try:
        from infinidev.config.settings import settings

        return str(getattr(settings, "BEHAVIOR_CHECK_MODE", "per_step")).lower()
    except Exception:
        return "per_step"


def register_behavior_hooks() -> None:
    """Register the behavior scorer. Safe to call multiple times.

    If called after a settings reload with a different mode, the previous
    registration is cleared first. Honors ``BEHAVIOR_CHECKERS_ENABLED`` —
    when disabled, no hook is registered at all so the loop pays zero
    dispatch overhead (previously the scorer was always registered and
    only the inner work was guarded, which made the flag misleading).
    """
    global _registered_events
    if _registered_events:
        # Re-registration only matters if the mode changed — clear either way.
        unregister_behavior_hooks()

    try:
        from infinidev.config.settings import settings
        if not getattr(settings, "BEHAVIOR_CHECKERS_ENABLED", False):
            return
    except Exception:
        pass

    mode = _current_mode()
    if mode == "per_message":
        hook_manager.register(
            HookEvent.POST_MODEL_MESSAGE,
            _on_post_model_message,
            priority=200,
            name="behavior:scorer:msg",
        )
        _registered_events.append(
            (HookEvent.POST_MODEL_MESSAGE, _on_post_model_message)
        )
    else:
        hook_manager.register(
            HookEvent.POST_STEP,
            _on_post_step,
            priority=200,
            name="behavior:scorer:step",
        )
        _registered_events.append((HookEvent.POST_STEP, _on_post_step))


def unregister_behavior_hooks() -> None:
    global _registered_events
    for event, fn in list(_registered_events):
        try:
            hook_manager.unregister(event, fn)
        except Exception:
            pass
    _registered_events = []
