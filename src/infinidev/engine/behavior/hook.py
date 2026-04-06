"""Wire BehaviorScorer into the POST_MODEL_MESSAGE hook event."""

from __future__ import annotations

from infinidev.engine.behavior.scorer import BehaviorScorer
from infinidev.engine.hooks.hooks import hook_manager, HookEvent, HookContext

_registered = False


def _on_post_model_message(ctx: HookContext) -> None:
    BehaviorScorer.instance().on_model_message(ctx)


def register_behavior_hooks() -> None:
    """Register the behavior scorer. Safe to call multiple times."""
    global _registered
    if _registered:
        return
    _registered = True
    hook_manager.register(
        HookEvent.POST_MODEL_MESSAGE,
        _on_post_model_message,
        priority=200,
        name="behavior:scorer",
    )


def unregister_behavior_hooks() -> None:
    global _registered
    if not _registered:
        return
    _registered = False
    hook_manager.unregister(HookEvent.POST_MODEL_MESSAGE, _on_post_model_message)
