"""Registry of available behavior checkers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.engine.behavior.checker_base import BehaviorChecker


def _all_checker_classes() -> list[type["BehaviorChecker"]]:
    # Lazy import to avoid circular deps at package init.
    from infinidev.engine.behavior.checkers.lazy_work import LazyWorkChecker
    from infinidev.engine.behavior.checkers.good_focus import GoodFocusChecker
    from infinidev.engine.behavior.checkers.repetitive_thinking import (
        RepetitiveThinkingChecker,
    )
    from infinidev.engine.behavior.checkers.graceful_recovery import (
        GracefulRecoveryChecker,
    )
    from infinidev.engine.behavior.checkers.small_safe_edits import (
        SmallSafeEditsChecker,
    )
    from infinidev.engine.behavior.checkers.ignores_tool_error import (
        IgnoresToolErrorChecker,
    )
    from infinidev.engine.behavior.checkers.shell_when_tool_exists import (
        ShellWhenToolExistsChecker,
    )
    from infinidev.engine.behavior.checkers.plan_drift import PlanDriftChecker
    from infinidev.engine.behavior.checkers.chatty_thinking import (
        ChattyThinkingChecker,
    )
    from infinidev.engine.behavior.checkers.fake_completion import (
        FakeCompletionChecker,
    )
    from infinidev.engine.behavior.checkers.prompt_pollution import (
        PromptPollutionChecker,
    )
    from infinidev.engine.behavior.checkers.plan_quality import PlanQualityChecker

    return [
        LazyWorkChecker,
        GoodFocusChecker,
        RepetitiveThinkingChecker,
        GracefulRecoveryChecker,
        SmallSafeEditsChecker,
        IgnoresToolErrorChecker,
        ShellWhenToolExistsChecker,
        PlanDriftChecker,
        ChattyThinkingChecker,
        FakeCompletionChecker,
        PromptPollutionChecker,
        PlanQualityChecker,
    ]


# Mapping: checker name → settings flat-key on the Settings model.
# Adding a new checker = add the class above + the bool field in
# config/settings.py and an entry here.
CHECKER_SETTING_KEYS: dict[str, str] = {
    "lazy_work": "BEHAVIOR_CHECKER_LAZY_WORK",
    "good_focus": "BEHAVIOR_CHECKER_GOOD_FOCUS",
    "repetitive_thinking": "BEHAVIOR_CHECKER_REPETITIVE_THINKING",
    "graceful_recovery": "BEHAVIOR_CHECKER_GRACEFUL_RECOVERY",
    "small_safe_edits": "BEHAVIOR_CHECKER_SMALL_SAFE_EDITS",
    "ignores_tool_error": "BEHAVIOR_CHECKER_IGNORES_TOOL_ERROR",
    "shell_when_tool_exists": "BEHAVIOR_CHECKER_SHELL_WHEN_TOOL_EXISTS",
    "plan_drift": "BEHAVIOR_CHECKER_PLAN_DRIFT",
    "chatty_thinking": "BEHAVIOR_CHECKER_CHATTY_THINKING",
    "fake_completion": "BEHAVIOR_CHECKER_FAKE_COMPLETION",
    "prompt_pollution": "BEHAVIOR_CHECKER_PROMPT_POLLUTION",
    "plan_quality": "BEHAVIOR_CHECKER_PLAN_QUALITY",
}


def all_checkers() -> list["BehaviorChecker"]:
    """Instantiate every registered checker (regardless of enabled state)."""
    return [cls() for cls in _all_checker_classes()]


def enabled_checkers() -> list["BehaviorChecker"]:
    """Instantiate only the checkers enabled in current settings."""
    from infinidev.config.settings import settings

    if not getattr(settings, "BEHAVIOR_CHECKERS_ENABLED", False):
        return []

    out: list[BehaviorChecker] = []
    for cls in _all_checker_classes():
        key = CHECKER_SETTING_KEYS.get(cls.name)
        if key is None:
            continue
        if bool(getattr(settings, key, cls.default_enabled)):
            out.append(cls())
    return out
