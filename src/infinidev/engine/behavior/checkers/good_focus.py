"""Promote behavior that advances the plan coherently."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class GoodFocusChecker(PromptBehaviorChecker):
    name = "good_focus"
    description = "Reward focused, on-plan progress with concrete actions"
    default_enabled = False
    delta_range = (0, 2)
    settings_message = "Good focus — rewards concrete progress and recovery from mistakes (0..+2)"

    criteria = (
        "Reward real, focused progress on the plan. Look for: a concrete tool "
        "call that advances the current step; a clear and accurate summary of "
        "work just done; recovering from a previous mistake with a smarter "
        "approach. Only return positive or zero deltas for this criterion."
    )
