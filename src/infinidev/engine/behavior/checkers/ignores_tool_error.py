"""Punish ignoring a tool error in the very next message."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class IgnoresToolErrorChecker(PromptBehaviorChecker):
    name = "ignores_tool_error"
    description = "Punish silently ignoring a tool failure"
    default_enabled = True
    delta_range = (-3, 0)
    settings_message = "Ignores tool error — punishes carrying on as if a failed tool call worked (-3..0)"

    criteria = (
        "Punish ignoring tool errors. Inspect recent_history for the most "
        "recent tool result. If it contains an error (the content has words "
        "like 'error', 'failed', 'not found', 'permission denied', a "
        "traceback, or starts with 'x ' / 'Error:'), check the latest_message:\n"
        "- -3 if the agent completely ignores the error and proceeds as if it "
        "  succeeded (e.g., calls step_complete with status=done, or moves on "
        "  to an unrelated action).\n"
        "- -2 if the agent acknowledges nothing about the failure even though "
        "  the next action depends on the failed call's result.\n"
        "- -1 if the agent vaguely references the failure but doesn't address it.\n"
        "- 0 if there was no recent error, or the agent properly handled it.\n"
        "Never return positive deltas for this criterion."
    )
