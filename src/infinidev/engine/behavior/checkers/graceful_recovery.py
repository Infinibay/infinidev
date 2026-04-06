"""Reward changing approach after a tool failure (vs blindly retrying)."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class GracefulRecoveryChecker(PromptBehaviorChecker):
    name = "graceful_recovery"
    description = "Reward switching strategy after a tool error instead of repeating it"
    default_enabled = True
    delta_range = (0, 2)
    settings_message = "Graceful recovery — rewards changing approach after a failed tool call (0..+2)"

    criteria = (
        "Reward graceful recovery from failures. Inspect the recent_history "
        "for any tool result that contained an error (role=\"tool\" with error "
        "text, or any message indicating a failed tool call). Then check the "
        "latest_message:\n"
        "- +2 if the agent clearly switched to a DIFFERENT tool or a "
        "  meaningfully different approach in response to the error.\n"
        "- +1 if the agent acknowledged the error and tried a small variation "
        "  that has a real chance of succeeding (different args, different file).\n"
        "- 0 if there was no recent error, or if the agent retried the exact "
        "  same call with the same arguments.\n"
        "Never return negative deltas for this criterion."
    )
