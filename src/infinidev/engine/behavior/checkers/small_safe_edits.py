"""Reward small, scoped edits over massive rewrites."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class SmallSafeEditsChecker(PromptBehaviorChecker):
    name = "small_safe_edits"
    description = "Reward scoped edit_symbol / replace_lines vs full-file rewrites"
    default_enabled = True
    delta_range = (0, 2)
    settings_message = "Small safe edits — rewards scoped edit_symbol / replace_lines (0..+2)"

    criteria = (
        "Reward small, surgical edits. Look at the tool_calls in the "
        "latest_message and their args:\n"
        "- +2 if the agent used edit_symbol, add_symbol, remove_symbol, or "
        "  replace_lines with a SMALL line range (under ~30 lines) on an "
        "  existing file.\n"
        "- +1 if the agent used create_file for a genuinely new, small file "
        "  (under ~80 lines) that the task actually requires.\n"
        "- 0 if no edit happened, or if the edit is appropriately sized but "
        "  unremarkable.\n"
        "- 0 (NOT negative) if the edit is large — other checkers handle that.\n"
        "Never return negative deltas for this criterion."
    )
