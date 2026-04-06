"""Punish very long thinking on trivially simple tasks."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class ChattyThinkingChecker(PromptBehaviorChecker):
    name = "chatty_thinking"
    description = "Punish overly long thinking for trivial tasks"
    default_enabled = False
    delta_range = (-2, 0)
    settings_message = "Chatty thinking — punishes huge reasoning blobs on simple tasks (-2..0)"

    criteria = (
        "Punish over-long thinking on trivial work. Look at the "
        "latest_message's 'reasoning_chars' (length of reasoning_content) and "
        "compare to the task complexity (use 'task' and 'plan' fields):\n"
        "- -2 if reasoning_chars > 3000 AND the task is clearly trivial "
        "  (single read + single edit, a one-line fix, a rename, listing files).\n"
        "- -1 if reasoning_chars > 1500 AND the task is simple.\n"
        "- 0 if the task is complex enough to justify the thinking, or if "
        "  reasoning is short.\n"
        "Different from repetitive_thinking: this judges QUANTITY, not repetition.\n"
        "Never return positive deltas for this criterion."
    )
