"""Punish meta-instructional filler in the agent's messages."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class PromptPollutionChecker(PromptBehaviorChecker):
    name = "prompt_pollution"
    description = "Punish meta filler like 'As an AI...' / 'I will now...'"
    default_enabled = False
    delta_range = (-1, 0)
    settings_message = "Prompt pollution — punishes meta-instructional filler tokens (-1..0)"

    criteria = (
        "Punish prompt pollution: meta-instructional filler that wastes tokens "
        "and adds no information. Look at the latest_message's content for "
        "phrases like:\n"
        "  'As an AI...', 'I am an AI assistant...', 'I will now proceed to...',\n"
        "  'Let me think step by step...', 'I understand your request...',\n"
        "  'Sure, here's what I'll do...', 'In conclusion...', long restatements\n"
        "  of the user's request before acting.\n"
        "Scoring:\n"
        "- -1 if the message contains one or more such filler phrases.\n"
        "- 0 if the message is direct and to the point.\n"
        "Never return positive deltas for this criterion."
    )
