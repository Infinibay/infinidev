"""Penalize repetitive thinking — same thoughts looping without action."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class RepetitiveThinkingChecker(PromptBehaviorChecker):
    name = "repetitive_thinking"
    description = "Penalize repeated thinking that restates the same ideas without acting"
    default_enabled = True
    delta_range = (-3, 0)
    settings_message = "Repetitive thinking — punishes loops that re-think instead of acting (-3..0)"

    criteria = (
        "Punish thinking loops. Compare the latest message's thinking/text "
        "content against the recent history. Penalize when: the agent re-states "
        "the same plan, hypothesis or doubt it already expressed without taking "
        "action; multiple consecutive turns are pure thinking with no tool calls; "
        "the agent paraphrases its previous reasoning instead of moving on; the "
        "agent keeps deliberating about a decision it already considered without "
        "new information. Do NOT punish if the latest message contains tool calls "
        "that act on the plan, or if there is genuine new information. Only "
        "return negative or zero deltas for this criterion."
    )
