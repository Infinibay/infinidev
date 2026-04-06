"""Penalize evasive / lazy / non-committal model behavior."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class LazyWorkChecker(PromptBehaviorChecker):
    name = "lazy_work"
    description = "Penalize evasive answers, TODO placeholders, or refusal to do real work"
    default_enabled = True
    delta_range = (-3, 0)
    settings_message = "Lazy/evasive work — punishes TODOs, vague summaries, repeated failures (-3..0)"

    criteria = (
        "Punish lazy or evasive work. Look for: empty TODOs / placeholder code "
        "instead of real implementation; vague summaries that hide that no work "
        "was done; repeating the same failed action without changing approach; "
        "refusing to act when the task is clearly within scope. Only return "
        "negative or zero deltas for this criterion."
    )
