"""Bidirectional: reward concrete plans, punish vague or bloated ones."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class PlanQualityChecker(PromptBehaviorChecker):
    name = "plan_quality"
    description = "Reward concrete plans (+1..+2); punish vague or bloated ones (-2..-1)"
    default_enabled = True
    delta_range = (-2, 2)
    settings_message = "Plan quality — bidirectional: rewards concrete plans, punishes vague/bloated (-2..+2)"

    criteria = (
        "Evaluate plan quality. ONLY apply this when the latest_message is "
        "creating or modifying the plan (e.g., the tool_calls include "
        "create_plan, add_plan_step, modify_plan_step, or step_complete with "
        "plan modifications). Otherwise return 0.\n"
        "Inspect the resulting plan in 'plan.steps':\n"
        "- +2 if the plan has 2-4 concrete, actionable steps that mention "
        "  specific files / functions / behaviors and are clearly aligned "
        "  with the task.\n"
        "- +1 if the plan is decent but a step is slightly vague.\n"
        "- 0 if there is no plan change in this message.\n"
        "- -1 if the plan has vague steps like 'explore the codebase', "
        "  'understand the problem', 'fix the issue'.\n"
        "- -2 if the plan has 7+ steps with redundancy, or is so vague it "
        "  provides no guidance.\n"
        "This is the only checker that returns BOTH positive and negative deltas."
    )
