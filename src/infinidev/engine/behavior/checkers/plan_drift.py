"""Punish working on something unrelated to the active plan step."""

from infinidev.engine.behavior.checker_base import PromptBehaviorChecker


class PlanDriftChecker(PromptBehaviorChecker):
    name = "plan_drift"
    description = "Punish working on something unrelated to the active plan step"
    default_enabled = True
    delta_range = (-2, 0)
    settings_message = "Plan drift — punishes wandering off the active step without justification (-2..0)"

    criteria = (
        "Punish drifting away from the active plan step. Use the 'plan' field "
        "(active_step_title + steps) and the 'task' field as the source of "
        "truth for what the agent SHOULD be doing. Then look at the "
        "latest_message's tool_calls and content:\n"
        "- -2 if the agent is operating on files / modules clearly unrelated "
        "  to the active step or to the task, with no justification.\n"
        "- -1 if the agent is partially off-track (related but tangential).\n"
        "- 0 if the agent is on-task, OR if there is no plan yet (planning "
        "  phase), OR if the agent justifies the detour in its message.\n"
        "Never return positive deltas for this criterion."
    )
