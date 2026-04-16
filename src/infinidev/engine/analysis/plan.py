"""Plan — the structured artifact produced by the analyst planner.

Consumed by LoopEngine.execute(initial_plan=plan) to seed the
developer's LoopState with a pre-approved execution plan. Replaces
the legacy AnalysisResult.specification dict as the single handoff
shape between analyst and developer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanStepSpec:
    """One step in a planner-emitted plan.

    The fields map directly to loop.plan_step.PlanStep: ``title`` goes
    to PlanStep.title, ``detail`` to PlanStep.detail, and
    ``expected_output`` to PlanStep.expected_output. Keeping this as a
    separate frozen dataclass (rather than reusing PlanStep) makes the
    handoff boundary explicit: the planner does not produce mutable
    LoopState objects.
    """

    title: str
    detail: str = ""
    expected_output: str = ""


@dataclass(frozen=True)
class Plan:
    """Planner output: prose narrative plus ordered step specs.

    Attributes:
        overview: 1-2 paragraph prose narrative — what, why, which
            files, validation approach. Shown to the user via
            ``notify("Planner", plan.overview)`` before execution
            begins, and rendered every iteration as ``<plan-overview>``
            so the developer always has the big picture.
        steps: Ordered list of step specs. Each becomes a user-approved
            PlanStep in LoopState; the LLM cannot remove or modify
            them.
    """

    overview: str
    steps: list[PlanStepSpec] = field(default_factory=list)
