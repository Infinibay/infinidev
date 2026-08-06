"""Plan — the structured artifact produced by the analyst planner.

Consumed by LoopEngine.execute(initial_plan=plan) to seed the
developer's LoopState with a provenance-labelled execution plan. Replaces
the legacy AnalysisResult.specification dict as the single handoff
shape between analyst and developer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.authority import AuthorityLevel


@dataclass(frozen=True)
class PlanStepSpec:
    """One step in a planner-emitted plan.

    The fields map directly to loop.plan_step.PlanStep: ``title`` goes
    to PlanStep.title, ``detail`` to PlanStep.detail, ``expected_output``
    to PlanStep.expected_output, and ``verify`` to PlanStep.verify.
    Keeping this as a separate frozen dataclass (rather than reusing
    PlanStep) makes the handoff boundary explicit: the planner does not
    produce mutable LoopState objects.

    ``authority`` records who actually authorized the step. Planner-created
    steps default to ``model_inferred``; merely showing the plan overview is
    not user confirmation. Callers may use ``user_explicit`` or
    ``user_confirmed`` only when they have corresponding evidence.

    ``verify`` is the planner-authored success check. Authoring it before code
    exists gives review an independent proposal, but it remains untrusted
    model output and carries the step's authority. Runtime permission policy
    still governs executable checks.
    """

    title: str
    detail: str = ""
    expected_output: str = ""
    verify: StepVerification | None = None
    authority: AuthorityLevel = "model_inferred"


@dataclass(frozen=True)
class Plan:
    """Planner output: prose narrative plus ordered step specs.

    Attributes:
        overview: 1-2 paragraph prose narrative — what, why, which
            files, validation approach. Shown to the user via
            ``notify("Planner", plan.overview)`` before execution
            begins, and rendered every iteration as ``<plan-overview>``
            so the developer always has the big picture.
        steps: Ordered list of provenance-labelled step specs. Only steps
            carrying direct or confirmed user authority receive the loop's
            user-approved scope protections.
    """

    overview: str
    steps: list[PlanStepSpec] = field(default_factory=list)
    # Compatibility name for planner-derived Task checks (distinct from each
    # Step's ``verify``). The pipeline places them in
    # Task.derived_verification_criteria; they never become user-authored
    # acceptance requirements merely because the legacy field says
    # ``acceptance_criteria``.
    acceptance_criteria: list[str] = field(default_factory=list)
    acceptance_criteria_authority: AuthorityLevel = "model_inferred"
