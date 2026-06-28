"""Plan — the structured artifact produced by the analyst planner.

Consumed by LoopEngine.execute(initial_plan=plan) to seed the
developer's LoopState with a pre-approved execution plan. Replaces
the legacy AnalysisResult.specification dict as the single handoff
shape between analyst and developer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from infinidev.engine.analysis.step_verification import StepVerification


@dataclass(frozen=True)
class PlanStepSpec:
    """One step in a planner-emitted plan.

    The fields map directly to loop.plan_step.PlanStep: ``title`` goes
    to PlanStep.title, ``detail`` to PlanStep.detail, ``expected_output``
    to PlanStep.expected_output, and ``verify`` to PlanStep.verify.
    Keeping this as a separate frozen dataclass (rather than reusing
    PlanStep) makes the handoff boundary explicit: the planner does not
    produce mutable LoopState objects.

    ``verify`` is the planner-authored, machine-checkable success
    condition. Authoring it here — read-only, before any code exists —
    keeps the success bar adversarial (the developer cannot back-rationalise
    a check against its own diff) and frozen (planner steps are
    user_approved, so the developer cannot relax it mid-run).
    """

    title: str
    detail: str = ""
    expected_output: str = ""
    verify: StepVerification | None = None


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
    # Task-level, falsifiable "done" conditions for the WHOLE task (distinct
    # from per-step ``verify`` checks). Authored by the planner, they become
    # the real Task.acceptance_criteria (replacing the synthesised
    # placeholder) and are fed to the post-loop reviewer as the accept gate.
    acceptance_criteria: list[str] = field(default_factory=list)
