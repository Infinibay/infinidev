"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from infinidev.engine.analysis.step_verification import StepVerification


class PlanStep(BaseModel):
    """A single step in the agent's execution plan."""

    index: int
    title: str
    explanation: str = ""
    # Self-defined success criterion: a short, verifiable statement the model
    # commits to before running the step. Used to render <expected-output> in
    # the iteration prompt and as the post-step verification anchor. Empty if
    # the model didn't declare one (older flows / quick steps).
    expected_output: str = ""
    # Long-form guidance written upfront by the planner: exact files, changes,
    # verification approach. Rendered ONLY while the step is active to keep
    # the iteration prompt small — pending steps show their title only.
    detail: str = ""
    # True when this step came from an analyst-emitted plan. The user is shown
    # the plan's prose overview, not the step list, so this marks "part of the
    # scope the run committed to", not "the user read this line". It is what
    # StepCompleteGate counts to refuse a close that would abandon scope, and
    # what apply_operations checks before allowing a removal. Refinement of the
    # wording is allowed — see loop_plan.APPROVED_MUTABLE_FIELDS.
    user_approved: bool = False
    # What the step established, written when it closes. Distinct from the
    # summary in ActionRecord: that one narrates the work, this one states the
    # outcome that later steps depend on.
    conclusion: str = ""
    # Working-memory record titles for this step's evidence. They are the exact
    # strings ``working_memory._format_call`` stored, which makes them valid
    # ``recall_context`` queries: the plan block doubles as an index into the
    # archive instead of duplicating the history block.
    evidence: list[str] = Field(default_factory=list)
    # Machine-checkable success condition authored by the planner. When
    # present and executable, the engine runs it on step_complete and
    # blocks closure until it passes (see ObjectiveVerifier + the gate in
    # LoopEngine._objective_gate_blocks). None / kind 'none' falls back to
    # the self-attested ``expected_output``.
    verify: StepVerification | None = None
    # ``blocked`` is a real outcome, not a variant of done: the step was
    # attempted and could not be finished. Recording it as ``done`` told the
    # reviewer the work had succeeded. It is terminal, so ``has_pending`` is
    # False once every step is blocked — the scope gate in the engine's
    # idle-completion branch is what keeps that from closing the run quietly.
    status: Literal["pending", "active", "done", "skipped", "blocked"] = "pending"


