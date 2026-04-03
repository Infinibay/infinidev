"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_operation import StepOperation


class LoopPlan(BaseModel):
    """The agent's mutable execution plan."""

    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def active_step(self) -> PlanStep | None:
        """Return the first step with status='active', or None."""
        for step in self.steps:
            if step.status == "active":
                return step
        return None

    @property
    def has_pending(self) -> bool:
        """True if any step is pending or active."""
        return any(s.status in ("pending", "active") for s in self.steps)

    def mark_active_done(self) -> None:
        """Mark the current active step as done (without activating next)."""
        for step in self.steps:
            if step.status == "active":
                step.status = "done"
                break

    def activate_next(self) -> None:
        """Activate the next pending step."""
        for step in self.steps:
            if step.status == "pending":
                step.status = "active"
                break

    def advance(self) -> None:
        """Mark the active step as done and activate the next pending step."""
        self.mark_active_done()
        self.activate_next()

    def apply_operations(self, ops: list[StepOperation]) -> None:
        """Apply structured add/modify/remove operations to the plan."""
        for op in ops:
            if op.op == "add":
                # Replace existing step at same index if present, but preserve done steps
                self.steps = [s for s in self.steps if s.index != op.index or s.status == "done"]
                # Only add if we actually removed the old step (i.e., it wasn't done)
                if not any(s.index == op.index and s.status == "done" for s in self.steps):
                    self.steps.append(PlanStep(index=op.index, description=op.description))

            elif op.op == "modify":
                for step in self.steps:
                    if step.index == op.index:
                        step.description = op.description
                        break

            elif op.op == "remove":
                for step in self.steps:
                    if step.index == op.index and step.status in ("pending", "active"):
                        step.status = "skipped"
                        break

        self.steps.sort(key=lambda s: s.index)

    def render(self) -> str:
        """Render the plan as a numbered list with status markers."""
        lines: list[str] = []
        for step in self.steps:
            tag = f"[{step.status}] " if step.status != "pending" else ""
            lines.append(f"{step.index}. {tag}{step.description}")
        return "\n".join(lines)


