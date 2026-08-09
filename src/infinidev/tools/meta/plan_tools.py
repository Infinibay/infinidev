"""Plan management tools — add, modify, remove steps from the execution plan."""

from __future__ import annotations

import re
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


# Regex sentinels for "concrete" step titles. A title is considered concrete
# when it names something locatable: a path, a file with extension, a function
# call (foo()), or a file:line reference. Titles that match none of these are
# vague ("Implement feature", "Fix the bug") and we surface a warning so the
# model can choose to refine via modify_step. Deliberately permissive — this
# is a nudge, not a gate.
_CONCRETE_HINTS = (
    re.compile(r"\.[a-zA-Z]{1,4}\b"),  # has a file extension
    re.compile(r"[/\\][\w./-]+"),  # has a path separator
    re.compile(r"\b\w+\([^)]*\)"),  # has a function call
    re.compile(r":\d+\b"),  # has a :line reference
)


def _looks_concrete(title: str) -> bool:
    return any(p.search(title) for p in _CONCRETE_HINTS)


class AddStepInput(BaseModel):
    title: str = Field(description="Short step title naming FILE, FUNCTION, and CHANGE")
    explanation: str = Field(
        default="",
        description="Detailed explanation: tools to use, approach, edge cases (optional)",
    )
    expected_output: str = Field(
        default="",
        description=(
            "Your own success criterion for this step — one short, verifiable sentence "
            "stating how you will know the step is done correctly. "
            "Examples: 'pytest tests/test_auth.py::test_expired_token passes', "
            "'src/auth.py:52 contains payload[\"exp\"] check', "
            "'I can name the entry point file and the persistence layer'."
        ),
    )
    index: int = Field(
        default=0, description="Step number. 0 or omit to append at end of plan."
    )
    before: int = Field(
        default=0,
        description=(
            "Insert immediately BEFORE this step number, shifting it and the "
            "steps after it down by one. Use when you discover a prerequisite "
            "that must run before work already on the plan. 0 or omit to place "
            "by index / append."
        ),
    )


class ModifyStepInput(BaseModel):
    index: int = Field(description="Step number to modify")
    title: str = Field(default="", description="New title (empty = keep current)")
    explanation: str = Field(
        default="", description="New explanation (empty = keep current)"
    )
    expected_output: str = Field(
        default="",
        description="New success criterion for this step (empty = keep current)",
    )


class RemoveStepInput(BaseModel):
    index: int = Field(description="Step number to remove")


# ── Execution-plan tools ─────────────────────────────────────────────────────


class AddStepTool(InfinibayBaseTool):
    name: str = "add_step"
    description: str = (
        "Add a new step to the plan WITHOUT completing the current step. "
        "Use this when you discover new work mid-step. "
        "If index is 0 or omitted, the step is appended at the end of the plan. "
        "Pass before=N to insert a prerequisite ahead of step N."
    )
    args_schema: Type[BaseModel] = AddStepInput

    def _run(
        self,
        title: str,
        explanation: str = "",
        expected_output: str = "",
        index: int = 0,
        before: int = 0,
    ) -> str:
        from infinidev.tools.base.context import get_context_for_agent

        ctx = get_context_for_agent(self.agent_id)
        if not ctx or not hasattr(ctx, "loop_state") or ctx.loop_state is None:
            return self._error("No active plan context")

        plan = ctx.loop_state.plan
        from infinidev.engine.loop.step_operation import StepOperation

        duplicate = plan.find_similar_open_step(title)
        if duplicate is not None:
            return self._success({
                "status": "duplicate",
                "existing_index": duplicate.index,
                "existing_title": duplicate.title,
                "message": (
                    "This work is already represented by an open step. "
                    "Continue that step or refine it with modify_step."
                ),
            })

        horizon_limit = max(0, int(getattr(plan, "rolling_horizon_limit", 0) or 0))
        open_steps = sum(
            step.status in ("pending", "active") for step in plan.steps
        )
        if horizon_limit and open_steps >= horizon_limit:
            return self._error(
                f"Rolling horizon already has {open_steps}/{horizon_limit} open steps. "
                "Execute, remove, or complete one before planning further ahead."
            )

        # ``before`` and ``index`` name the same slot — the step lands there and
        # whatever was pending at that number shifts down. They differ only in
        # what the model means by it, so the intent is kept for the error text.
        inserting = before > 0
        if inserting:
            index = before
        elif index <= 0:
            existing_max = max((s.index for s in plan.steps), default=0)
            index = existing_max + 1

        # Detect a real insertion by object identity: apply_operations silently
        # drops the add when the slot is held by a step whose index cannot move,
        # so a title check can't tell "added" from "rejected".
        before_ids = {id(s) for s in plan.steps}
        shifted = [s.index for s in plan.steps if s.index >= index and s.status == "pending"]
        op = StepOperation(
            op="add",
            index=index,
            title=title,
            explanation=explanation,
            expected_output=expected_output,
        )
        plan.apply_operations([op])
        added = next(
            (s for s in plan.steps if s.index == index and id(s) not in before_ids),
            None,
        )
        if added is None:
            occupant = next((s for s in plan.steps if s.index == index), None)
            held_by = f"a {occupant.status} step" if occupant else "a finished step"
            return self._error(
                f"Could not add step at index {index} — that slot, or a step "
                f"after it, is held by {held_by}. Its number is already "
                "referenced by archived work, so nothing can shift past it. "
                "Omit index and before (or pass 0) to append at the end instead."
            )
        # Bootstrap planning has no active Step. Keep new Steps pending until
        # step_complete closes that planning turn; StepManager then activates
        # the first one. Activating here would make the planning turn's
        # status="continue" incorrectly mark an unexecuted Step done.
        result: dict = {
            "status": "added",
            "index": index,
            "total_steps": len(plan.steps),
        }
        if shifted:
            result["shifted"] = (
                f"steps {min(shifted)}-{max(shifted)} moved down by one to make room"
            )
        from infinidev.engine.static_analysis_timer import measure

        with measure("plan_validate"):
            _vague = not _looks_concrete(title)
        if _vague:
            result["warning"] = (
                "Vague step title — name a file path, function(), or file:line so "
                "the step is locatable. You can refine it with modify_step."
            )
        if not expected_output.strip():
            result["hint"] = (
                "No expected_output set — define a short, verifiable success "
                "criterion now (or via modify_step) so the step has an explicit "
                "verification anchor."
            )
        return self._success(result)


class ModifyStepTool(InfinibayBaseTool):
    name: str = "modify_step"
    description: str = (
        "Modify the title or description of an existing pending step "
        "WITHOUT completing the current step."
    )
    args_schema: Type[BaseModel] = ModifyStepInput

    def _run(
        self,
        index: int,
        title: str = "",
        explanation: str = "",
        expected_output: str = "",
    ) -> str:
        from infinidev.tools.base.context import get_context_for_agent

        ctx = get_context_for_agent(self.agent_id)
        if not ctx or not hasattr(ctx, "loop_state") or ctx.loop_state is None:
            return self._error("No active plan context")

        plan = ctx.loop_state.plan
        from infinidev.engine.loop.loop_plan import APPROVED_MUTABLE_FIELDS
        from infinidev.engine.loop.step_operation import StepOperation

        # Pre-validate so the success report matches reality — apply_operations
        # silently no-ops a modify on a missing or finished step.
        target = next((s for s in plan.steps if s.index == index), None)
        if target is None:
            return self._error(f"No step with index {index} in the plan")
        if target.status in ("done", "skipped", "blocked"):
            return self._error(
                f"Step {index} is {target.status} and cannot be modified"
            )

        requested = {
            "title": title,
            "explanation": explanation,
            "expected_output": expected_output,
        }
        requested = {k: v for k, v in requested.items() if v}
        if not requested:
            return self._error(
                "Nothing to change — pass at least one of title, explanation "
                "or expected_output."
            )
        # An approved step can be refined but not repurposed. Today every field
        # this tool accepts is refinable, so `refused` is empty; it is computed
        # rather than assumed so that widening StepOperation cannot quietly
        # start reporting changes that apply_operations then drops.
        refused = (
            [k for k in requested if k not in APPROVED_MUTABLE_FIELDS]
            if target.user_approved
            else []
        )

        op = StepOperation(
            op="modify",
            index=index,
            title=title,
            explanation=explanation,
            expected_output=expected_output,
        )
        plan.apply_operations([op])
        result: dict = {
            "status": "modified",
            "index": index,
            "applied": [k for k in requested if k not in refused],
        }
        if refused:
            result["refused"] = refused
            result["note"] = (
                f"Step {index} came from the approved plan: its wording can be "
                "refined, but these fields are fixed."
            )
        return self._success(result)


class RemoveStepTool(InfinibayBaseTool):
    name: str = "remove_step"
    description: str = (
        "Remove a pending step from the plan WITHOUT completing the current step."
    )
    args_schema: Type[BaseModel] = RemoveStepInput

    def _run(self, index: int) -> str:
        from infinidev.tools.base.context import get_context_for_agent

        ctx = get_context_for_agent(self.agent_id)
        if not ctx or not hasattr(ctx, "loop_state") or ctx.loop_state is None:
            return self._error("No active plan context")

        plan = ctx.loop_state.plan
        from infinidev.engine.loop.step_operation import StepOperation

        # Pre-validate so "removed" can't be reported when apply_operations
        # actually no-ops (missing/user-approved/already-finished index, or the
        # bulk-removal guard refusing to drop the only pending step).
        target = next((s for s in plan.steps if s.index == index), None)
        if target is None:
            return self._error(f"No step with index {index} in the plan")
        if target.user_approved:
            return self._error(f"Step {index} is user-approved and cannot be removed")
        if target.status not in ("pending", "active"):
            return self._error(f"Step {index} is {target.status}; nothing to remove")
        pending = [s for s in plan.steps if s.status in ("pending", "active")]
        if len(pending) <= 1:
            return self._error(
                "Refusing to remove the only remaining pending step — add a "
                "replacement step first or mark it done via step_complete"
            )

        op = StepOperation(op="remove", index=index)
        plan.apply_operations([op])
        return self._success({"status": "removed", "index": index})
