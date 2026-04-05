"""Plan management tools — add, modify, remove steps from the execution plan."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class AddStepInput(BaseModel):
    title: str = Field(description="Short step title naming FILE, FUNCTION, and CHANGE")
    description: str = Field(default="", description="Detailed guidance (optional)")
    index: int = Field(default=0, description="Step number. 0 or omit to append at end of plan.")


class ModifyStepInput(BaseModel):
    index: int = Field(description="Step number to modify")
    title: str = Field(default="", description="New title (empty = keep current)")
    description: str = Field(default="", description="New description (empty = keep current)")


class RemoveStepInput(BaseModel):
    index: int = Field(description="Step number to remove")


class AddStepTool(InfinibayBaseTool):
    name: str = "add_step"
    description: str = (
        "Add a new step to the plan WITHOUT completing the current step. "
        "Use this when you discover new work mid-step. "
        "If index is 0 or omitted, the step is appended at the end of the plan."
    )
    args_schema: Type[BaseModel] = AddStepInput

    def _run(self, title: str, description: str = "", index: int = 0) -> str:
        from infinidev.tools.base.context import get_context_for_agent
        ctx = get_context_for_agent(self.agent_id)
        if not ctx or not hasattr(ctx, "loop_state") or ctx.loop_state is None:
            return self._error("No active plan context")

        plan = ctx.loop_state.plan
        from infinidev.engine.loop.step_operation import StepOperation

        if index <= 0:
            existing_max = max((s.index for s in plan.steps), default=0)
            index = existing_max + 1

        op = StepOperation(op="add", index=index, title=title, description=description)
        plan.apply_operations([op])
        return self._success({"status": "added", "index": index, "total_steps": len(plan.steps)})


class ModifyStepTool(InfinibayBaseTool):
    name: str = "modify_step"
    description: str = (
        "Modify the title or description of an existing pending step "
        "WITHOUT completing the current step."
    )
    args_schema: Type[BaseModel] = ModifyStepInput

    def _run(self, index: int, title: str = "", description: str = "") -> str:
        from infinidev.tools.base.context import get_context_for_agent
        ctx = get_context_for_agent(self.agent_id)
        if not ctx or not hasattr(ctx, "loop_state") or ctx.loop_state is None:
            return self._error("No active plan context")

        plan = ctx.loop_state.plan
        from infinidev.engine.loop.step_operation import StepOperation

        op = StepOperation(op="modify", index=index, title=title, description=description)
        plan.apply_operations([op])
        return self._success({"status": "modified", "index": index})


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

        op = StepOperation(op="remove", index=index)
        plan.apply_operations([op])
        return self._success({"status": "removed", "index": index})
