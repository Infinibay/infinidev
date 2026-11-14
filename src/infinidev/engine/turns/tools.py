"""Bound tools for explicit plan replacement and conversational close."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, PrivateAttr, field_validator

from infinidev.engine.turns.models import PlanItem, TurnState, validate_plan
from infinidev.tools.base.base_tool import BaseTool
from infinidev.tools.base.tool_effects import ToolEffects


class FinishTurnInput(BaseModel):
    """The literal user-facing message ending this conversation turn."""

    model_config = ConfigDict(extra="forbid")

    message: str

    @field_validator("message")
    @classmethod
    def _require_nonblank_message(cls, message: str) -> str:
        if not message.strip():
            raise ValueError("A finish_turn message must not be blank")
        return message


class UpdatePlanInput(BaseModel):
    """A complete replacement plan with explicit stable identifiers."""

    model_config = ConfigDict(extra="forbid")

    plan: list[PlanItem]

    _validate_plan = field_validator("plan")(validate_plan)


class FinishTurnTool(BaseTool):
    """Return control to the user without changing any work Task."""

    name: str = "finish_turn"
    description: str = (
        "End this conversation turn with a message to the user. "
        "This does not complete, block, or advance any work Task or plan item."
    )
    args_schema: type[BaseModel] = FinishTurnInput
    effects: ToolEffects = ToolEffects(communicates_user=True)
    _state: TurnState = PrivateAttr()

    def __init__(self, state: TurnState) -> None:
        super().__init__()
        self._state = state

    def _run(self, message: str) -> str:
        self._state.reply = message
        self._state.finished = True
        return message

    async def _arun(self, message: str) -> str:
        return self._run(message)


class UpdatePlanTool(BaseTool):
    """Atomically replace an explicitly supplied plan."""

    name: str = "update_plan"
    description: str = (
        "Replace the entire optional work plan. Supply stable unique item IDs and "
        "explicit statuses: pending, in_progress, completed, or blocked. At most one "
        "item may be in_progress. Titles are descriptive text, never execution policy."
    )
    args_schema: type[BaseModel] = UpdatePlanInput
    effects: ToolEffects = ToolEffects(mutates_internal_state=True)
    _state: TurnState = PrivateAttr()

    def __init__(self, state: TurnState) -> None:
        super().__init__()
        self._state = state

    def _run(self, plan: list[PlanItem]) -> str:
        # BaseTool dispatch dumps nested models to dicts; validate before assignment.
        replacement = UpdatePlanInput.model_validate({"plan": plan})
        self._state.plan = replacement.plan
        return "Plan updated."

    async def _arun(self, plan: list[PlanItem]) -> str:
        return self._run(plan)


def build_turn_tools(state: TurnState) -> list[BaseTool]:
    """Build fresh runtime-bindable tool instances sharing this turn's state."""
    return [FinishTurnTool(state), UpdatePlanTool(state)]
