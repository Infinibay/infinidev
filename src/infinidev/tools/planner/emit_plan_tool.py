"""EmitPlanTool — terminator used by the analyst planner.

Called exactly once per planner turn. The orchestrator parses the
tool_call args into a ``Plan`` and returns it to the pipeline. This
tool is NOT part of the developer's toolbox — it is exclusive to the
planner tier (registered under PLANNER_TOOLS in tools/__init__.py).
"""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class PlanStepArg(BaseModel):
    title: str = Field(
        ...,
        description="Short, action-oriented step title (5-10 words).",
    )
    detail: str = Field(
        "",
        description=(
            "Concrete execution guidance for this step: files to "
            "touch, changes to make, how to verify. Rendered ONLY "
            "when the step is active, not for pending or done steps, "
            "so context stays compact. Aim for 2-5 sentences."
        ),
    )
    expected_output: str = Field(
        "",
        description=(
            "Verifiable success criterion — what is true after the "
            "step completes. Example: 'tests/test_auth.py passes' or "
            "'validate_token returns None for expired tokens'."
        ),
    )


class EmitPlanInput(BaseModel):
    overview: str = Field(
        ...,
        description=(
            "1-2 paragraph prose narrative: what will be done, why, "
            "which files are involved, how success will be verified. "
            "Shown to the user and rendered every iteration of the "
            "developer loop as <plan-overview>, so keep it compact — "
            "around 150-300 tokens."
        ),
    )
    steps: list[PlanStepArg] = Field(
        ...,
        description=(
            "Ordered list of execution steps. Each step becomes a "
            "user-approved PlanStep that the developer executes. The "
            "developer can add new steps mid-execution but cannot "
            "remove or modify these."
        ),
    )


class EmitPlanTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "emit_plan"
    description: str = (
        "Emit the final execution plan and end the planning turn. "
        "Call this EXACTLY once, after you have enough information to "
        "break the work into concrete steps. Do not emit an empty plan "
        "or a single-step plan for non-trivial work. The developer "
        "will execute your steps in order without re-asking the user."
    )
    args_schema: Type[BaseModel] = EmitPlanInput

    def _run(self, overview: str, steps: list) -> str:
        # Like RespondTool/EscalateTool, this is a schema-level
        # terminator — the planner orchestrator reads the tool_call
        # args directly. This _run is the safe fallback.
        return json.dumps({"kind": "plan", "overview": overview, "steps": steps})
