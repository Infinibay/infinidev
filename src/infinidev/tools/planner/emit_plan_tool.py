"""Task-plan terminators used by the legacy and staged planners.

The orchestrator parses the tool-call arguments into a ``Plan``. The legacy
``emit_plan`` name remains available during the staged-planner migration;
``emit_task_plan`` is the canonical Task Planner terminator.
"""

import json
from typing import Literal, Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class PlanStepArg(BaseModel):
    title: str = Field(
        ...,
        description="Action-oriented Step title naming the observed target and change.",
    )
    detail: str = Field(
        "",
        description=(
            "Concrete execution guidance for this step: files to "
            "touch, changes to make, how to verify. Rendered ONLY "
            "when the Step is active, not for pending or done Steps."
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
    # A MACHINE-checkable version of the success criterion. When you can name
    # a concrete check, fill these — the engine RUNS it on step completion and
    # rejects the step (with the failure output) until it passes, so the
    # developer cannot self-declare a green check. Leave verify_kind 'none'
    # only when no command/file/test can decide the step (e.g. a pure
    # readability refactor).
    verify_kind: Literal[
        "none", "command", "test_id", "file_contains", "symbol_exists", "llm_judge"
    ] = Field(
        "none",
        description=(
            "How the step's success is checked. Choose in this order, and "
            "take the first that fits. A pytest node id you have seen in "
            "this repository, then 'test_id'. A shell command that exits 0 "
            "exactly when the step succeeded, then 'command'. A substring a "
            "named file must contain, then 'file_contains'. A string whose "
            "presence proves this step ran, then 'symbol_exists'. A sentence "
            "an independent reviewer checks against the diff at task end, "
            "then 'llm_judge'. Use 'none' when you cannot write that "
            "sentence. NEVER name a node id you have not seen: the engine "
            "runs it, and one that does not exist fails on every attempt."
        ),
    )
    verify_spec: str = Field(
        "",
        description=(
            "The thing to run/inspect/judge, per verify_kind: the command, the "
            "pytest node id, the file path (file_contains), the name/snippet "
            "(symbol_exists), or — for llm_judge — a precise acceptance "
            "statement a reviewer can check against the code (e.g. 'the three "
            "duplicated parse blocks in reader.py are replaced by one helper'). "
            "Required when verify_kind is not 'none'."
        ),
    )
    verify_observable: str = Field(
        "",
        description=(
            "The proof that means PASS. For file_contains: the REQUIRED "
            "substring (mandatory). For command/test_id: an optional stdout "
            "fragment that must also appear (empty means the exit code alone "
            "decides). For llm_judge: an optional hint of where to look "
            "(file or area). Ignored for symbol_exists and none."
        ),
    )


class EmitPlanInput(BaseModel):
    overview: str = Field(
        ...,
        description=(
            "Account of what will be done, why, which observed files are "
            "involved and how success will be verified. Shown to the user "
            "and rendered during the developer loop; keep Step-local detail "
            "inside each Step."
        ),
    )
    steps: list[PlanStepArg] = Field(
        ...,
        description=(
            "Initial execution route proposed by the planner. These steps "
            "carry model-inferred authority: the developer can add, revise, "
            "reorder or remove them when new evidence changes the tactic "
            "without changing the requested outcome."
        ),
    )
    # Required, not optional-with-a-default: these become the accept gate the
    # post-loop reviewer judges against, and an omission silently leaves
    # review_criteria None — a task that ends with nothing to check it against.
    acceptance_criteria: list[str] = Field(
        ...,
        description=(
            "Planner-derived checks for the task outcome. Each statement can "
            "be checked by running a command, reading a file or inspecting "
            "behavior. They guide review but cannot add requirements or "
            "authority to the user request. Avoid quality claims without an "
            "observable test; those claims are dropped at the parse boundary."
        ),
    )


class EmitTaskPlanInput(BaseModel):
    overview: str = Field(
        ...,
        description=(
            "Account of the Task outcome, the observed targets and the "
            "verification route. Shown to the user and rendered during "
            "execution; keep Step-local detail inside each Step."
        ),
    )
    derived_verification_criteria: list[str] = Field(
        ...,
        description=(
            "Falsifiable checks proposed by the Task Planner. They guide "
            "verification and remain model-derived; they cannot expand the "
            "Goal, Stage or Task."
        ),
    )
    steps: list[PlanStepArg] = Field(
        ...,
        min_length=1,
        description=(
            "Initial execution route for this Task. Split Steps where checks "
            "distinguish outcomes or one result feeds another Step. The "
            "developer can revise model-inferred tactics when new evidence "
            "changes the route while preserving the Task outcome."
        ),
    )


class EmitPlanTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "emit_plan"
    description: str = (
        "Emit the legacy execution plan and end the planning turn. "
        "Call this EXACTLY once: the turn ends on the first call and a "
        "second is never read. A plan with zero steps is rejected. Steps "
        "are model-inferred tactics and do not acquire user authority from "
        "this call."
    )
    args_schema: Type[BaseModel] = EmitPlanInput

    def _run(self, overview: str, steps: list, acceptance_criteria: list | None = None) -> str:
        # Like RespondTool/EscalateTool, this is a schema-level
        # terminator — the planner orchestrator reads the tool_call
        # args directly. This _run is the safe fallback. Under normal
        # dispatch `steps` arrives as PlanStepArg pydantic models, which
        # json.dumps can't serialize — coerce via model_dump so the
        # fallback never crashes.
        return json.dumps(
            {
                "kind": "plan",
                "overview": overview,
                "steps": steps,
                "acceptance_criteria": acceptance_criteria or [],
            },
            default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
        )


class EmitTaskPlanTool(InfinibayBaseTool):
    """Canonical Task Planner terminator."""

    is_read_only: bool = True
    name: str = "emit_task_plan"
    description: str = (
        "Emit the execution plan for the current Task and end the planning "
        "turn. Call this exactly once after each Step is grounded in the "
        "handoff, current evidence or an earlier Step's output. The first "
        "call ends the turn."
    )
    args_schema: Type[BaseModel] = EmitTaskPlanInput

    def _run(
        self,
        overview: str,
        derived_verification_criteria: list,
        steps: list,
    ) -> str:
        return json.dumps(
            {
                "kind": "task_plan",
                "overview": overview,
                "steps": steps,
                "derived_verification_criteria": derived_verification_criteria,
            },
            default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o),
        )
