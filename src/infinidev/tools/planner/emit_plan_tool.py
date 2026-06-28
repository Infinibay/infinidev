"""EmitPlanTool — terminator used by the analyst planner.

Called exactly once per planner turn. The orchestrator parses the
tool_call args into a ``Plan`` and returns it to the pipeline. This
tool is NOT part of the developer's toolbox — it is exclusive to the
planner tier (registered under PLANNER_TOOLS in tools/__init__.py).
"""

import json
from typing import Literal, Type

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
            "How the step's success is checked. Prefer a DETERMINISTIC kind: "
            "'command' (a shell command that must exit 0), "
            "'test_id' (a pytest node id like tests/test_x.py::test_y), "
            "'file_contains' (a file must contain a substring), "
            "'symbol_exists' (a name/snippet must appear somewhere in the "
            "codebase). For a SOFT objective that no command can decide "
            "(readability, clearer error messages, duplication removed), use "
            "'llm_judge' — an independent reviewer judges verify_spec against "
            "the diff at task end. Use 'none' only when even that is "
            "impossible."
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
            "fragment that must also appear (empty = exit code 0 alone "
            "decides). For llm_judge: an optional hint of where to look "
            "(file/area). Ignored for the other kinds."
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
    acceptance_criteria: list[str] = Field(
        default_factory=list,
        description=(
            "Task-level 'done' conditions for the WHOLE task — each a "
            "short, FALSIFIABLE statement whose truth can be checked by "
            "running a command, reading a file, or inspecting behaviour "
            "(e.g. 'expired JWTs are rejected by validate_token', 'no "
            "references to legacy_verify() remain'). These are the accept "
            "gate the post-loop reviewer checks against — distinct from "
            "each step's own verify check. Avoid vague quality words "
            "('looks good', 'is clean'); they are dropped. 1-5 items."
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
