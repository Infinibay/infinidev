"""Moderator-tier council terminators.

The moderator orchestrates the council in three distinct calls, each a
schema-level terminator the runner parses directly:

  * ``seed_council``     — assign personas + objectives and open threads.
  * ``council_verdict``  — judge, each round, whether the debate converged.
  * ``synthesize_brief`` — emit the final DesignBrief.
"""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


# ── seed_council ─────────────────────────────────────────────────────────


class MemberSpec(BaseModel):
    member_id: str = Field(
        ...,
        description=(
            "Short kebab-case id for this member, descriptive of its role "
            "(e.g. 'advocate-mvp', 'concurrency-skeptic', 'pg-expert')."
        ),
    )
    persona: str = Field(
        ...,
        description=(
            "HOW this member thinks — its stable stance, bias, and "
            "priorities. E.g. 'A skeptic who assumes every proposal hides "
            "a concurrency bug and demands proof it doesn't.' Make personas "
            "genuinely diverse and partly in tension with each other — that "
            "tension is what makes the debate worth more than one agent."
        ),
    )
    objective: str = Field(
        ...,
        description=(
            "WHAT this member must achieve in THIS debate — a concrete "
            "target, not a generic 'help solve it'. E.g. 'Find the failure "
            "mode of the proposed cache under parallel writes.'"
        ),
    )
    seed_tools: list[str] = Field(
        default_factory=list,
        description=(
            "Optional read-only tools you suggest this member lean on "
            "(e.g. ['code_search', 'read_file', 'web_search'])."
        ),
    )


class OpeningThreadSpec(BaseModel):
    title: str = Field(..., description="Short topic title for the thread.")
    prompt: str = Field(
        ...,
        description="The opening message seeding the thread's discussion.",
    )


class SeedCouncilInput(BaseModel):
    question: str = Field(
        ...,
        description=(
            "The single, sharp question the council will debate, in the "
            "user's language. Frame it as a design/research decision, not "
            "an implementation task."
        ),
    )
    members: list[MemberSpec] = Field(
        ...,
        description=(
            "3-5 council members. Each gets its own persona, objective, "
            "and isolated context. Diversity of perspective is the whole "
            "point — avoid near-duplicate members."
        ),
    )
    opening_threads: list[OpeningThreadSpec] = Field(
        ...,
        description="1-3 threads to open the debate with.",
    )


class SeedCouncilTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "seed_council"
    description: str = (
        "Seed the council: assign each subagent its persona, role and "
        "objective, and open the initial discussion threads. Call this "
        "EXACTLY once to start the deliberation."
    )
    args_schema: Type[BaseModel] = SeedCouncilInput

    def _run(self, question: str, members: list, opening_threads: list) -> str:
        # Schema-level terminator; the orchestrator reads tool_call args
        # directly. Coerce nested pydantic models so this fallback _run never
        # crashes on json.dumps if it is ever dispatched normally.
        return json.dumps({
            "kind": "seed_council",
            "question": question,
            "members": members,
            "opening_threads": opening_threads,
        }, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o))


# ── council_verdict ──────────────────────────────────────────────────────


class CouncilVerdictInput(BaseModel):
    converged: bool = Field(
        ...,
        description=(
            "True if the debate has reached a workable consensus (or is "
            "merely repeating itself and further rounds won't help). False "
            "if another round would still produce new, useful argument."
        ),
    )
    reason: str = Field(
        "",
        description="One sentence justifying the verdict (for the audit log).",
    )


class CouncilVerdictTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "council_verdict"
    description: str = (
        "Judge whether the council has converged after the latest round. "
        "Call EXACTLY once. Return converged=true to stop and synthesise, "
        "false to run another round."
    )
    args_schema: Type[BaseModel] = CouncilVerdictInput

    def _run(self, converged: bool, reason: str = "") -> str:
        return json.dumps({
            "kind": "council_verdict", "converged": converged, "reason": reason,
        })


# ── synthesize_brief ─────────────────────────────────────────────────────


class AlternativeSpec(BaseModel):
    approach: str = Field(..., description="The alternative approach.")
    why_rejected: str = Field(..., description="Why the council set it aside.")


class SynthesizeBriefInput(BaseModel):
    chosen_approach: str = Field(
        ...,
        description=(
            "The recommended approach, synthesised from the debate, in the "
            "user's language. This is the council's answer."
        ),
    )
    rationale: str = Field("", description="Why this approach won.")
    alternatives_considered: list[AlternativeSpec] = Field(
        default_factory=list,
        description="Approaches weighed and set aside, with reasons.",
    )
    open_risks: list[str] = Field(
        default_factory=list,
        description="Risks the developer should watch for.",
    )
    research_findings: list[str] = Field(
        default_factory=list,
        description="Grounded facts surfaced during debate (cite files/symbols).",
    )
    affected_files: list[str] = Field(
        default_factory=list,
        description="Files the work is expected to touch.",
    )
    dissent: list[str] = Field(
        default_factory=list,
        description="Minority positions worth flagging.",
    )
    user_decision_required: bool = Field(
        False,
        description=(
            "Set True ONLY if the council hit a genuine PRODUCT/DESIGN fork "
            "that the model must not decide alone (e.g. 'optimise for latency "
            "or for cost?'). Resolve purely technical questions yourself — "
            "do NOT punt those to the user. Dissent is the best signal that a "
            "real user-facing choice exists."
        ),
    )
    open_questions_for_user: list[str] = Field(
        default_factory=list,
        description=(
            "The concrete questions to ask the user, in their language. "
            "Required when user_decision_required is True, empty otherwise."
        ),
    )


class SynthesizeBriefTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "synthesize_brief"
    description: str = (
        "Emit the final design brief synthesising the whole debate, and "
        "end the council. Call EXACTLY once. Fold in the strongest points "
        "from every member, name the dissent honestly, and only flag "
        "user_decision_required for a true product fork."
    )
    args_schema: Type[BaseModel] = SynthesizeBriefInput

    def _run(
        self,
        chosen_approach: str,
        rationale: str = "",
        alternatives_considered: list | None = None,
        open_risks: list | None = None,
        research_findings: list | None = None,
        affected_files: list | None = None,
        dissent: list | None = None,
        user_decision_required: bool = False,
        open_questions_for_user: list | None = None,
    ) -> str:
        return json.dumps({
            "kind": "synthesize_brief",
            "chosen_approach": chosen_approach,
            "rationale": rationale,
            "alternatives_considered": alternatives_considered or [],
            "open_risks": open_risks or [],
            "research_findings": research_findings or [],
            "affected_files": affected_files or [],
            "dissent": dissent or [],
            "user_decision_required": user_decision_required,
            "open_questions_for_user": open_questions_for_user or [],
        }, default=lambda o: o.model_dump() if hasattr(o, "model_dump") else str(o))


__all__ = [
    "SeedCouncilTool",
    "CouncilVerdictTool",
    "SynthesizeBriefTool",
]
