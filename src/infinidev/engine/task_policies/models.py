"""Structured task-profile and policy-selection contracts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Operation = Literal[
    "bugfix", "feature", "refactor", "research", "review",
    "performance", "docs", "migration", "security",
]
Authority = Literal["answer", "diagnose", "modify", "commit", "publish"]
Constraint = Literal["preserve_behavior", "preserve_public_api", "read_only"]
Risk = Literal["security", "migration", "destructive", "external_write"]
ResultKind = Literal["code", "report", "plan", "recommendation"]
SequenceStep = Literal["investigate", "implement", "verify", "review", "commit", "publish"]
SelectionSource = Literal["deterministic", "embedding", "llm"]


class PolicySelection(BaseModel):
    """One selected policy and the evidence used to select it."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: int
    source: SelectionSource
    evidence: tuple[str, ...] = ()
    score: float | None = None
    policy_hash: str


class RejectedPolicyCandidate(BaseModel):
    """A candidate intentionally omitted after conflict resolution."""

    model_config = ConfigDict(frozen=True)

    id: str
    reason: str
    score: float | None = None


class TaskProfile(BaseModel):
    """Versioned, reusable interpretation of task method and literal authority."""

    model_config = ConfigDict(frozen=True)

    version: int = 1
    operations: tuple[Operation, ...] = ()
    authority: tuple[Authority, ...] = ("answer",)
    constraints: tuple[Constraint, ...] = ()
    risks: tuple[Risk, ...] = ()
    result: tuple[ResultKind, ...] = ()
    sequence: tuple[SequenceStep, ...] = ()
    selected_policies: tuple[PolicySelection, ...] = ()
    rejected_candidates: tuple[RejectedPolicyCandidate, ...] = ()
    llm_classifier_used: bool = False
    llm_fallback_used: bool = False
    semantic_space_id: str | None = None
    semantic_classifier_version: str | None = None
    semantic_abstained: bool = False
    semantic_abstention_reason: str = ""
    router_version: int = 2

    def event_payload(self) -> dict[str, object]:
        """Return a stable JSON-compatible payload for history and replay."""
        return {
            "task_profile_version": self.version,
            "router_version": self.router_version,
            "operations": list(self.operations),
            "authority": list(self.authority),
            "constraints": list(self.constraints),
            "risks": list(self.risks),
            "result": list(self.result),
            "sequence": list(self.sequence),
            "selected_policies": [item.model_dump() for item in self.selected_policies],
            "rejected_candidates": [item.model_dump() for item in self.rejected_candidates],
            "llm_classifier_used": self.llm_classifier_used,
            "llm_fallback_used": self.llm_fallback_used,
            "semantic_space_id": self.semantic_space_id,
            "semantic_classifier_version": self.semantic_classifier_version,
            "semantic_abstained": self.semantic_abstained,
            "semantic_abstention_reason": self.semantic_abstention_reason,
        }


class ClassifierResult(BaseModel):
    """Closed method schema accepted from an optional single-call LLM classifier."""

    model_config = ConfigDict(extra="forbid")

    operations: list[Operation] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    result: list[ResultKind] = Field(default_factory=list)
    sequence: list[SequenceStep] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
