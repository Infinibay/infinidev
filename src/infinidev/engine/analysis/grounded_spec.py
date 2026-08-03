"""GroundedSpec — the artefact produced by the spec-elaboration loop.

A vague user requirement is turned into a grounded specification BEFORE the
planner decomposes it: scope made explicit, gaps enumerated and resolved
against evidence (real code / retrieval / web), product-intent gaps surfaced
for the user, and a design direction chosen with its rejected alternatives.

This is a frozen dataclass attached to the EscalationPacket (mirroring
``design_brief``) and rendered into the planner handoff so the planner builds
steps on top of a complete spec instead of the raw request. See
``docs/SPEC_ELABORATION.md`` for the runtime contract.

Every field is produced by the single configured model (no tiering) plus
deterministic checks; ``render_for_planner`` is the only contract the planner
depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResolvedFact:
    """A technical/theory gap resolved against evidence."""

    question: str
    answer: str
    evidence: str = ""  # "file:line" or a URL/citation; empty if none
    confidence: str = "medium"  # low | medium | high


@dataclass(frozen=True)
class Assumption:
    """A gap that could NOT be resolved from evidence — stated, never hidden."""

    statement: str
    why_no_evidence: str = ""
    reversible: bool = True


@dataclass(frozen=True)
class RejectedAlternative:
    """A candidate design direction the deterministic checks (or the model) killed."""

    alternative: str
    why_rejected: str


@dataclass(frozen=True)
class Clarification:
    """A product decision that genuinely forks the implementation.

    The shape is the filter. A question only earns the user's attention if
    the model can name the concrete ``options`` and commit to the ``default``
    it will build in the meantime — an open-ended question with no
    alternatives and no default is a questionnaire, not a decision, and
    ``_admissible_clarifications`` demotes it to an assumption.

    ``default`` is load-bearing: it is what actually gets implemented, so the
    turn proceeds instead of stalling on an answer that may never come.
    """

    question: str
    options: list[str] = field(default_factory=list)
    default: str = ""
    impact: str = ""  # what changes in the code depending on the answer

    def render(self) -> str:
        """One line: the decision, what we build now, what else was possible."""
        others = [o for o in self.options if o.strip() and o != self.default]
        line = f"{self.question} — proceeding with: {self.default}"
        if others:
            line += f" (alternatives: {'; '.join(others)})"
        return line


@dataclass(frozen=True)
class GroundedSpec:
    """The elaborated specification handed to the planner.

    ``provenance`` is always the single configured model — there is no tier
    routing. ``evidence_count`` (facts backed by real evidence vs assumptions)
    is the grounding signal surfaced for validation.
    """

    deliverable: str
    in_scope: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    resolved_facts: list[ResolvedFact] = field(default_factory=list)
    assumptions: list[Assumption] = field(default_factory=list)
    # Product decisions surfaced for the user. Non-blocking: each carries the
    # default that IS implemented this turn, so the user corrects course
    # instead of being interrogated before any work happens.
    clarifications_needed: list[Clarification] = field(default_factory=list)
    design_direction: str = ""
    alternatives_rejected: list[RejectedAlternative] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    # Rich retrieval key for the Recipe Bank — much better than the raw request.
    signature_text: str = ""

    @property
    def evidence_count(self) -> int:
        """How many facts are backed by real evidence (grounding signal)."""
        return sum(1 for f in self.resolved_facts if f.evidence.strip())

    def render_for_planner(self) -> str:
        """Render the spec as a prompt block for the planner handoff.

        Kept compact: the planner needs the spec, not a transcript. The
        ContextGovernor prunes further if the window is tight.
        """
        lines = ["GROUNDED SPEC (elaborated from the request — build steps ON TOP of this):"]
        if self.deliverable:
            lines.append(f"  Deliverable: {self.deliverable}")
        if self.in_scope:
            lines.append("  In scope:")
            lines += [f"    - {s}" for s in self.in_scope]
        if self.out_of_scope:
            lines.append("  Out of scope (do NOT do these):")
            lines += [f"    - {s}" for s in self.out_of_scope]
        if self.resolved_facts:
            lines.append("  Resolved facts (grounded in evidence):")
            for f in self.resolved_facts:
                ev = f" [{f.evidence}]" if f.evidence.strip() else " [no evidence]"
                # Spelled out: this block renders into the planner's user
                # message, whose own prompt bans arrow glyphs.
                lines.append(f"    - {f.question} — answer: {f.answer}{ev}")
        if self.assumptions:
            lines.append("  ASSUMPTIONS (unverified — flag if any is wrong):")
            lines += [f"    - {a.statement}" for a in self.assumptions]
        if self.clarifications_needed:
            # The planner is told to BUILD the default, not to stall on the
            # question: the user was shown the same default and can correct
            # it. What the planner must not do is silently pick a third
            # option, or block the plan waiting for an answer.
            lines.append(
                "  PRODUCT DECISIONS (implement the stated default; do NOT invent a "
                "different answer and do NOT block on these):"
            )
            lines += [f"    - {c.render()}" for c in self.clarifications_needed]
        if self.design_direction:
            lines.append(f"  Design direction: {self.design_direction}")
        if self.alternatives_rejected:
            lines.append("  Alternatives rejected:")
            lines += [f"    - {r.alternative} — {r.why_rejected}" for r in self.alternatives_rejected]
        if self.risks:
            lines.append("  Risks:")
            lines += [f"    - {r}" for r in self.risks]
        return "\n".join(lines)

    def to_log_dict(self) -> dict[str, Any]:
        """Compact dict for logging / persistence."""
        return {
            "deliverable": self.deliverable,
            "in_scope": self.in_scope,
            "out_of_scope": self.out_of_scope,
            "resolved_facts": [vars(f) for f in self.resolved_facts],
            "assumptions": [vars(a) for a in self.assumptions],
            "clarifications_needed": [vars(c) for c in self.clarifications_needed],
            "design_direction": self.design_direction,
            "alternatives_rejected": [vars(r) for r in self.alternatives_rejected],
            "risks": self.risks,
            "open_questions": self.open_questions,
            "evidence_count": self.evidence_count,
            "signature_text": self.signature_text,
        }
