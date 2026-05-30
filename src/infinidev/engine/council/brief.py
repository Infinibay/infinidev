"""Structured artifacts the council produces.

Two contracts:

  * :class:`CouncilRoster` — emitted by the moderator when it *seeds*
    the council. It is the moderator's answer to "who debates and to
    what end": a list of :class:`MemberAssignment` (persona + objective
    per subagent) plus the threads to open. This is the literal
    realisation of "the orchestrator gives each subagent its personality,
    role and objective".

  * :class:`DesignBrief` — the council's output. It enriches the
    :class:`~infinidev.engine.orchestration.escalation_packet.EscalationPacket`
    and is rendered into the planner's handoff. Carries
    ``user_decision_required`` + ``open_questions_for_user`` so the
    pipeline can surface a genuine product fork to the user — and only
    then.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemberAssignment:
    """One council member, as assigned by the moderator.

    ``persona`` is *how* the member thinks (stable stance/bias, e.g.
    "a skeptic who assumes every proposal hides a bug"). ``objective``
    is *what* this member must achieve in this specific debate (e.g.
    "find how the caching approach breaks under concurrency"). Keeping
    them separate is deliberate: persona gives point of view, objective
    gives a concrete target. Merged, you get generic agents.
    """

    member_id: str
    persona: str
    objective: str
    seed_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OpeningThread:
    title: str
    prompt: str


@dataclass(frozen=True)
class CouncilRoster:
    """The moderator's seed output: who debates, and the opening topics."""

    question: str
    members: list[MemberAssignment] = field(default_factory=list)
    opening_threads: list[OpeningThread] = field(default_factory=list)


@dataclass(frozen=True)
class Alternative:
    approach: str
    why_rejected: str


@dataclass(frozen=True)
class DesignBrief:
    """The council's synthesised conclusion.

    Attributes:
        question: The question the council deliberated.
        chosen_approach: The recommended synthesis.
        rationale: Why this approach won.
        alternatives_considered: Approaches weighed and set aside.
        open_risks: Known risks the developer should watch for.
        research_findings: Grounded facts surfaced during the debate
            (ideally with file/symbol refs).
        affected_files: Files the council expects the work to touch.
        dissent: Minority positions worth flagging — the best signal of
            where a real product decision may be hiding.
        user_decision_required: True when the council hit a fork it must
            NOT resolve on its own (a product/design choice that belongs
            to the user). The model decides technical questions itself;
            this flag is reserved for genuine product forks.
        open_questions_for_user: The concrete questions to ask, in the
            user's language. Empty unless ``user_decision_required``.
    """

    question: str
    chosen_approach: str
    rationale: str = ""
    alternatives_considered: list[Alternative] = field(default_factory=list)
    open_risks: list[str] = field(default_factory=list)
    research_findings: list[str] = field(default_factory=list)
    affected_files: list[str] = field(default_factory=list)
    dissent: list[str] = field(default_factory=list)
    user_decision_required: bool = False
    open_questions_for_user: list[str] = field(default_factory=list)

    # ── Rendering ────────────────────────────────────────────────────────

    def render_for_planner(self) -> str:
        """Render as a context block for the analyst planner's handoff.

        The planner reads this to ground its plan in the council's
        decision instead of re-deciding the design from scratch.
        """
        lines = [
            "DESIGN BRIEF (produced by a multi-agent council — treat as "
            "the agreed design; build the plan on top of it):",
            "",
            f"Chosen approach:\n  {self.chosen_approach}",
        ]
        if self.rationale:
            lines += ["", f"Rationale:\n  {self.rationale}"]
        if self.alternatives_considered:
            lines += ["", "Alternatives considered (do NOT revisit):"]
            for a in self.alternatives_considered:
                lines.append(f"  - {a.approach} — rejected: {a.why_rejected}")
        if self.research_findings:
            lines += ["", "Research findings (grounded facts):"]
            lines += [f"  - {f}" for f in self.research_findings]
        if self.affected_files:
            lines += ["", "Likely affected files:"]
            lines += [f"  - {p}" for p in self.affected_files]
        if self.open_risks:
            lines += ["", "Open risks to watch:"]
            lines += [f"  - {r}" for r in self.open_risks]
        if self.dissent:
            lines += ["", "Dissent (minority positions, flagged):"]
            lines += [f"  - {d}" for d in self.dissent]
        return "\n".join(lines)

    def render_questions_for_user(self) -> str:
        """Render the open questions as a single prompt for ``ask_user``."""
        if not self.open_questions_for_user:
            return ""
        lines = [
            "The council reached a point that needs your decision before "
            "implementation:",
            "",
            f"Context: {self.chosen_approach}",
            "",
        ]
        for i, q in enumerate(self.open_questions_for_user, 1):
            lines.append(f"  {i}. {q}")
        lines += ["", "Your answer (free text):"]
        return "\n".join(lines)

    def render_user_preview(self) -> str:
        """Short, user-facing summary shown via ``notify`` after the council."""
        head = f"Consejo deliberó — enfoque elegido: {self.chosen_approach}"
        if self.dissent:
            head += f"\n(Disidencia registrada: {self.dissent[0]})"
        return head


__all__ = [
    "MemberAssignment",
    "OpeningThread",
    "CouncilRoster",
    "Alternative",
    "DesignBrief",
]
