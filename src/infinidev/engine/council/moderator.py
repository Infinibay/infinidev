"""The moderator: seeds the council, judges convergence, synthesises.

Three thin wrappers around :func:`run_terminating_loop`, each parsing
one terminator's args into a typed artifact:

  * :func:`seed_council`      → :class:`CouncilRoster`
  * :func:`judge_convergence` → ``bool``
  * :func:`synthesize`        → :class:`DesignBrief`

Every function fails *closed* into a sensible default rather than
raising, so a flaky model never strands the pipeline — the council
degrades to "fewer/weaker members" or "stop now", never to a crash.
"""

from __future__ import annotations

import logging
from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.council.agent_loop import run_terminating_loop
from infinidev.engine.council.brief import (
    Alternative,
    CouncilRoster,
    DesignBrief,
    MemberAssignment,
    OpeningThread,
)
from infinidev.engine.council import prompts as P
from infinidev.tools import get_tools_for_role

logger = logging.getLogger(__name__)


def _ctx(session_id, project_id, workspace_path) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "project_id": project_id,
        "workspace_path": workspace_path,
    }


# ── Seed ─────────────────────────────────────────────────────────────────


def seed_council(
    handoff: str,
    *,
    session_id: str | None = None,
    project_id: int | None = None,
    workspace_path: str | None = None,
) -> CouncilRoster:
    """Run the moderator's seed turn → a roster of members + threads."""
    tools = get_tools_for_role("council_moderator")
    result = run_terminating_loop(
        system_prompt=P.build_moderator_seed_prompt(),
        user_content=P.render_seed_user_message(handoff),
        tools=tools,
        terminator_names={"seed_council"},
        max_iterations=settings.COUNCIL_MODERATOR_MAX_ITERS,
        agent_id_prefix="council-mod-seed",
        temperature=0.5,  # a little heat → more diverse personas
        max_tokens=2500,
        **_ctx(session_id, project_id, workspace_path),
    )
    if result.terminator != "seed_council":
        logger.warning("Moderator did not seed; using fallback roster")
        return _fallback_roster(handoff)
    return _parse_roster(result.args, handoff)


def _parse_roster(args: dict[str, Any], handoff: str) -> CouncilRoster:
    question = (args.get("question") or "").strip()
    members: list[MemberAssignment] = []
    seen_ids: set[str] = set()
    for i, m in enumerate(args.get("members") or []):
        if not isinstance(m, dict):
            continue
        mid = (m.get("member_id") or f"member-{i+1}").strip() or f"member-{i+1}"
        # De-dup ids so two members never share a context-key prefix.
        if mid in seen_ids:
            mid = f"{mid}-{i+1}"
        seen_ids.add(mid)
        persona = (m.get("persona") or "").strip()
        objective = (m.get("objective") or "").strip()
        if not persona or not objective:
            continue
        st = m.get("seed_tools") or []
        members.append(MemberAssignment(
            member_id=mid, persona=persona, objective=objective,
            seed_tools=[str(t) for t in st] if isinstance(st, list) else [],
        ))
    threads: list[OpeningThread] = []
    for t in args.get("opening_threads") or []:
        if not isinstance(t, dict):
            continue
        title = (t.get("title") or "").strip()
        prompt = (t.get("prompt") or "").strip()
        if title and prompt:
            threads.append(OpeningThread(title=title, prompt=prompt))

    # Clamp member count and guarantee a non-degenerate council.
    if len(members) < 2 or not question or not threads:
        logger.warning(
            "Seed underspecified (members=%d, q=%s, threads=%d); "
            "falling back", len(members), bool(question), len(threads),
        )
        return _fallback_roster(handoff)
    members = members[: settings.COUNCIL_MAX_MEMBERS]
    return CouncilRoster(question=question, members=members, opening_threads=threads)


def _fallback_roster(handoff: str) -> CouncilRoster:
    """Deterministic minimal council when the moderator misbehaves.

    Three members from the palette with generic objectives derived from
    the handoff. Not as good as a tailored roster, but keeps the council
    functional instead of aborting to the planner.
    """
    from infinidev.engine.council.personas import PERSONA_PALETTE

    question = (
        "What is the best design/approach for the escalated request? "
        "(auto-framed — the moderator did not produce a question)"
    )
    members = [
        MemberAssignment(
            member_id=pid,
            persona=desc,
            objective=(
                "Argue your perspective on the best approach for the "
                "escalated request and critique the others'."
            ),
        )
        for pid, desc in PERSONA_PALETTE[:3]
    ]
    threads = [OpeningThread(
        title="Approach",
        prompt=(
            "Propose and debate the best approach for the request. "
            f"Context:\n{handoff[:1500]}"
        ),
    )]
    return CouncilRoster(question=question, members=members, opening_threads=threads)


# ── Judge ────────────────────────────────────────────────────────────────


def judge_convergence(
    digest: str,
    round_num: int,
    *,
    session_id: str | None = None,
    project_id: int | None = None,
    workspace_path: str | None = None,
) -> tuple[bool, str]:
    """Ask the moderator whether the debate converged. Returns (converged, reason).

    Fails closed to ``(False, ...)`` so a parse failure means "keep
    debating" rather than prematurely cutting off — the hard round cap in
    the runner still bounds total cost.
    """
    tools = get_tools_for_role("council_moderator")
    result = run_terminating_loop(
        system_prompt=P.build_moderator_judge_prompt(),
        user_content=P.render_judge_user_message(
            digest, round_num, settings.COUNCIL_MAX_ROUNDS,
        ),
        tools=tools,
        terminator_names={"council_verdict"},
        max_iterations=2,  # judge shouldn't need to explore
        agent_id_prefix="council-mod-judge",
        temperature=0.1,
        max_tokens=400,
        **_ctx(session_id, project_id, workspace_path),
    )
    if result.terminator != "council_verdict":
        return (False, "no verdict emitted")
    converged = bool(result.args.get("converged"))
    reason = (result.args.get("reason") or "").strip()
    return (converged, reason)


# ── Synthesize ───────────────────────────────────────────────────────────


def synthesize(
    digest: str,
    question: str,
    *,
    session_id: str | None = None,
    project_id: int | None = None,
    workspace_path: str | None = None,
) -> DesignBrief:
    """Run the moderator's synthesis turn → the final DesignBrief."""
    tools = get_tools_for_role("council_moderator")
    result = run_terminating_loop(
        system_prompt=P.build_moderator_synth_prompt(),
        user_content=P.render_synth_user_message(digest),
        tools=tools,
        terminator_names={"synthesize_brief"},
        # Generous: the moderator may read files / search to verify the
        # brief's claims and affected-files list before emitting it. It
        # terminates as soon as it calls synthesize_brief.
        max_iterations=settings.COUNCIL_MODERATOR_MAX_ITERS,
        agent_id_prefix="council-mod-synth",
        temperature=0.2,
        max_tokens=2500,
        **_ctx(session_id, project_id, workspace_path),
    )
    if result.terminator != "synthesize_brief":
        logger.warning("Moderator did not synthesise; using minimal brief")
        return DesignBrief(
            question=question,
            chosen_approach=(
                "The council debated but produced no explicit synthesis. "
                "Proceed with the most-supported approach from the channel."
            ),
        )
    return _parse_brief(result.args, question)


def _parse_brief(args: dict[str, Any], question: str) -> DesignBrief:
    def _slist(key: str) -> list[str]:
        v = args.get(key) or []
        return [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []

    alts: list[Alternative] = []
    for a in args.get("alternatives_considered") or []:
        if isinstance(a, dict) and (a.get("approach") or "").strip():
            alts.append(Alternative(
                approach=str(a.get("approach")).strip(),
                why_rejected=str(a.get("why_rejected") or "").strip(),
            ))

    questions = _slist("open_questions_for_user")
    needs_user = bool(args.get("user_decision_required")) and bool(questions)
    return DesignBrief(
        question=question,
        chosen_approach=(args.get("chosen_approach") or "").strip() or
        "(no approach stated)",
        rationale=(args.get("rationale") or "").strip(),
        alternatives_considered=alts,
        open_risks=_slist("open_risks"),
        research_findings=_slist("research_findings"),
        affected_files=_slist("affected_files"),
        dissent=_slist("dissent"),
        user_decision_required=needs_user,
        open_questions_for_user=questions if needs_user else [],
    )


__all__ = ["seed_council", "judge_convergence", "synthesize"]
