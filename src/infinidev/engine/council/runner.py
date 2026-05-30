"""run_council — orchestrates the whole deliberation.

Flow:

    seed (moderator) ──► open threads on a fresh Channel
        │
        ▼
    for round in 1..MAX_ROUNDS:
        digest = channel.snapshot().render_digest()      # frozen view
        turns  = parallel(run_member_round, members)      # barrier
        apply turns to the channel in deterministic order # commit
        if all concluded: break
        if moderator.judge(channel) == converged: break
        │
        ▼
    synthesize (moderator) ──► DesignBrief

Returns the :class:`DesignBrief`, or ``None`` if the council is disabled
or fails before producing anything — the pipeline then proceeds straight
to the planner, exactly as it did before the council existed.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from infinidev.config.settings import settings
from infinidev.engine.council import moderator as MOD
from infinidev.engine.council.brief import DesignBrief
from infinidev.engine.council.channel import Channel
from infinidev.engine.council.member import MemberTurn, run_member_round

logger = logging.getLogger(__name__)


def run_council(
    handoff: str,
    *,
    session_id: str | None = None,
    project_id: int | None = None,
    workspace_path: str | None = None,
    hooks: Any | None = None,
) -> DesignBrief | None:
    """Deliberate on ``handoff`` and return a synthesised DesignBrief.

    ``handoff`` is free text describing the escalated request (the
    pipeline builds it from the EscalationPacket). ``hooks`` is an
    optional :class:`OrchestrationHooks` used purely to surface the
    debate to the user (each post → ``notify('Council:<id>', ...)``).
    """
    if not settings.COUNCIL_ENABLED:
        return None

    ctx = dict(
        session_id=session_id, project_id=project_id, workspace_path=workspace_path,
    )

    def _status(level: str, msg: str) -> None:
        if hooks is not None:
            try:
                hooks.on_status(level, msg)
            except Exception:
                pass

    def _say(speaker: str, msg: str) -> None:
        if hooks is not None and msg:
            try:
                hooks.notify(speaker, msg, "agent")
            except Exception:
                pass

    try:
        # ── Seed ─────────────────────────────────────────────────────────
        _status("info", "Convening council — assigning roles...")
        roster = MOD.seed_council(handoff, **ctx)
        if not roster.members:
            logger.warning("Council seeded with no members; skipping")
            return None

        _say(
            "Council",
            "Convoco un consejo de "
            f"{len(roster.members)} subagentes para debatir:\n"
            f"  {roster.question}\n\n"
            + "\n".join(
                f"  • {m.member_id}: {m.objective}" for m in roster.members
            ),
        )

        channel = Channel(roster.question)
        for ot in roster.opening_threads:
            channel.open_thread(
                author="moderator", title=ot.title,
                opening_text=ot.prompt, round=0,
            )

        # ── Rounds ───────────────────────────────────────────────────────
        concluded: set[str] = set()
        final_positions: dict[str, str] = {}
        max_rounds = max(1, settings.COUNCIL_MAX_ROUNDS)

        for round_num in range(1, max_rounds + 1):
            active = [m for m in roster.members if m.member_id not in concluded]
            if not active:
                break
            _status("info", f"Council round {round_num}/{max_rounds}...")

            snapshot = channel.snapshot()
            turns = _run_round_parallel(
                active, question=roster.question, snapshot=snapshot,
                round_num=round_num, ctx=ctx,
            )

            # Commit in deterministic order (by member_id) so the channel
            # is reproducible regardless of thread completion order.
            for turn in sorted(turns, key=lambda t: t.member_id):
                _commit_turn(turn, channel, round_num, concluded, final_positions, _say)

            if len(concluded) >= len(roster.members):
                _status("info", "All members concluded — closing council.")
                break
            if round_num >= max_rounds:
                break

            converged, reason = MOD.judge_convergence(
                channel.snapshot().render_digest(current_round=round_num),
                round_num, **ctx,
            )
            if converged:
                _status("info", f"Council converged: {reason}")
                break

        # ── Synthesize ───────────────────────────────────────────────────
        _status("info", "Synthesising design brief...")
        full_digest = channel.snapshot().render_digest(
            current_round=max_rounds + 1, recent_rounds=max_rounds + 1,
        )
        if final_positions:
            full_digest += "\n\nFINAL POSITIONS:\n" + "\n".join(
                f"  - {mid}: {pos}" for mid, pos in sorted(final_positions.items())
            )
        brief = MOD.synthesize(full_digest, roster.question, **ctx)
        return brief
    except Exception:
        logger.exception("Council failed; proceeding without a design brief")
        _status("warn", "Council failed — proceeding without a design brief.")
        return None


def _run_round_parallel(
    members: list, *, question: str, snapshot: Channel, round_num: int,
    ctx: dict,
) -> list[MemberTurn]:
    """Fan members out across a thread pool; barrier on all results.

    On a single local LLM server these serialise at the GPU — the
    parallelism buys deliberation quality, not wall-clock. With a cloud
    COUNCIL_MODEL it is genuinely concurrent.
    """
    max_workers = max(1, min(settings.COUNCIL_MAX_CONCURRENCY, len(members)))

    def _one(assignment) -> MemberTurn:
        digest = snapshot.render_digest(
            for_author=assignment.member_id, current_round=round_num,
        )
        return run_member_round(
            assignment, question=question, digest=digest, round_num=round_num,
            **ctx,
        )

    if max_workers == 1:
        return [_one(m) for m in members]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, members))


def _commit_turn(
    turn: MemberTurn, channel: Channel, round_num: int,
    concluded: set[str], final_positions: dict[str, str], say,
) -> None:
    """Apply one member's decided action to the live channel."""
    if turn.action == "post":
        if turn.new_thread_title:
            channel.open_thread(
                author=turn.member_id, title=turn.new_thread_title,
                opening_text=turn.message, round=round_num,
                refs=list(turn.refs),
            )
        else:
            tid = turn.thread_id or _first_thread_id(channel)
            posted = channel.post(
                author=turn.member_id, thread_id=tid, text=turn.message,
                round=round_num, parent_id=turn.parent_id or None,
                refs=list(turn.refs),
            )
            if posted is None and tid != _first_thread_id(channel):
                # Stale thread id — retry into the first thread so the
                # contribution isn't silently lost.
                channel.post(
                    author=turn.member_id, thread_id=_first_thread_id(channel),
                    text=turn.message, round=round_num, refs=list(turn.refs),
                )
        say(f"Council:{turn.member_id}", turn.message)
    elif turn.action == "conclude":
        concluded.add(turn.member_id)
        if turn.final_position:
            final_positions[turn.member_id] = turn.final_position
            say(
                f"Council:{turn.member_id}",
                f"[concluye] {turn.final_position}",
            )


def _first_thread_id(channel: Channel) -> str:
    titles = channel.thread_titles
    return titles[0][0] if titles else ""


__all__ = ["run_council"]
