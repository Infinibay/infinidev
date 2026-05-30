"""A council member's single round-turn.

Stateless across rounds by design: the member reads a frozen digest of
the channel (its only "memory" is the board), explores read-only if it
wants evidence, then contributes one post or concludes. The runner
applies the returned action to the live channel at the round barrier.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from infinidev.config.settings import settings
from infinidev.engine.council import prompts as P
from infinidev.engine.council.agent_loop import run_terminating_loop
from infinidev.engine.council.brief import MemberAssignment
from infinidev.tools import get_tools_for_role

logger = logging.getLogger(__name__)


@dataclass
class MemberTurn:
    """What a member decided this round — to be applied to the channel.

    ``action`` is ``"post"``, ``"conclude"``, or ``"noop"`` (the member
    failed to produce a terminator; the round simply skips it).
    """

    member_id: str
    action: str  # "post" | "conclude" | "noop"
    # post fields
    message: str = ""
    thread_id: str = ""
    new_thread_title: str = ""
    parent_id: str = ""
    refs: tuple[str, ...] = ()
    # conclude fields
    final_position: str = ""
    confidence: str = "medium"


def run_member_round(
    assignment: MemberAssignment,
    *,
    question: str,
    digest: str,
    round_num: int,
    session_id: str | None = None,
    project_id: int | None = None,
    workspace_path: str | None = None,
) -> MemberTurn:
    """Run one member for one round and return its intended channel action.

    Never raises — any failure degrades to a ``noop`` turn so one bad
    member can't take down the whole council (the runner runs members
    concurrently and tolerates noops).
    """
    try:
        tools = get_tools_for_role("council_member")
        result = run_terminating_loop(
            system_prompt=P.build_member_system_prompt(assignment, question),
            user_content=P.render_member_round_message(digest, round_num),
            tools=tools,
            terminator_names={"channel_post", "conclude"},
            max_iterations=settings.COUNCIL_MEMBER_MAX_ITERS,
            agent_id_prefix=f"council-{assignment.member_id}",
            temperature=0.6,  # members are meant to be opinionated/diverse
            max_tokens=1200,
            session_id=session_id,
            project_id=project_id,
            workspace_path=workspace_path,
        )
    except Exception:
        logger.exception("Member %s crashed this round", assignment.member_id)
        return MemberTurn(member_id=assignment.member_id, action="noop")

    if result.terminator == "channel_post":
        a = result.args
        refs = a.get("refs") or []
        return MemberTurn(
            member_id=assignment.member_id,
            action="post",
            message=(a.get("message") or "").strip(),
            thread_id=(a.get("thread_id") or "").strip(),
            new_thread_title=(a.get("new_thread_title") or "").strip(),
            parent_id=(a.get("parent_id") or "").strip(),
            refs=tuple(str(r) for r in refs) if isinstance(refs, list) else (),
        )
    if result.terminator == "conclude":
        a = result.args
        return MemberTurn(
            member_id=assignment.member_id,
            action="conclude",
            final_position=(a.get("final_position") or "").strip(),
            confidence=(a.get("confidence") or "medium").strip(),
        )
    return MemberTurn(member_id=assignment.member_id, action="noop")


__all__ = ["MemberTurn", "run_member_round"]
