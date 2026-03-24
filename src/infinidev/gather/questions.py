"""Question registry — maps ticket types to fixed question templates."""

from __future__ import annotations

from infinidev.gather.models import Question, TicketType
from infinidev.gather.templates import bug, feature, refactor, sysadmin, other


_QUESTION_MAP: dict[TicketType, list[Question]] = {
    TicketType.bug: bug.QUESTIONS,
    TicketType.feature: feature.QUESTIONS,
    TicketType.refactor: refactor.QUESTIONS,
    TicketType.sysadmin: sysadmin.QUESTIONS,
    TicketType.other: other.QUESTIONS,
}


def get_questions_for_type(ticket_type: TicketType) -> list[Question]:
    """Return the fixed questions for a given ticket type."""
    return list(_QUESTION_MAP.get(ticket_type, other.QUESTIONS))
