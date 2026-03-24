"""Compile question results into a GatherBrief."""

from __future__ import annotations

from infinidev.gather.models import (
    ClassificationResult,
    GatherBrief,
    QuestionResult,
)


def compile_brief(
    ticket_description: str,
    classification: ClassificationResult,
    fixed_answers: list[QuestionResult],
    dynamic_answers: list[QuestionResult],
) -> GatherBrief:
    """Compile all gathered information into a GatherBrief."""
    return GatherBrief(
        ticket_description=ticket_description,
        classification=classification,
        fixed_answers=fixed_answers,
        dynamic_answers=dynamic_answers,
    )
