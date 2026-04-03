"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from infinidev.gather.classification_result import ClassificationResult
from infinidev.gather.question_result import QuestionResult


class GatherBrief(BaseModel):
    """Complete brief from the gathering phase."""

    ticket_description: str = ""
    classification: ClassificationResult = Field(default_factory=ClassificationResult)
    fixed_answers: list[QuestionResult] = Field(default_factory=list)
    dynamic_answers: list[QuestionResult] = Field(default_factory=list)

    def render(self) -> str:
        """Render as XML text for injection into the developer system prompt."""
        parts = ["<gathered-context>"]

        # Classification
        parts.append(
            f'<classification type="{self.classification.ticket_type.value}">\n'
            f"Reasoning: {self.classification.reasoning}\n"
            f"Keywords: {', '.join(self.classification.keywords)}\n"
            f"</classification>"
        )

        # Investigation results
        all_answers = self.fixed_answers + self.dynamic_answers
        if all_answers:
            parts.append("<investigation>")
            for r in all_answers:
                # Truncate answer to ~500 tokens (~2000 chars)
                answer = r.answer[:2000]
                if len(r.answer) > 2000:
                    answer += "...[truncated]"
                parts.append(
                    f'<q id="{r.question_id}" phase="{r.phase}">\n'
                    f"Q: {r.question_text}\n"
                    f"A: {answer}\n"
                    f"</q>"
                )
            parts.append("</investigation>")

        parts.append("</gathered-context>")
        return "\n\n".join(parts)

    def summary(self) -> str:
        """Short summary for TUI display."""
        total = len(self.fixed_answers) + len(self.dynamic_answers)
        return (
            f"Gathered {total} context items "
            f"(type: {self.classification.ticket_type.value})"
        )

