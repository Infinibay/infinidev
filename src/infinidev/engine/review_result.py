"""Post-development code review engine.

Runs a single LLM call to review code changes after the developer loop
completes. If the review finds blocking issues, it provides feedback
that is fed back into the developer loop for fixes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of the code review phase."""

    verdict: str  # "APPROVED" | "REJECTED" | "SKIPPED"
    summary: str = ""
    issues: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def is_approved(self) -> bool:
        return self.verdict == "APPROVED"

    @property
    def is_rejected(self) -> bool:
        return self.verdict == "REJECTED"

    def format_feedback_for_developer(self) -> str:
        """Format rejection feedback as context for the developer loop retry."""
        if not self.is_rejected:
            return ""

        parts = [
            "## Code Review Feedback — REJECTED",
            "",
            f"**Summary:** {self.summary}",
            "",
            "### Blocking Issues (MUST fix)",
        ]

        for i, issue in enumerate(self.issues, 1):
            severity = issue.get("severity", "blocking")
            if severity != "blocking":
                continue
            parts.append(f"\n**Issue {i}:** {issue.get('description', '')}")
            if issue.get("file"):
                parts.append(f"  - **File:** {issue['file']}")
            if issue.get("why"):
                parts.append(f"  - **Why:** {issue['why']}")
            if issue.get("fix"):
                parts.append(f"  - **Fix:** {issue['fix']}")

        # Non-blocking notes
        important = [
            issue for issue in self.issues
            if issue.get("severity") in ("important", "suggestion")
        ]
        if important:
            parts.append("\n### Additional Notes (non-blocking)")
            for issue in important:
                parts.append(f"- [{issue.get('severity', 'note')}] {issue.get('description', '')}")

        if self.notes:
            parts.append("\n### Reviewer Notes")
            for note in self.notes:
                parts.append(f"- {note}")

        return "\n".join(parts)

    def format_for_user(self) -> str:
        """Format review result for display to the user."""
        if self.verdict == "SKIPPED":
            return ""

        if self.is_approved:
            text = f"Code review: APPROVED. {self.summary}"
            if self.notes:
                text += "\nNotes: " + "; ".join(self.notes)
            return text

        text = f"Code review: REJECTED. {self.summary}"
        blocking = [i for i in self.issues if i.get("severity") == "blocking"]
        if blocking:
            text += f"\n{len(blocking)} blocking issue(s) found — sending back to developer."
        return text


