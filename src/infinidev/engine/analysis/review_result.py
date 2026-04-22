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


# Categories the judge may assign to an issue. ``structural`` is the
# carve-out for whole-file issues (missing test file entirely, missing
# module, etc) that cannot meaningfully cite a single line.
ISSUE_CATEGORIES: tuple[str, ...] = (
    "test_missing", "test_failure", "regression", "logic_bug",
    "api_break", "style", "docstring", "structural",
)


@dataclass
class ReviewResult:
    """Result of the code review phase.

    Each entry in ``issues`` is a dict with the following keys:

        severity     — "blocking" | "important" | "suggestion"
        file         — path of the affected file
        description  — what's wrong (short, factual)
        why          — why it matters
        fix          — suggested remediation
        line         — int | None; relevant line number (required for
                       blocking issues unless category == "structural")
        quoted_text  — verbatim excerpt from the diff or current file at
                       ``line`` (required for blocking non-structural)
        category     — one of :data:`ISSUE_CATEGORIES`
        check_id     — str | None; name of the deterministic check that
                       surfaced this issue (e.g. "orphaned_references"),
                       None if it came purely from the judge LLM
        id           — 10-char sha1 hash assigned deterministically
                       from the other fields; stable across retries
    """

    verdict: str  # "APPROVED" | "REJECTED" | "SKIPPED"
    summary: str = ""
    issues: list[dict[str, Any]] = field(default_factory=list)
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
            issue_id = issue.get("id") or "?"
            category = issue.get("category") or "uncategorized"
            parts.append(f"\n**Issue {i} [{issue_id} · {category}]:** {issue.get('description', '')}")
            # Citation line: file:line — "quoted_text" is the reviewer's
            # proof. Putting it up top makes it impossible to miss.
            file_path = issue.get("file")
            line = issue.get("line")
            quoted = issue.get("quoted_text")
            if file_path and line is not None and quoted:
                parts.append(f"  - **Where:** `{file_path}:{line}` — `{_trim_quote(quoted)}`")
            elif file_path and line is not None:
                parts.append(f"  - **Where:** `{file_path}:{line}`")
            elif file_path:
                parts.append(f"  - **File:** `{file_path}`")
            if issue.get("why"):
                parts.append(f"  - **Why:** {issue['why']}")
            if issue.get("fix"):
                parts.append(f"  - **Fix:** {issue['fix']}")
            check_id = issue.get("check_id")
            if check_id:
                parts.append(f"  - **Source:** deterministic check `{check_id}`")

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


def _trim_quote(text: str, max_len: int = 80) -> str:
    """Trim a quoted excerpt for prompt rendering, keeping it single-line."""
    if not text:
        return ""
    cleaned = text.replace("\n", " ").replace("\r", " ").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


