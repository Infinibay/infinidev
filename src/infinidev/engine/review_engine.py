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


class ReviewEngine:
    """Post-development code review engine.

    Makes a single LLM call with the reviewer prompt to evaluate code
    changes. Returns APPROVED, REJECTED, or SKIPPED (if no code changes).
    """

    def __init__(self) -> None:
        self._review_count: int = 0
        self._max_reviews: int = 3  # Max review-rework cycles

    def reset(self) -> None:
        """Reset state for a new task."""
        self._review_count = 0

    def review(
        self,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        *,
        file_reasons: dict[str, list[str]] | None = None,
        file_contents: dict[str, str] | None = None,
        recent_messages: list[str] | None = None,
        previous_feedback: str = "",
        event_callback: Any | None = None,
    ) -> ReviewResult:
        """Review code changes and return a ReviewResult.

        Args:
            task_description: The original task given to the developer.
            developer_result: The developer's final answer.
            file_changes_summary: Unified diffs of all changed files.
            file_reasons: path → list of reasons for each changed file.
            file_contents: path → current content for each changed file.
            recent_messages: Last few conversation messages for context.
            previous_feedback: Feedback from a previous review round (for re-reviews).
            event_callback: Optional callback for emitting review events.

        Returns:
            ReviewResult with verdict and feedback.
        """
        # Skip review if no code changes AND no files provided directly
        if not file_changes_summary.strip() and not file_contents:
            logger.info("ReviewEngine: no file changes and no files provided, skipping review")
            return ReviewResult(
                verdict="SKIPPED",
                summary="No file changes to review",
            )

        from infinidev.config.llm import get_litellm_params
        from infinidev.prompts.reviewer.system import REVIEWER_SYSTEM_PROMPT

        llm_params = get_litellm_params()
        if llm_params is None:
            logger.warning("ReviewEngine: no LLM params, skipping review")
            return ReviewResult(
                verdict="SKIPPED",
                summary="No LLM configured",
            )

        self._review_count += 1

        # Emit review start event
        if event_callback:
            event_callback("review_start", 0, "", {
                "round": self._review_count,
            })

        user_prompt = self._build_review_prompt(
            task_description, developer_result,
            file_changes_summary, previous_feedback,
            file_reasons=file_reasons or {},
            file_contents=file_contents or {},
            recent_messages=recent_messages or [],
        )

        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            import litellm
            response = litellm.completion(
                messages=messages,
                **llm_params,
                temperature=0.2,  # Low temp for consistent reviews
            )
            raw_content = response.choices[0].message.content or ""
            result = self._parse_response(raw_content)

        except Exception as e:
            logger.warning("ReviewEngine: LLM call failed (%s), skipping review", e)
            result = ReviewResult(
                verdict="SKIPPED",
                summary=f"Review skipped due to error: {e}",
            )

        # Emit review complete event
        if event_callback:
            event_callback("review_complete", 0, "", {
                "verdict": result.verdict,
                "round": self._review_count,
                "issue_count": len(result.issues),
            })

        return result

    def _build_review_prompt(
        self,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        previous_feedback: str,
        *,
        file_reasons: dict[str, list[str]] | None = None,
        file_contents: dict[str, str] | None = None,
        recent_messages: list[str] | None = None,
    ) -> str:
        """Build the user prompt for the review LLM call."""
        parts = []

        # Conversation context (last few messages, highlight the latest)
        if recent_messages:
            parts.append("## Conversation Context")
            for i, msg in enumerate(recent_messages):
                if i == len(recent_messages) - 1:
                    parts.append(f">>> CURRENT REQUEST <<<\n{msg}")
                else:
                    parts.append(msg)

        # Task context
        parts.append(f"## Original Task\n{task_description}")

        # Developer's result
        parts.append(f"## Developer's Report\n{developer_result}")

        # Previous review feedback (for re-reviews)
        if previous_feedback:
            parts.append(
                f"## Previous Review Feedback (Round {self._review_count})\n"
                f"The developer was asked to fix these issues. Verify they were addressed:\n"
                f"{previous_feedback}"
            )

        # Files changed with reasons and current content
        if file_contents or file_reasons:
            parts.append("## Files Changed")
            all_paths = set(list((file_contents or {}).keys()) + list((file_reasons or {}).keys()))
            for path in sorted(all_paths):
                file_part = [f"### `{path}`"]
                reasons = (file_reasons or {}).get(path, [])
                if reasons:
                    file_part.append("**Reasons for changes:**")
                    for reason in reasons:
                        file_part.append(f"- {reason}")
                content = (file_contents or {}).get(path)
                if content is not None:
                    # Truncate very large files
                    max_chars = 50_000
                    if len(content) > max_chars:
                        content = content[:max_chars] + f"\n... (truncated, {len(content)} total chars)"
                    ext = path.rsplit(".", 1)[-1] if "." in path else ""
                    file_part.append(f"**Current content:**\n```{ext}\n{content}\n```")
                parts.append("\n".join(file_part))

        # The actual diffs to review
        parts.append(f"## Diffs\n{file_changes_summary}")

        return "\n\n".join(parts)

    def _parse_response(self, raw: str) -> ReviewResult:
        """Parse the LLM response into a ReviewResult."""
        raw = raw.strip()

        # Handle markdown code blocks
        if raw.startswith("```"):
            lines = raw.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.strip() == "```" and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            raw = "\n".join(json_lines)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    data = json.loads(raw[start:end])
                except json.JSONDecodeError:
                    logger.warning("ReviewEngine: could not parse response")
                    return ReviewResult(
                        verdict="SKIPPED",
                        summary="Could not parse review response",
                    )
            else:
                return ReviewResult(
                    verdict="SKIPPED",
                    summary="No JSON in review response",
                )

        verdict = data.get("verdict", "APPROVED").upper()
        if verdict not in ("APPROVED", "REJECTED"):
            verdict = "APPROVED"

        return ReviewResult(
            verdict=verdict,
            summary=data.get("summary", ""),
            issues=data.get("issues", []),
            notes=data.get("notes", []),
        )

    @property
    def can_review_again(self) -> bool:
        """Whether we can do another review-rework cycle."""
        return self._review_count < self._max_reviews
