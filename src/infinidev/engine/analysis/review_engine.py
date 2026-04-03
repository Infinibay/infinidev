"""Post-development code review engine.

Runs a single LLM call to review code changes after the developer loop
completes. If the review finds blocking issues, it provides feedback
that is fed back into the developer loop for fixes.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


from infinidev.engine.analysis.review_result import ReviewResult

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

        Returns:
            ReviewResult with verdict and feedback.
        """
        from infinidev.flows.event_listeners import event_bus

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

        event_bus.emit("review_start", 0, "", {
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

        event_bus.emit("review_complete", 0, "", {
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

        from infinidev.engine.formats.tool_call_parser import safe_json_loads
        try:
            data = safe_json_loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("ReviewEngine: could not parse response")
            return ReviewResult(
                verdict="SKIPPED",
                summary="Could not parse review response",
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


def run_review_rework_loop(
    *,
    engine: Any,
    agent: Any,
    session_id: str,
    task_prompt: tuple[str, str],
    initial_result: str,
    reviewer: ReviewEngine,
    recent_messages: list[str] | None = None,
    on_status: Any | None = None,
) -> tuple[str, ReviewResult | None]:
    """Run verification + review-rework cycle.

    1. Run tests/import checks (VerificationEngine)
    2. If tests fail, feed failure output back to developer for fixes
    3. Run textual code review (ReviewEngine)
    4. If rejected, feed feedback back to developer for fixes
    5. Loop until approved or max rounds reached

    Args:
        engine: LoopEngine (or any engine with ``execute()`` and file-change methods).
        agent: The developer agent.
        session_id: Current session ID for agent context activation.
        task_prompt: Original ``(description, expected_output)`` tuple.
        initial_result: The developer's first result string.
        reviewer: ReviewEngine instance (will be reset).
        recent_messages: Optional recent conversation summaries.
        on_status: Optional ``(level: str, message: str) -> None`` callback.
            Called with levels: "verification_pass", "verification_fail",
            "approved", "rejected", "max_reviews".

    Returns:
        ``(final_result, last_review)`` — the (possibly updated) result and
        the last ReviewResult, or None if review was skipped entirely.
    """
    def _notify(level: str, msg: str) -> None:
        if on_status:
            on_status(level, msg)

    def _run_verification_and_fix(current_result: str) -> str:
        """Run tests; if they fail, re-execute developer with failure context."""
        from infinidev.engine.analysis.verification_engine import VerificationEngine

        workspace = getattr(engine, '_workspace', None)
        if not workspace:
            from infinidev.tools.base.context import get_current_workspace_path
            workspace = get_current_workspace_path()

        if not workspace:
            return current_result

        verifier = VerificationEngine(workspace=workspace)
        changed = list((engine.get_file_contents() or {}).keys())
        vresult = verifier.verify(changed_files=changed)

        if vresult.passed:
            _notify("verification_pass", vresult.summary)
            return current_result

        # Tests failed — feed output back to developer
        _notify("verification_fail", vresult.summary)
        failure_feedback = vresult.format_for_developer()
        if not failure_feedback:
            return current_result

        fix_description = (
            f"{task_prompt[0]}\n\n"
            f"## IMPORTANT: Tests are FAILING\n"
            f"You MUST fix the test failures below before proceeding.\n\n"
            f"{failure_feedback}"
        )
        fix_prompt = (fix_description, task_prompt[1])

        agent.activate_context(session_id=session_id)
        try:
            new_result = engine.execute(
                agent=agent,
                task_prompt=fix_prompt,
                verbose=True,
            )
            return new_result if new_result and new_result.strip() else current_result
        finally:
            agent.deactivate()

    reviewer.reset()
    result = initial_result

    # Run verification before review (catch real breakage first)
    result = _run_verification_and_fix(result)

    previous_feedback = ""

    while True:
        review = reviewer.review(
            task_description=task_prompt[0],
            developer_result=result,
            file_changes_summary=engine.get_changed_files_summary(),
            file_reasons=engine.get_file_change_reasons(),
            file_contents=engine.get_file_contents(),
            recent_messages=recent_messages or [],
            previous_feedback=previous_feedback,
        )

        if review.is_approved:
            _notify("approved", review.summary)
            return result, review

        if not review.is_rejected:
            # SKIPPED
            return result, review

        # Rejected
        feedback = review.format_feedback_for_developer()
        _notify("rejected", review.format_for_user())

        if not reviewer.can_review_again or not feedback:
            _notify("max_reviews", "")
            return result, review

        # Re-execute the developer with feedback prepended to the prompt
        fix_description = (
            f"{task_prompt[0]}\n\n"
            f"## IMPORTANT: Code Review Feedback\n"
            f"Your previous implementation was REJECTED by the reviewer. "
            f"You MUST address ALL blocking issues below before proceeding.\n\n"
            f"{feedback}"
        )
        fix_prompt = (fix_description, task_prompt[1])
        previous_feedback = feedback

        agent.activate_context(session_id=session_id)
        try:
            result = engine.execute(
                agent=agent,
                task_prompt=fix_prompt,
                verbose=True,
            )
            if not result or not result.strip():
                result = "Done. (no additional output)"
        finally:
            agent.deactivate()
