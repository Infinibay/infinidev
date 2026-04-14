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


# Multi-pass temperatures: extraction is maximally deterministic; the
# judge matches the long-standing single-pass value so verdicts stay
# consistent when we fall back.
EXTRACTOR_TEMPERATURE = 0.0
JUDGE_TEMPERATURE = 0.2


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
        plan_steps: list[dict] | None = None,
        automated_checks: dict[str, Any] | None = None,
        pre_extraction: dict | None = None,
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
            plan_steps: Ordered list of plan step dicts (title + explanation).
                When provided, the reviewer is asked to verify plan fidelity.
            automated_checks: Pre-computed deterministic check results keyed
                by check name. Shape: ``{"orphaned_references": [...],
                "missing_docstrings": [...], "verification_passed": bool}``.
                Items with severity='error' are BLOCKING by definition.
            pre_extraction: Pre-computed Pass A output. When provided, the
                reviewer skips its own extraction call and goes straight to
                judgment. Used by run_review_rework_loop to parallelize
                Pass A with ``collect_automated_checks``.

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

        plan_steps = plan_steps or []
        automated_checks = automated_checks or {}
        file_reasons = file_reasons or {}
        file_contents = file_contents or {}
        recent_messages = recent_messages or []

        # Dispatch: multi-pass for complex diffs, single-pass otherwise.
        # ``pre_extraction`` lets run_review_rework_loop parallelize Pass A
        # with collect_automated_checks — when supplied, we skip the
        # extraction call and go straight to judgment.
        result: ReviewResult | None = None
        use_multi_pass = (
            pre_extraction is not None
            or self._should_multi_pass(file_changes_summary, file_contents)
        )
        if use_multi_pass:
            extraction = pre_extraction
            if extraction is None:
                extraction = self._run_extraction_pass(
                    task_description=task_description,
                    developer_result=developer_result,
                    file_changes_summary=file_changes_summary,
                    file_contents=file_contents,
                    plan_steps=plan_steps,
                )
            if extraction is not None:
                result = self._run_judgment_pass(
                    llm_params=llm_params,
                    extraction=extraction,
                    task_description=task_description,
                    developer_result=developer_result,
                    plan_steps=plan_steps,
                    automated_checks=automated_checks,
                    previous_feedback=previous_feedback,
                    recent_messages=recent_messages,
                )
            else:
                logger.info("ReviewEngine: extraction pass failed, falling back to single-pass")

        if result is None:
            result = self._single_pass_review(
                llm_params=llm_params,
                task_description=task_description,
                developer_result=developer_result,
                file_changes_summary=file_changes_summary,
                previous_feedback=previous_feedback,
                file_reasons=file_reasons,
                file_contents=file_contents,
                recent_messages=recent_messages,
                plan_steps=plan_steps,
                automated_checks=automated_checks,
            )

        event_bus.emit("review_complete", 0, "", {
            "verdict": result.verdict,
            "round": self._review_count,
            "issue_count": len(result.issues),
        })

        return result

    # ── Dispatch helpers ────────────────────────────────────────────────

    @staticmethod
    def _compute_complexity(
        file_changes_summary: str,
        file_contents: dict[str, str] | None,
    ) -> int:
        """Score = changed_lines + 50 * changed_files.

        `changed_lines` counts non-empty lines in the unified diff (a rough
        but robust proxy). `changed_files` uses the file_contents dict
        when available, else counts ``diff --git`` / ``--- a/`` headers.
        """
        lines = sum(
            1 for ln in file_changes_summary.splitlines()
            if ln.strip() and not ln.startswith("@@")
        )
        if file_contents:
            file_count = len(file_contents)
        else:
            # Fallback: count diff headers.
            file_count = file_changes_summary.count("\n--- a/") + (
                1 if file_changes_summary.startswith("--- a/") else 0
            )
            if file_count == 0:
                file_count = max(1, file_changes_summary.count("diff --git"))
        return lines + 50 * file_count

    def _should_multi_pass(
        self,
        file_changes_summary: str,
        file_contents: dict[str, str] | None,
    ) -> bool:
        """Decide single vs multi based on REVIEW_MULTI_PASS_MODE + threshold."""
        from infinidev.config.settings import settings as _settings
        mode = (getattr(_settings, "REVIEW_MULTI_PASS_MODE", "auto") or "auto").lower()
        if mode == "off":
            return False
        if mode == "always":
            return True
        threshold = int(getattr(_settings, "REVIEW_MULTI_PASS_COMPLEXITY_THRESHOLD", 400))
        return self._compute_complexity(file_changes_summary, file_contents) > threshold

    # ── Pass A: extraction ──────────────────────────────────────────────

    def _run_extraction_pass(
        self,
        *,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        file_contents: dict[str, str],
        plan_steps: list[dict],
    ) -> dict | None:
        """Call Pass A LLM. Returns parsed JSON, or None on failure.

        One retry on malformed JSON with a 'return valid JSON' reminder.
        Any exception falls through as None so the caller can fall back to
        single-pass.
        """
        from infinidev.config.llm import get_litellm_params_for_review_extractor
        from infinidev.prompts.reviewer.extractor_system import EXTRACTOR_SYSTEM_PROMPT

        try:
            ext_params = get_litellm_params_for_review_extractor()
        except Exception as exc:
            logger.warning("ReviewEngine: extractor LLM params unavailable (%s)", exc)
            return None

        user_prompt = self._build_extractor_prompt(
            task_description=task_description,
            developer_result=developer_result,
            file_changes_summary=file_changes_summary,
            file_contents=file_contents,
            plan_steps=plan_steps,
        )
        messages = [
            {"role": "system", "content": EXTRACTOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        for attempt in range(2):
            try:
                import litellm
                response = litellm.completion(
                    messages=messages,
                    **ext_params,
                    temperature=EXTRACTOR_TEMPERATURE,
                )
                raw = (response.choices[0].message.content or "").strip()
                parsed = self._parse_extraction(raw)
                if parsed is not None:
                    return parsed
            except Exception as exc:
                logger.warning("ReviewEngine: extractor LLM call failed: %s", exc)
                return None
            # Retry once with an explicit reminder.
            messages = messages + [
                {"role": "user", "content": "Your previous reply was not valid JSON. Return ONLY the JSON object specified in the schema."},
            ]

        return None

    @staticmethod
    def _parse_extraction(raw: str) -> dict | None:
        """Parse the extractor's JSON output, tolerating markdown fences."""
        if not raw:
            return None
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            keep: list[str] = []
            in_block = False
            for ln in lines:
                if ln.strip().startswith("```") and not in_block:
                    in_block = True
                    continue
                if ln.strip() == "```" and in_block:
                    break
                if in_block:
                    keep.append(ln)
            text = "\n".join(keep) if keep else text
        from infinidev.engine.formats.tool_call_parser import safe_json_loads
        try:
            data = safe_json_loads(text)
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(data, dict):
            return None
        return data

    def _build_extractor_prompt(
        self,
        *,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        file_contents: dict[str, str],
        plan_steps: list[dict],
    ) -> str:
        """Build the user prompt for the extractor pass."""
        parts: list[str] = []
        parts.append(f"## Original Task\n{task_description}")

        if plan_steps:
            plan_lines = ["## Plan"]
            for s in plan_steps:
                num = s.get("step", "?")
                title = s.get("title") or s.get("explanation", "")
                files = s.get("files") or []
                files_str = f" [{', '.join(files)}]" if files else ""
                plan_lines.append(f"{num}. {title}{files_str}")
                detail = s.get("explanation", "")
                if detail and detail != title:
                    plan_lines.append(f"   → {detail[:300]}")
            parts.append("\n".join(plan_lines))

        parts.append(f"## Developer's Report\n{developer_result}")

        if file_contents:
            parts.append("## Current File Contents")
            for path in sorted(file_contents.keys()):
                content = file_contents[path]
                max_chars = 50_000
                if len(content) > max_chars:
                    content = content[:max_chars] + f"\n... (truncated, {len(content)} total chars)"
                ext = path.rsplit(".", 1)[-1] if "." in path else ""
                parts.append(f"### `{path}`\n```{ext}\n{content}\n```")

        parts.append(f"## Diffs\n{file_changes_summary}")
        return "\n\n".join(parts)

    # ── Pass B: judgment ────────────────────────────────────────────────

    def _run_judgment_pass(
        self,
        *,
        llm_params: dict,
        extraction: dict,
        task_description: str,
        developer_result: str,
        plan_steps: list[dict],
        automated_checks: dict,
        previous_feedback: str,
        recent_messages: list[str],
    ) -> ReviewResult:
        """Call Pass B LLM with the extraction + checks (no diffs)."""
        from infinidev.prompts.reviewer.judge_system import JUDGE_SYSTEM_PROMPT

        user_prompt = self._build_judge_prompt(
            extraction=extraction,
            task_description=task_description,
            developer_result=developer_result,
            plan_steps=plan_steps,
            automated_checks=automated_checks,
            previous_feedback=previous_feedback,
            recent_messages=recent_messages,
        )
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            import litellm
            response = litellm.completion(
                messages=messages,
                **llm_params,
                temperature=JUDGE_TEMPERATURE,
            )
            raw_content = response.choices[0].message.content or ""
            return self._parse_response(raw_content)
        except Exception as exc:
            logger.warning("ReviewEngine: judge LLM call failed (%s), skipping", exc)
            return ReviewResult(
                verdict="SKIPPED",
                summary=f"Judge skipped due to error: {exc}",
            )

    def _build_judge_prompt(
        self,
        *,
        extraction: dict,
        task_description: str,
        developer_result: str,
        plan_steps: list[dict],
        automated_checks: dict,
        previous_feedback: str,
        recent_messages: list[str],
    ) -> str:
        """Build the user prompt for the judge pass.

        Crucially, this does NOT include the raw diffs or file contents —
        the extraction is the authoritative summary the judge works from.
        """
        parts: list[str] = []

        if recent_messages:
            parts.append("## Conversation Context")
            for i, msg in enumerate(recent_messages):
                if i == len(recent_messages) - 1:
                    parts.append(f">>> CURRENT REQUEST <<<\n{msg}")
                else:
                    parts.append(msg)

        parts.append(f"## Original Task\n{task_description}")

        if plan_steps:
            plan_lines = ["## Plan (what the developer committed to)"]
            for s in plan_steps:
                num = s.get("step", "?")
                title = s.get("title") or s.get("explanation", "")
                files = s.get("files") or []
                files_str = f" [{', '.join(files)}]" if files else ""
                plan_lines.append(f"{num}. {title}{files_str}")
                detail = s.get("explanation", "")
                if detail and detail != title:
                    plan_lines.append(f"   → {detail[:300]}")
            parts.append("\n".join(plan_lines))

        if automated_checks:
            parts.append(self._format_automated_checks(automated_checks))

        parts.append("## Extraction (authoritative factual summary)")
        parts.append("```json\n" + json.dumps(extraction, indent=2) + "\n```")

        parts.append(f"## Developer's Report\n{developer_result}")

        if previous_feedback:
            parts.append(
                f"## Previous Review Feedback (Round {self._review_count})\n"
                f"The developer was asked to fix these issues. Verify they were addressed:\n"
                f"{previous_feedback}"
            )

        return "\n\n".join(parts)

    # ── Single-pass (legacy, also fallback) ─────────────────────────────

    def _single_pass_review(
        self,
        *,
        llm_params: dict,
        task_description: str,
        developer_result: str,
        file_changes_summary: str,
        previous_feedback: str,
        file_reasons: dict[str, list[str]],
        file_contents: dict[str, str],
        recent_messages: list[str],
        plan_steps: list[dict],
        automated_checks: dict,
    ) -> ReviewResult:
        """Classic one-shot reviewer. Also used as fallback from multi-pass."""
        from infinidev.prompts.reviewer.system import REVIEWER_SYSTEM_PROMPT

        user_prompt = self._build_review_prompt(
            task_description, developer_result,
            file_changes_summary, previous_feedback,
            file_reasons=file_reasons,
            file_contents=file_contents,
            recent_messages=recent_messages,
            plan_steps=plan_steps,
            automated_checks=automated_checks,
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
                temperature=JUDGE_TEMPERATURE,
            )
            raw_content = response.choices[0].message.content or ""
            return self._parse_response(raw_content)
        except Exception as exc:
            logger.warning("ReviewEngine: LLM call failed (%s), skipping review", exc)
            return ReviewResult(
                verdict="SKIPPED",
                summary=f"Review skipped due to error: {exc}",
            )

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
        plan_steps: list[dict] | None = None,
        automated_checks: dict[str, Any] | None = None,
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

        # Plan (what the developer committed to doing, step-by-step)
        if plan_steps:
            plan_lines = ["## Plan (what the developer committed to)"]
            for s in plan_steps:
                num = s.get("step", "?")
                title = s.get("title") or s.get("explanation", "")
                files = s.get("files") or []
                files_str = f" [{', '.join(files)}]" if files else ""
                plan_lines.append(f"{num}. {title}{files_str}")
                detail = s.get("explanation", "")
                if detail and detail != title:
                    plan_lines.append(f"   → {detail[:300]}")
            parts.append("\n".join(plan_lines))

        # Automated checks (deterministic evidence — pre-classified severity)
        if automated_checks:
            parts.append(self._format_automated_checks(automated_checks))

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

    @staticmethod
    def _format_automated_checks(checks: dict[str, Any]) -> str:
        """Render automated check results as a compact, high-signal block.

        The reviewer is told to trust these — they came from deterministic
        tooling (tree-sitter index queries), not from another LLM.
        """
        lines = ["## Automated Checks (deterministic, trust these)"]

        verif = checks.get("verification_passed")
        if verif is True:
            lines.append("- tests/import-check: PASSED")
        elif verif is False:
            lines.append("- tests/import-check: FAILED (treat as blocking)")

        orphaned = checks.get("orphaned_references") or []
        lines.append(
            f"- orphaned_references: {len(orphaned)} "
            f"(BLOCKING — symbols removed but still referenced)"
        )
        for item in orphaned[:10]:
            lines.append(f"    • {item.get('message', '')}")
        if len(orphaned) > 10:
            lines.append(f"    • ... {len(orphaned) - 10} more")

        missing = checks.get("missing_docstrings") or []
        lines.append(
            f"- missing_docstrings: {len(missing)} "
            f"(suggestion — flag in notes, not sole reason to reject)"
        )
        for item in missing[:5]:
            lines.append(f"    • {item.get('message', '')}")
        if len(missing) > 5:
            lines.append(f"    • ... {len(missing) - 5} more")

        return "\n".join(lines)


def collect_automated_checks(
    changed_files: list[str],
    file_tracker: Any = None,
    verification_passed: bool | None = None,
) -> dict[str, Any]:
    """Gather deterministic check results to feed into the reviewer.

    Runs orphaned-reference and missing-docstring checks against the
    code-intel index for the given changed files. Resilient: any single
    check that errors returns an empty list for that key.
    """
    from infinidev.code_intel.analyzer import (
        check_missing_docstrings, check_orphaned_references,
    )
    from infinidev.code_intel.smart_index import ensure_indexed

    project_id = 1  # CLI uses a single project by default
    result: dict[str, Any] = {
        "verification_passed": verification_passed,
        "orphaned_references": [],
        "missing_docstrings": [],
    }

    # Reindex changed files so the checks see post-edit state
    import os as _os
    abs_changed = [_os.path.abspath(p) for p in (changed_files or [])]
    for p in abs_changed:
        try:
            ensure_indexed(project_id, p)
        except Exception as exc:
            logger.debug("Reindex failed for %s before checks: %s", p, exc)

    # Orphaned references — only if tracker recorded deletions
    if file_tracker is not None:
        try:
            deleted = file_tracker.get_deleted_symbols()
            if deleted:
                # Make sure source files are reindexed too
                for src in deleted.keys():
                    try:
                        ensure_indexed(project_id, src)
                    except Exception:
                        pass
                diags = check_orphaned_references(project_id, deleted)
                result["orphaned_references"] = [
                    {"file": d.file_path, "line": d.line, "message": d.message}
                    for d in diags
                ]
        except Exception as exc:
            logger.debug("Orphaned references collection failed: %s", exc)

    # Missing docstrings — one query per changed file
    for p in abs_changed:
        try:
            diags = check_missing_docstrings(project_id, p)
            for d in diags:
                result["missing_docstrings"].append(
                    {"file": d.file_path, "line": d.line, "message": d.message}
                )
        except Exception as exc:
            logger.debug("Missing-docstrings check failed for %s: %s", p, exc)

    return result


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
        tracker = getattr(engine, 'get_file_tracker', lambda: None)()
        vresult = verifier.verify(changed_files=changed, file_tracker=tracker)

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
    plan_steps = getattr(engine, "get_plan_steps", lambda: [])()

    while True:
        changed_files = list((engine.get_file_contents() or {}).keys())
        tracker = getattr(engine, "get_file_tracker", lambda: None)()
        file_changes_summary = engine.get_changed_files_summary()
        file_contents = engine.get_file_contents() or {}

        pre_extraction: dict | None = None
        # When multi-pass is in play, extraction (LLM-bound) and
        # collect_automated_checks (disk+SQLite-bound) are independent —
        # run both at once to save a couple of seconds per review round.
        if reviewer._should_multi_pass(file_changes_summary, file_contents):
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as ex:
                ext_future = ex.submit(
                    reviewer._run_extraction_pass,
                    task_description=task_prompt[0],
                    developer_result=result,
                    file_changes_summary=file_changes_summary,
                    file_contents=file_contents,
                    plan_steps=plan_steps,
                )
                checks_future = ex.submit(
                    collect_automated_checks,
                    changed_files=changed_files,
                    file_tracker=tracker,
                    verification_passed=True,
                )
                pre_extraction = ext_future.result()
                automated = checks_future.result()
        else:
            automated = collect_automated_checks(
                changed_files=changed_files,
                file_tracker=tracker,
                verification_passed=True,
            )

        review = reviewer.review(
            task_description=task_prompt[0],
            developer_result=result,
            file_changes_summary=file_changes_summary,
            file_reasons=engine.get_file_change_reasons(),
            file_contents=file_contents,
            recent_messages=recent_messages or [],
            previous_feedback=previous_feedback,
            plan_steps=plan_steps,
            automated_checks=automated,
            pre_extraction=pre_extraction,
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
