"""Evidence gate for research, analysis, and other non-code outcomes."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EvidenceReviewResult:
    """Grounded judgment over an informational developer response."""

    verdict: str
    summary: str = ""
    issues: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_rejected(self) -> bool:
        return self.verdict == "REJECTED"

    def format_feedback_for_developer(self) -> str:
        if not self.is_rejected:
            return ""
        lines = [
            "## Evidence review — REJECTED",
            "",
            self.summary,
            "",
            "Correct only the grounded issues below. Preserve the original scope.",
        ]
        for index, issue in enumerate(self.issues, start=1):
            if issue.get("severity") != "blocking":
                continue
            lines.extend([
                "",
                f"### Issue {index}: {issue.get('category', 'unsupported_claim')}",
                f"- Response excerpt: {issue.get('claim_excerpt', '')}",
                f"- Problem: {issue.get('problem', '')}",
                f"- Evidence: {issue.get('evidence', '')}",
                f"- Minimal fix: {issue.get('fix', '')}",
            ])
        return "\n".join(lines)


class EvidenceReviewEngine:
    """Use an independent LLM pass, then ground its issues in the answer."""

    def review(
        self,
        *,
        task_description: str,
        developer_result: str,
        evidence: str,
        acceptance_criteria: list[str] | None = None,
        derived_verification_criteria: list[str] | None = None,
    ) -> EvidenceReviewResult:
        if not developer_result.strip():
            return EvidenceReviewResult(
                verdict="REJECTED",
                summary="The developer returned no substantive response.",
                issues=[{
                    "severity": "blocking",
                    "category": "instruction_miss",
                    "claim_excerpt": "(empty response)",
                    "problem": "There is no result to evaluate or deliver.",
                    "evidence": "The submitted response is empty.",
                    "fix": "Provide the requested evidence-backed result.",
                    "grounded": True,
                }],
            )

        from infinidev.config.llm import get_litellm_params

        params = get_litellm_params()
        if params is None:
            return EvidenceReviewResult("SKIPPED", "No reviewer LLM configured.")

        from infinidev.prompts.reviewer.evidence_system import (
            EVIDENCE_REVIEW_SYSTEM_PROMPT,
        )

        user_prompt = self._build_prompt(
            task_description=task_description,
            developer_result=developer_result,
            evidence=evidence,
            acceptance_criteria=acceptance_criteria or [],
            derived_verification_criteria=derived_verification_criteria or [],
        )
        messages = [
            {"role": "system", "content": EVIDENCE_REVIEW_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            from infinidev.engine.analysis.review_engine import ReviewEngine
            from infinidev.config.settings import settings

            response = ReviewEngine._completion_with_caching(
                params,
                messages,
                0.0,
                settings.LLM_PROVIDER or "",
            )
            raw = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Evidence review LLM call failed: %s", exc)
            return EvidenceReviewResult("SKIPPED", "Evidence reviewer call failed.")

        parsed = self._parse(raw)
        if parsed is None:
            return EvidenceReviewResult("SKIPPED", "Evidence reviewer returned invalid JSON.")
        return self._ground(parsed, developer_result)

    @staticmethod
    def _build_prompt(
        *,
        task_description: str,
        developer_result: str,
        evidence: str,
        acceptance_criteria: list[str],
        derived_verification_criteria: list[str],
    ) -> str:
        user_criteria = "\n".join(f"- {item}" for item in acceptance_criteria) or "(none)"
        derived = (
            "\n".join(f"- {item}" for item in derived_verification_criteria)
            or "(none)"
        )
        return (
            "## Original request\n"
            f"{task_description}\n\n"
            "## User-authored acceptance criteria\n"
            f"{user_criteria}\n\n"
            "## Planner-derived checks (not user requirements)\n"
            f"{derived}\n\n"
            "## Archived tool evidence\n"
            f"{evidence or '(no archived tool evidence was available)'}\n\n"
            "## Submitted response\n"
            f"{developer_result}"
        )

    @staticmethod
    def _parse(raw: str) -> dict[str, Any] | None:
        from infinidev.engine.formats.tool_call_parser import safe_json_loads

        try:
            value = safe_json_loads(raw)
        except Exception:
            return None
        if not isinstance(value, dict) or not isinstance(value.get("issues", []), list):
            return None
        return value

    @staticmethod
    def _ground(data: dict[str, Any], developer_result: str) -> EvidenceReviewResult:
        """Demote reviewer allegations not anchored in the submitted answer."""

        grounded: list[dict[str, Any]] = []
        for raw_issue in data.get("issues", []):
            if not isinstance(raw_issue, dict):
                continue
            issue = dict(raw_issue)
            excerpt = str(issue.get("claim_excerpt") or "").strip()
            is_grounded = bool(excerpt and excerpt in developer_result)
            issue["grounded"] = is_grounded
            if issue.get("severity") == "blocking" and not is_grounded:
                issue["severity"] = "suggestion"
                issue["problem"] = (
                    "Reviewer allegation was not grounded in an exact response excerpt. "
                    + str(issue.get("problem") or "")
                ).strip()
            grounded.append(issue)

        blocking = [item for item in grounded if item.get("severity") == "blocking"]
        verdict = "REJECTED" if blocking else "APPROVED"
        return EvidenceReviewResult(
            verdict=verdict,
            summary=str(data.get("summary") or "").strip(),
            issues=grounded,
        )


def _recent_tool_evidence(session_id: str, *, limit: int = 20, max_chars: int = 24000) -> str:
    """Render recent raw tool records with explicit archive provenance."""

    try:
        from infinidev.engine.working_memory import get_working_memory

        records = get_working_memory(session_id).recent_records(
            limit=limit,
            kinds={"tool_output", "artifact_analysis", "auto_note"},
        )
    except Exception:
        logger.debug("Could not load evidence-review archive", exc_info=True)
        return ""

    parts: list[str] = []
    total = 0
    for record in records:
        rendered = record.render(max_chars=3000)
        if total + len(rendered) > max_chars:
            break
        parts.append(rendered)
        total += len(rendered)
    return "\n\n".join(parts)


def run_evidence_review_rework_loop(
    *,
    engine: Any,
    agent: Any,
    session_id: str,
    task_prompt: tuple[str, str],
    initial_result: str,
    on_status: Any | None = None,
    acceptance_criteria: list[str] | None = None,
    derived_verification_criteria: list[str] | None = None,
    evidence_reviewer: EvidenceReviewEngine | None = None,
    task: Any | None = None,
    max_iterations: int | None = None,
    max_total_tool_calls: int | None = None,
) -> tuple[str, EvidenceReviewResult | None]:
    """Review and minimally rework informational output within a bounded loop."""

    from infinidev.config.settings import settings

    reviewer = evidence_reviewer or EvidenceReviewEngine()
    current = initial_result
    last: EvidenceReviewResult | None = None
    max_rounds = max(1, int(settings.EVIDENCE_REVIEW_MAX_ROUNDS))

    for round_index in range(max_rounds):
        evidence = _recent_tool_evidence(session_id)
        last = reviewer.review(
            task_description=task_prompt[0],
            developer_result=current,
            evidence=evidence,
            acceptance_criteria=acceptance_criteria,
            derived_verification_criteria=derived_verification_criteria,
        )
        if last.verdict in {"APPROVED", "SKIPPED"}:
            if on_status:
                on_status(last.verdict.lower(), last.summary)
            return current, last
        if on_status:
            on_status("rejected", last.summary)
        if round_index + 1 >= max_rounds:
            if on_status:
                on_status("max_reviews", "Evidence review reached its retry limit.")
            break

        feedback = last.format_feedback_for_developer()
        rework_description = (
            f"{task_prompt[0]}\n\n"
            "## Evidence-review rework — preserve the original objective\n"
            "Correct only claims identified below. Do not add new scope, perform new "
            "external actions, or present reviewer suggestions as user requirements. "
            "Use the existing evidence when sufficient; gather more only when required "
            "to answer the original request. Clearly label inference and uncertainty.\n\n"
            f"{feedback}"
        )
        agent.activate_context(session_id=session_id)
        try:
            rework_kwargs: dict[str, Any] = {}
            if task is not None:
                rework_kwargs.update(
                    task=task,
                    preserve_task_state=True,
                    max_iterations=max_iterations,
                    max_total_tool_calls=max_total_tool_calls,
                )
            updated = engine.execute(
                agent=agent,
                task_prompt=(rework_description, task_prompt[1]),
                verbose=True,
                preserve_file_tracker=True,
                **rework_kwargs,
            )
        finally:
            agent.deactivate()
        if updated and updated.strip():
            current = updated

    return current, last
