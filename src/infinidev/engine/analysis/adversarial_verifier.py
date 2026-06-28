"""AdversarialVerifier — cited-evidence LLM judge for soft objectives.

Some objectives have no deterministic check ("error messages are clearer",
"the refactor removes the duplication", "the API reads naturally"). The
deterministic ``ObjectiveVerifier`` can't touch those, so today they fall
back to pure self-attestation. This verifier closes that gap WITHOUT
reinventing the sycophantic self-grading the project already distrusts:

  * INDEPENDENT context — it sees ONLY the objective + the changed code,
    never the developer's narrative/reasoning, so it can't inherit the
    worker's framing.
  * ADVERSARIAL stance — it is told to ASSUME the objective FAILED and to
    find verbatim evidence it PASSED, flipping the LLM's default agreement
    bias toward refutation.
  * GROUNDED verdict — a PASS must quote exact text, and that quote is
    substring-checked against the changed files. A quote that isn't really
    there is demoted to FAIL, so the judge cannot hallucinate a green check.
  * HONEST about ignorance — an objective it genuinely cannot judge from
    code returns UNVERIFIABLE (surfaced, not silently passed or endlessly
    reworked).

Runs only at task end (post-loop), never per step — one LLM call per soft
objective per re-verification round.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable

from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.verification_result import VerificationResult

logger = logging.getLogger(__name__)

_MAX_CONTEXT_CHARS = 12000  # budget for changed-file contents in the prompt
_MIN_EVIDENCE_CHARS = 8  # quotes shorter than this can't ground a PASS

_SYSTEM_PROMPT = (
    "You are a SKEPTICAL, INDEPENDENT verifier. You did NOT write this code "
    "and you owe it no benefit of the doubt. Your DEFAULT assumption is that "
    "the objective was NOT met.\n\n"
    "Your job: decide whether the objective is satisfied by the CHANGED CODE "
    "shown, and back it with VERBATIM evidence.\n"
    "- verdict PASS: only if you can quote EXACT text from the changes that "
    "demonstrably satisfies the objective. Quote a substantial, distinctive "
    "span (a full line or clause, not a single short token) so the evidence "
    "can be located unambiguously. Put that exact text in cited_evidence. "
    "No quote → not a PASS.\n"
    "- verdict FAIL: the evidence is missing, partial, or you would have to "
    "assume/infer it.\n"
    "- verdict UNVERIFIABLE: the objective genuinely cannot be judged from "
    "code alone (needs a human, a running system, or external context).\n\n"
    "Be harsh. Never output PASS without a verbatim quote from the changes. "
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"verdict": "PASS|FAIL|UNVERIFIABLE", "cited_evidence": "<exact quoted '
    'text from the changes, or empty>", "reason": "<one short sentence>"}'
)


class AdversarialVerifier:
    """Judge a ``llm_judge`` StepVerification in an independent context."""

    def __init__(
        self,
        workspace: str | None = None,
        llm_params: dict | None = None,
        completion_fn: Callable[[list[dict]], str] | None = None,
    ) -> None:
        self._workspace = workspace or os.getcwd()
        self._params = llm_params
        # Injectable for tests: takes messages, returns the raw model content.
        self._completion_fn = completion_fn

    def verify(
        self,
        check: StepVerification,
        *,
        changed_files: dict[str, str] | None = None,
        diff_summary: str = "",
    ) -> VerificationResult:
        changed_files = changed_files or {}
        try:
            raw = self._call(self._build_messages(check, changed_files, diff_summary))
        except Exception as exc:
            logger.warning("adversarial verifier call failed for %r: %s", check.spec[:80], exc)
            return self._unverifiable(check, f"verifier call failed: {exc}")

        data = self._parse(raw)
        if data is None:
            return self._unverifiable(check, "verifier returned unparseable output")

        verdict = str(data.get("verdict", "")).strip().upper()
        cited = str(data.get("cited_evidence") or "").strip()
        reason = str(data.get("reason") or "").strip()

        if verdict == "PASS":
            if not self._evidence_grounded(cited, changed_files):
                # The quote isn't actually in the changes → the judge invented
                # it. Demote to FAIL rather than trust a hallucinated PASS.
                return self._result(
                    check, passed=False,
                    summary=f"llm_judge PASS demoted — cited evidence not found in changes",
                    detail=f"reason: {reason}\nclaimed evidence (ungrounded): {cited[:300]}",
                )
            return self._result(
                check, passed=True,
                summary="llm_judge PASS",
                detail=f"reason: {reason}\nevidence: {cited[:300]}",
            )

        if verdict == "FAIL":
            return self._result(
                check, passed=False, summary="llm_judge FAIL",
                detail=f"reason: {reason}",
            )

        # UNVERIFIABLE or any unexpected verdict → surface, don't block.
        return self._unverifiable(check, reason or f"verdict={verdict!r}")

    # ── prompt + call ────────────────────────────────────────────────────

    def _build_messages(
        self, check: StepVerification, changed_files: dict[str, str], diff_summary: str,
    ) -> list[dict]:
        where = check.observable.strip() or "(not specified)"
        body = self._render_changes(changed_files, diff_summary)
        user = (
            f"## Objective to verify\n{check.spec}\n\n"
            f"## Where to look (hint)\n{where}\n\n"
            f"## Changed code (current contents)\n{body}\n\n"
            "Judge ONLY from the changes above. Quote exact text for a PASS."
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    def _render_changes(self, changed_files: dict[str, str], diff_summary: str) -> str:
        if not changed_files:
            return diff_summary or "(no changed files captured)"
        parts: list[str] = []
        budget = _MAX_CONTEXT_CHARS
        for path, content in changed_files.items():
            if budget <= 0:
                parts.append("... (remaining changed files omitted for length)")
                break
            snippet = content[:budget]
            budget -= len(snippet)
            parts.append(f"### {path}\n{snippet}")
        return "\n\n".join(parts)

    def _call(self, messages: list[dict]) -> str:
        if self._completion_fn is not None:
            return self._completion_fn(messages)
        import litellm
        from infinidev.config.llm import get_litellm_params_for_assistant

        params = self._params or get_litellm_params_for_assistant()
        resp = litellm.completion(**params, messages=messages, temperature=0.0)
        return resp.choices[0].message.content or ""

    # ── parsing + grounding ──────────────────────────────────────────────

    @staticmethod
    def _parse(raw: str) -> dict | None:
        from infinidev.engine.formats.tool_call_parser import safe_json_loads
        try:
            data = safe_json_loads(raw)
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _evidence_grounded(self, cited: str, changed_files: dict[str, str]) -> bool:
        """A PASS quote must actually appear (whitespace-normalised) in the
        changed code. Too-short quotes can't carry a verdict."""
        cited_norm = self._norm(cited)
        if len(cited_norm) < _MIN_EVIDENCE_CHARS:
            return False
        haystack = self._norm("\n".join(changed_files.values()))
        return cited_norm in haystack

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    # ── result helpers ───────────────────────────────────────────────────

    @staticmethod
    def _result(check: StepVerification, *, passed: bool, summary: str, detail: str) -> VerificationResult:
        entry = {
            "command": f"(llm_judge: {check.spec[:120]})",
            "exit_code": 0 if passed else 1,
            "output": detail,
        }
        return VerificationResult(passed=passed, summary=summary, commands_run=[entry])

    @staticmethod
    def _unverifiable(check: StepVerification, reason: str) -> VerificationResult:
        entry = {
            "command": f"(llm_judge: {check.spec[:120]})",
            "exit_code": 0,
            "output": f"UNVERIFIABLE: {reason}",
        }
        return VerificationResult(
            passed=True, unverifiable=True,
            summary=f"llm_judge UNVERIFIABLE: {reason}", commands_run=[entry],
        )
