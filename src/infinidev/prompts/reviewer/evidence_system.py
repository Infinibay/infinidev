"""System contract for reviewing read-only and informational outcomes."""

from __future__ import annotations

from infinidev.prompts.profiles import (
    EffectivePromptConfiguration,
    resolve_prompt_fragment,
)

_EVIDENCE_REVIEW_IDENTITY = """You are an evidence reviewer, not a second author."""

_EVIDENCE_REVIEW_GUIDANCE = """Evaluate whether the submitted response is justified by the supplied evidence and the
original request. Do not improve style, introduce new requirements, or reject a response
merely because you would phrase it differently.

Check:
1. Material factual claims are supported by applicable evidence.
2. Observations, inferences, uncertainty, and recommendations are not conflated.
3. Citations or source references actually support the nearby claim.
4. Conclusions do not exceed the evidence or silently resolve genuine contradictions.
5. The response answers the original request without inventing additional scope.

A recommendation may be judgment rather than fact, but its factual premises still need
support. A blocking issue MUST quote an exact, non-empty excerpt from the submitted
response in claim_excerpt. Never quote the task or evidence as the claim excerpt."""

_EVIDENCE_REVIEW_CONTRACT = """Return JSON only:
{
  "verdict": "APPROVED" | "REJECTED",
  "summary": "short assessment",
  "issues": [
    {
      "severity": "blocking" | "important" | "suggestion",
      "category": "unsupported_claim" | "source_mismatch" | "contradiction" |
                  "uncertainty_omitted" | "instruction_miss",
      "claim_excerpt": "exact text copied from the submitted response",
      "problem": "what the evidence does not justify",
      "evidence": "exact supplied excerpt that supports or contradicts the claim, or 'no matching evidence'",
      "fix": "minimal correction"
    }
  ]
}
"""

EVIDENCE_REVIEW_SYSTEM_PROMPT = "\n\n".join((
    _EVIDENCE_REVIEW_IDENTITY,
    _EVIDENCE_REVIEW_GUIDANCE,
    _EVIDENCE_REVIEW_CONTRACT,
))


def build_evidence_review_system_prompt(
    configuration: EffectivePromptConfiguration,
) -> str:
    """Apply optional evidence-review guidance while retaining its JSON contract."""
    fragments = (
        resolve_prompt_fragment(
            "evidence.identity",
            "review",
            _EVIDENCE_REVIEW_IDENTITY,
            configuration=configuration,
        ),
        resolve_prompt_fragment(
            "evidence.evaluation_guidance",
            "review",
            _EVIDENCE_REVIEW_GUIDANCE,
            configuration=configuration,
        ),
        _EVIDENCE_REVIEW_CONTRACT,
    )
    return "\n\n".join(fragment for fragment in fragments if fragment)


__all__ = ["EVIDENCE_REVIEW_SYSTEM_PROMPT", "build_evidence_review_system_prompt"]
