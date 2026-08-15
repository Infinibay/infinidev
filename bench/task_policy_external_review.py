"""Load manually reviewed task-policy examples kept outside the source tree."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from bench.task_policy_multilabel_head import METHOD_LABELS, MultiLabelExample


SHORT_LABEL_TO_POLICY = {
    "bugfix": "bugfix.root_cause",
    "feature": "feature.contract_first",
    "performance": "performance.measure_first",
    "refactor": "refactor.preserve_behavior",
    "research": "research.evidence_first",
    "review": "review.read_only",
}
EXTERNAL_UNCATEGORIZED_REASONS = frozenset({
    "ambiguous_method",
    "answer_only",
    "out_of_domain",
    "unsupported_method",
})
_GENERATED_INTERFACE_MARKERS = (
    "\nNew interfaces introduced:",
    "\r\nNew interfaces introduced:",
)


def clean_external_request(text: str) -> str:
    """Remove Open-SWE augmentation that was not part of the source issue."""
    cleaned = text
    for marker in _GENERATED_INTERFACE_MARKERS:
        cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


@dataclass(frozen=True)
class ExternalCandidate:
    """One unlabeled external request used for frozen prediction only."""

    candidate_id: str
    repo: str
    language: str
    text: str


@dataclass(frozen=True)
class ExternalReview:
    """One independently reviewed external request and its provenance."""

    candidate_id: str
    repo: str
    language: str
    text: str
    policies: tuple[str, ...]
    notes: str
    uncategorized_reason: str | None = None
    annotation_kind: str = "human"
    annotation_confidence: float = 1.0

    def as_example(self) -> MultiLabelExample:
        """Return the common benchmark representation."""
        return MultiLabelExample(
            id=self.candidate_id,
            text=self.text,
            policies=self.policies,
            split="external_manual_review",
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _candidate_paths(value: Path | Iterable[Path]) -> tuple[Path, ...]:
    if isinstance(value, Path):
        return (value,)
    return tuple(value)


def load_external_candidates(
    candidates_path: Path | Iterable[Path],
) -> list[ExternalCandidate]:
    """Load unlabeled candidates without consulting review decisions."""
    candidates = []
    seen: set[str] = set()
    for path in _candidate_paths(candidates_path):
        for row in _read_jsonl(path):
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id:
                raise ValueError("external candidate is missing candidate_id")
            if candidate_id in seen:
                raise ValueError(f"duplicate external candidate: {candidate_id}")
            seen.add(candidate_id)
            source = row.get("source")
            if not isinstance(source, dict):
                raise ValueError(f"candidate {candidate_id} is missing source provenance")
            text = clean_external_request(str(row.get("issue_text", "")))
            if not text:
                raise ValueError(f"candidate {candidate_id} has empty issue_text")
            candidates.append(ExternalCandidate(
                candidate_id=candidate_id,
                repo=str(source.get("repo", "")),
                language=str(source.get("programming_language", "unknown")),
                text=text,
            ))
    return candidates


def load_external_reviews(
    candidates_path: Path | Iterable[Path],
    reviews_path: Path,
) -> list[ExternalReview]:
    """Join source candidates with explicit human decisions.

    Upstream category hints are deliberately ignored. Reviews may use the six
    short operation names for readability, but every returned example carries
    canonical policy IDs.
    """
    candidates: dict[str, dict[str, Any]] = {}
    for path in _candidate_paths(candidates_path):
        for row in _read_jsonl(path):
            candidate_id = str(row.get("candidate_id", ""))
            if not candidate_id:
                raise ValueError("external candidate is missing candidate_id")
            if candidate_id in candidates:
                raise ValueError(f"duplicate external candidate: {candidate_id}")
            candidates[candidate_id] = row

    decisions: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(reviews_path):
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id:
            raise ValueError("external review is missing candidate_id")
        if candidate_id in decisions:
            raise ValueError(f"duplicate external review: {candidate_id}")
        if candidate_id not in candidates:
            raise ValueError(f"review references unknown candidate: {candidate_id}")
        decisions[candidate_id] = row

    reviewed = []
    for candidate_id, decision in decisions.items():
        if decision.get("include") is not True:
            continue
        raw_policies = decision.get("policies")
        if not isinstance(raw_policies, list):
            raise ValueError(f"review {candidate_id} must contain a policies list")
        try:
            policies = tuple(SHORT_LABEL_TO_POLICY[str(label)] for label in raw_policies)
        except KeyError as exc:
            raise ValueError(
                f"review {candidate_id} contains unknown policy label: {exc.args[0]}"
            ) from exc
        if len(policies) != len(set(policies)):
            raise ValueError(f"review {candidate_id} repeats a policy label")
        if len(policies) > 3:
            raise ValueError(f"review {candidate_id} exceeds maximum policy cardinality")
        if any(policy not in METHOD_LABELS for policy in policies):
            raise ValueError(f"review {candidate_id} resolved to an unsupported policy")
        notes = str(decision.get("notes", "")).strip()
        if not notes:
            raise ValueError(f"review {candidate_id} needs a decision note")
        raw_reason = decision.get("uncategorized_reason")
        uncategorized_reason = str(raw_reason).strip() if raw_reason is not None else ""
        if not policies and not uncategorized_reason:
            raise ValueError(
                f"review {candidate_id} needs uncategorized_reason when policies are empty"
            )
        if not policies and uncategorized_reason not in EXTERNAL_UNCATEGORIZED_REASONS:
            raise ValueError(
                f"review {candidate_id} has unknown uncategorized_reason: "
                f"{uncategorized_reason}"
            )
        if policies and uncategorized_reason:
            raise ValueError(
                f"review {candidate_id} cannot have uncategorized_reason with policies"
            )
        candidate = candidates[candidate_id]
        source = candidate.get("source")
        if not isinstance(source, dict):
            raise ValueError(f"candidate {candidate_id} is missing source provenance")
        text = clean_external_request(str(candidate.get("issue_text", "")))
        if not text:
            raise ValueError(f"candidate {candidate_id} has empty issue_text")
        annotation = decision.get("annotation")
        annotation_kind = "human"
        annotation_confidence = 1.0
        if annotation is not None:
            if not isinstance(annotation, dict) or annotation.get("kind") != "model":
                raise ValueError(f"review {candidate_id} has invalid annotation provenance")
            confidence = annotation.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
            ):
                raise ValueError(f"review {candidate_id} has invalid annotation confidence")
            annotation_kind = "model"
            annotation_confidence = float(confidence)
        reviewed.append(ExternalReview(
            candidate_id=candidate_id,
            repo=str(source.get("repo", "")),
            language=str(source.get("programming_language", "unknown")),
            text=text,
            policies=policies,
            notes=notes,
            uncategorized_reason=uncategorized_reason or None,
            annotation_kind=annotation_kind,
            annotation_confidence=annotation_confidence,
        ))
    return reviewed


__all__ = [
    "EXTERNAL_UNCATEGORIZED_REASONS",
    "ExternalCandidate",
    "ExternalReview",
    "SHORT_LABEL_TO_POLICY",
    "clean_external_request",
    "load_external_candidates",
    "load_external_reviews",
]
