"""Auditable category manifest for conditional-prompt mini-model corpora."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


MIN_EXAMPLES_PER_CATEGORY = 20
TARGET_POSITIVES_PER_CATEGORY = 48
TARGET_NEGATIVES_PER_CATEGORY = 64
CATALOG_VERSION = "conditional-prompt-example-catalog-v1"


@dataclass(frozen=True)
class ExampleCategory:
    """One independently audited semantic or safety distinction."""

    id: str
    detector: str
    label: str
    negative_strategy: str
    positive_target: int = TARGET_POSITIVES_PER_CATEGORY
    negative_target: int = TARGET_NEGATIVES_PER_CATEGORY


_TASK_METHODS = (
    "bugfix.root_cause",
    "feature.contract_first",
    "refactor.preserve_behavior",
    "research.evidence_first",
    "review.read_only",
    "performance.measure_first",
)
_COMPOUND_METHODS = (
    "bugfix.root_cause+refactor.preserve_behavior",
    "feature.contract_first+refactor.preserve_behavior",
    "bugfix.root_cause+performance.measure_first",
    "feature.contract_first+research.evidence_first",
    "bugfix.root_cause+research.evidence_first",
    "feature.contract_first+performance.measure_first",
    "research.evidence_first+review.read_only",
)
_DISCOURSE_LABELS = (
    "acknowledgement",
    "quoted_action",
    "conceptual_question",
    "status_only",
    "hypothetical_future",
    "explanation_only",
    "ambiguous_method",
    "out_of_domain",
)
_AUTHORITY_LABELS = (
    "answer_only",
    "diagnose_only",
    "modify",
    "read_only",
    "commit",
    "publish",
    "negated_or_quoted_action",
)
_REASONING_LABELS = (
    "excessive_exploration",
    "retry_loop",
    "premature_completion",
    "speculative_claim",
    "verification_gap",
    "healthy_progress",
    "uncategorized",
)
_MESSAGE_LABELS = (
    "evidence_free_completion",
    "avoidable_user_question",
    "repeated_hypothesis",
    "unsupported_claim",
    "healthy_progress",
    "uncategorized",
)


CATEGORIES = (
    *(ExampleCategory(
        id=f"task.single.{label}", detector="task_method", label=label,
        negative_strategy="explicit_hard_negatives",
    ) for label in _TASK_METHODS),
    *(ExampleCategory(
        id=f"task.compound.{label.replace('.', '_').replace('+', '__')}",
        detector="task_compound", label=label,
        negative_strategy="other_compound_categories",
    ) for label in _COMPOUND_METHODS),
    *(ExampleCategory(
        id=f"task.discourse.{label}", detector="task_discourse", label=label,
        negative_strategy="other_discourse_categories_and_tasks",
    ) for label in _DISCOURSE_LABELS),
    *(ExampleCategory(
        id=f"authority.{label}", detector="literal_authority", label=label,
        negative_strategy="other_authority_categories",
        positive_target=256 if label == "answer_only" else TARGET_POSITIVES_PER_CATEGORY,
    ) for label in _AUTHORITY_LABELS),
    *(ExampleCategory(
        id=f"reasoning.{label}", detector="reasoning_head", label=label,
        negative_strategy="other_reasoning_categories",
        positive_target=256 if label == "uncategorized" else TARGET_POSITIVES_PER_CATEGORY,
    ) for label in _REASONING_LABELS),
    *(ExampleCategory(
        id=f"message.{label}", detector="message_head", label=label,
        negative_strategy="other_message_categories",
        positive_target=256 if label == "uncategorized" else TARGET_POSITIVES_PER_CATEGORY,
    ) for label in _MESSAGE_LABELS),
)


def audit_example_catalog() -> dict[str, object]:
    """Materialize every category and report positive and negative coverage."""
    from bench.authority_example_corpus import build_authority_corpus
    from bench.message_pattern_corpus import build_message_pattern_corpus
    from bench.reasoning_pattern_head import build_corpus as build_reasoning_corpus
    from bench.task_policy_compound_corpus import build_compound_corpus
    from bench.task_policy_discourse_corpus import build_discourse_corpus
    from infinidev.engine.task_policies.semantic_prototypes import PROTOTYPES

    positives: Counter[str] = Counter()
    explicit_negatives: Counter[str] = Counter()

    for label, prototypes in PROTOTYPES.items():
        positives[f"task.single.{label}"] = len(prototypes.positive)
        explicit_negatives[f"task.single.{label}"] = len(prototypes.negative)

    for example in build_compound_corpus("calibration"):
        label = "+".join(example.policies).replace(".", "_").replace("+", "__")
        positives[f"task.compound.{label}"] += 1
    for example in build_discourse_corpus("calibration"):
        positives[f"task.discourse.{example.category}"] += 1
    for example in build_authority_corpus():
        positives[f"authority.{example.category}"] += 1
    for example in build_reasoning_corpus("calibration"):
        positives[f"reasoning.{example.label}"] += 1
    for example in build_message_pattern_corpus():
        positives[f"message.{example.category}"] += 1

    group_totals = Counter()
    for category in CATEGORIES:
        group_totals[category.detector] += positives[category.id]

    rows = []
    for category in CATEGORIES:
        positive_count = positives[category.id]
        negative_count = explicit_negatives[category.id]
        if not negative_count:
            negative_count = group_totals[category.detector] - positive_count
        rows.append({
            "id": category.id,
            "detector": category.detector,
            "label": category.label,
            "negative_strategy": category.negative_strategy,
            "positives": positive_count,
            "negatives": negative_count,
            "minimum": MIN_EXAMPLES_PER_CATEGORY,
            "positive_target": category.positive_target,
            "negative_target": category.negative_target,
            "meets_minimum": (
                positive_count >= MIN_EXAMPLES_PER_CATEGORY
                and negative_count >= MIN_EXAMPLES_PER_CATEGORY
            ),
            "meets_targets": (
                positive_count >= category.positive_target
                and negative_count >= category.negative_target
            ),
        })
    return {
        "version": CATALOG_VERSION,
        "categories": len(CATEGORIES),
        "minimum": MIN_EXAMPLES_PER_CATEGORY,
        "positive_target": TARGET_POSITIVES_PER_CATEGORY,
        "negative_target": TARGET_NEGATIVES_PER_CATEGORY,
        "rows": rows,
        "below_minimum": [row["id"] for row in rows if not row["meets_minimum"]],
        "below_target": [row["id"] for row in rows if not row["meets_targets"]],
    }


__all__ = [
    "CATALOG_VERSION",
    "CATEGORIES",
    "MIN_EXAMPLES_PER_CATEGORY",
    "TARGET_NEGATIVES_PER_CATEGORY",
    "TARGET_POSITIVES_PER_CATEGORY",
    "ExampleCategory",
    "audit_example_catalog",
]
