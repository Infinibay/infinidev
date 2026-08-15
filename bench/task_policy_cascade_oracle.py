"""Measure the offline ceiling of a Qwen plus MiniMax task-policy cascade."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable

from infinidev.engine.task_policies.encoder_classifier import classify_task_methods


METHOD_LABELS = (
    "bugfix.root_cause",
    "feature.contract_first",
    "refactor.preserve_behavior",
    "research.evidence_first",
    "review.read_only",
    "performance.measure_first",
)
SHORT_TO_POLICY = {label.split(".", 1)[0]: label for label in METHOD_LABELS}
Prediction = Callable[[str], Iterable[str]]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected an object")
        rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _policy_set(value: Any, *, source: str) -> frozenset[str]:
    if not isinstance(value, list):
        raise ValueError(f"{source}: policies must be a list")
    policies: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            raise ValueError(f"{source}: every policy must be a string")
        policy = SHORT_TO_POLICY.get(raw, raw)
        if policy not in METHOD_LABELS:
            raise ValueError(f"{source}: unknown policy {raw!r}")
        if policy in policies:
            raise ValueError(f"{source}: duplicate policy {raw!r}")
        policies.add(policy)
    return frozenset(policies)


def _unique_by_id(rows: Iterable[dict[str, Any]], *, source: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{source}: every row needs a candidate_id")
        if candidate_id in indexed:
            raise ValueError(f"{source}: duplicate candidate_id {candidate_id}")
        indexed[candidate_id] = row
    return indexed


def load_examples(
    candidates_path: Path,
    reviews_path: Path,
    proposals_path: Path,
) -> list[dict[str, Any]]:
    """Join included human reviews and MiniMax proposals to immutable candidate text."""
    candidates = _unique_by_id(_read_jsonl(candidates_path), source=candidates_path)
    reviews = _unique_by_id(_read_jsonl(reviews_path), source=reviews_path)
    proposals = _unique_by_id(_read_jsonl(proposals_path), source=proposals_path)
    examples = []
    for candidate_id, review in reviews.items():
        if review.get("include", True) is not True:
            continue
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"{reviews_path}: missing candidate text for {candidate_id}")
        proposal = proposals.get(candidate_id)
        if proposal is None:
            raise ValueError(f"{proposals_path}: missing MiniMax proposal for {candidate_id}")
        text = candidate.get("issue_text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{candidates_path}: candidate {candidate_id} has no issue_text")
        examples.append({
            "candidate_id": candidate_id,
            "text": text,
            "expected": _policy_set(
                review.get("policies"), source=f"{reviews_path}:{candidate_id}",
            ),
            "llm": _policy_set(
                proposal.get("policies"), source=f"{proposals_path}:{candidate_id}",
            ),
        })
    if not examples:
        raise ValueError("no included human-reviewed examples were joined")
    return examples


def _binary_metrics(
    expected: list[frozenset[str]],
    predicted: list[frozenset[str]],
    label: str,
) -> dict[str, float | int]:
    true_positive = sum(
        label in truth and label in guess for truth, guess in zip(expected, predicted)
    )
    false_positive = sum(
        label not in truth and label in guess for truth, guess in zip(expected, predicted)
    )
    false_negative = sum(
        label in truth and label not in guess for truth, guess in zip(expected, predicted)
    )
    true_negative = len(expected) - true_positive - false_positive - false_negative
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (true_positive + true_negative) / len(expected),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_positive + false_negative,
        "predicted_positive": true_positive + false_positive,
    }


def prediction_report(
    expected: list[frozenset[str]],
    predicted: list[frozenset[str]],
) -> dict[str, Any]:
    """Report exact match and the requested per-label 95/95 gate."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("expected and predicted must be equally sized and non-empty")
    per_label = {
        label: _binary_metrics(expected, predicted, label) for label in METHOD_LABELS
    }
    viable = all(
        float(metrics["accuracy"]) >= 0.95 and float(metrics["recall"]) >= 0.95
        for metrics in per_label.values()
    )
    return {
        "examples": len(expected),
        "exact_match": sum(
            truth == guess for truth, guess in zip(expected, predicted)
        ) / len(expected),
        "micro_accuracy": sum(
            (label in truth) == (label in guess)
            for truth, guess in zip(expected, predicted)
            for label in METHOD_LABELS
        ) / (len(expected) * len(METHOD_LABELS)),
        "per_label": per_label,
        "gate": {
            "accuracy_target": 0.95,
            "recall_target": 0.95,
            "all_labels_pass": viable,
        },
    }


def _label_wise_oracle(
    expected: frozenset[str],
    qwen: frozenset[str],
    llm: frozenset[str],
) -> frozenset[str]:
    """Choose the correct source on disagreements; shared mistakes remain mistakes."""
    selected = set()
    for label in METHOD_LABELS:
        qwen_value = label in qwen
        llm_value = label in llm
        value = qwen_value if qwen_value == llm_value else label in expected
        if value:
            selected.add(label)
    return frozenset(selected)


def compare_predictions(
    expected: list[frozenset[str]],
    qwen: list[frozenset[str]],
    llm: list[frozenset[str]],
) -> dict[str, Any]:
    """Compare individual systems, boolean cascades, and their label-wise ceiling."""
    if not expected or len(expected) != len(qwen) or len(expected) != len(llm):
        raise ValueError("expected, qwen, and llm predictions must have equal non-zero length")
    strategies = {
        "qwen": qwen,
        "llm": llm,
        "or": [left | right for left, right in zip(qwen, llm)],
        "and": [left & right for left, right in zip(qwen, llm)],
        "label_wise_oracle": [
            _label_wise_oracle(truth, left, right)
            for truth, left, right in zip(expected, qwen, llm)
        ],
    }
    per_label_disagreements = {
        label: sum((label in left) != (label in right) for left, right in zip(qwen, llm))
        for label in METHOD_LABELS
    }
    examples_with_disagreement = sum(left != right for left, right in zip(qwen, llm))
    total_label_disagreements = sum(per_label_disagreements.values())
    reports = {
        name: prediction_report(expected, predictions)
        for name, predictions in strategies.items()
    }
    return {
        "strategies": reports,
        "disagreement": {
            "examples_with_disagreement": examples_with_disagreement,
            "example_rate": examples_with_disagreement / len(expected),
            "label_decisions": total_label_disagreements,
            "label_rate": total_label_disagreements / (len(expected) * len(METHOD_LABELS)),
            "per_label": {
                label: {
                    "count": count,
                    "rate": count / len(expected),
                }
                for label, count in per_label_disagreements.items()
            },
        },
        "ceiling_95_95_viable": reports["label_wise_oracle"]["gate"]["all_labels_pass"],
    }


def benchmark(
    examples: list[dict[str, Any]],
    *,
    qwen_predictor: Prediction,
) -> dict[str, Any]:
    """Run Qwen predictions and compute all offline cascade rules."""
    expected: list[frozenset[str]] = []
    qwen: list[frozenset[str]] = []
    llm: list[frozenset[str]] = []
    for example in examples:
        predicted = _policy_set(
            list(qwen_predictor(example["text"])),
            source=f"Qwen prediction for {example['candidate_id']}",
        )
        expected.append(example["expected"])
        qwen.append(predicted)
        llm.append(example["llm"])
    return compare_predictions(expected, qwen, llm)


def _runtime_predictor(checkpoint: Path, device: str) -> Prediction:
    def predict(text: str) -> tuple[str, ...]:
        result = classify_task_methods(text, checkpoint=str(checkpoint), device=device)
        if not result.scores:
            raise RuntimeError(
                f"Qwen checkpoint produced no scores: {result.abstention_reason or 'unknown error'}"
            )
        return tuple(item.policy_id for item in result.selected)

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples = load_examples(args.candidates, args.reviews, args.proposals)
    report = {
        "schema_version": 1,
        "purpose": "development cascade ceiling; not sealed-holdout evidence",
        "methods": list(METHOD_LABELS),
        "inputs": {
            "candidates": {"path": str(args.candidates), "sha256": _sha256(args.candidates)},
            "reviews": {"path": str(args.reviews), "sha256": _sha256(args.reviews)},
            "proposals": {"path": str(args.proposals), "sha256": _sha256(args.proposals)},
            "checkpoint": str(args.checkpoint),
            "device": args.device,
        },
        "joined_examples": len(examples),
        "oracle_interpretation": (
            "The label-wise oracle is correct on Qwen/LLM disagreements and preserves "
            "their shared errors; it is an upper bound, not a deployable selector."
        ),
        **benchmark(
            examples,
            qwen_predictor=_runtime_predictor(args.checkpoint, args.device),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "examples": len(examples),
        "ceiling_95_95_viable": report["ceiling_95_95_viable"],
        "exact_match": {
            name: metrics["exact_match"] for name, metrics in report["strategies"].items()
        },
        "disagreement": report["disagreement"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
