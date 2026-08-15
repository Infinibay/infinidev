"""Evaluate the selected main model as a task-method classifier.

The harness reads a reviewed natural split, selects a deterministic length-
stratified sample, and calls the same provider/model configured for Infinidev.
It records labels and timing, but never credentials or full request text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from infinidev.engine.task_policies.router import _default_llm_classifier


METHOD_LABELS = (
    "bugfix",
    "feature",
    "performance",
    "refactor",
    "research",
    "review",
)


BINARY_JUDGES_SYSTEM_PROMPT = """Classify the intended software-work method.
Return JSON only with exactly this shape:
{"decisions":{"bugfix":{"selected":false,"confidence":0.0},"feature":{"selected":false,"confidence":0.0},"performance":{"selected":false,"confidence":0.0},"refactor":{"selected":false,"confidence":0.0},"research":{"selected":false,"confidence":0.0},"review":{"selected":false,"confidence":0.0}}}

Judge every category independently. A true decision must never suppress another
true decision. Select only an independently requested goal, not an incidental
implementation step.

- bugfix: restore existing behavior or an intended contract currently violated.
  Exclude new capabilities, pure cleanup, and correct but slow code.
- feature: add observable behavior or a capability not present yet. Exclude
  restoring an existing contract and behavior-preserving restructuring.
- performance: measure or improve latency, throughput, memory, CPU, I/O, or cost
  while preserving intended semantics. Exclude incorrect output and cleanup.
- refactor: reorganize internal code while preserving observable behavior.
  Exclude bug fixes, new capabilities, and incidental optimization rewrites.
- research: gather evidence, compare alternatives, or design an experiment before
  deciding. Exclude implementation and assessment of one existing artifact.
- review: inspect existing code, a patch, or another artifact and report defects,
  risks, or findings without changing it. Exclude authorization to fix findings.

Use all false for how-to questions, raw code without an instruction, conversation,
explanation, translation, or unrelated work. Never infer permissions."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_reviewed_partition(data_root: Path, partition: str) -> list[dict[str, Any]]:
    """Join candidate text to reviewed labels by immutable candidate id."""
    candidates = {
        row["candidate_id"]: row
        for row in _read_jsonl(data_root / f"{partition}_candidates.jsonl")
    }
    joined = []
    for review in _read_jsonl(data_root / f"{partition}_reviews.jsonl"):
        if not review.get("include", True):
            continue
        candidate_id = review["candidate_id"]
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise ValueError(f"missing candidate for review {candidate_id}")
        joined.append({
            "candidate_id": candidate_id,
            "text": candidate["issue_text"],
            "expected": tuple(review.get("policies") or ()),
        })
    return joined


def _length_stratified(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Choose stable text-length quantiles without repeating rows."""
    if count <= 0 or not rows:
        return []
    ordered = sorted(rows, key=lambda row: (len(row["text"]), row["candidate_id"]))
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[len(ordered) // 2]]
    indices = {
        round(index * (len(ordered) - 1) / (count - 1))
        for index in range(count)
    }
    return [ordered[index] for index in sorted(indices)]


def select_stratified_sample(
    rows: list[dict[str, Any]],
    *,
    per_label: int,
    zero_label: int,
    compound: int,
) -> list[dict[str, Any]]:
    """Select single-label, zero-label, and compound requests deterministically."""
    selected: dict[str, dict[str, Any]] = {}
    for label in METHOD_LABELS:
        stratum = [row for row in rows if row["expected"] == (label,)]
        for row in _length_stratified(stratum, per_label):
            selected[row["candidate_id"]] = row
    for row in _length_stratified(
        [row for row in rows if not row["expected"]], zero_label,
    ):
        selected[row["candidate_id"]] = row
    for row in _length_stratified(
        [row for row in rows if len(row["expected"]) > 1], compound,
    ):
        selected[row["candidate_id"]] = row
    return list(selected.values())


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def parse_binary_judges(content: str) -> dict[str, Any]:
    """Parse one complete set of six independent binary decisions."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        raise ValueError("binary-judge response contains no JSON object")
    payload = json.loads(content[start:end + 1])
    if set(payload) != {"decisions"} or not isinstance(payload["decisions"], dict):
        raise ValueError("binary-judge response must contain only decisions")
    decisions = payload["decisions"]
    if set(decisions) != set(METHOD_LABELS):
        raise ValueError("binary-judge response must decide every canonical label exactly once")
    operations = []
    confidences = {}
    for label in METHOD_LABELS:
        decision = decisions[label]
        if not isinstance(decision, dict) or set(decision) != {"selected", "confidence"}:
            raise ValueError(f"invalid binary decision for {label}")
        selected = decision["selected"]
        confidence = decision["confidence"]
        if not isinstance(selected, bool):
            raise ValueError(f"selected must be boolean for {label}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"confidence must be numeric for {label}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"confidence out of range for {label}")
        if selected:
            operations.append(label)
        confidences[label] = float(confidence)
    return {"operations": operations, "confidences": confidences}


def _binary_judges_classifier(text: str, *, max_tokens: int) -> dict[str, Any] | None:
    """Ask the selected main model for six independent binary judgments."""
    try:
        from infinidev.config.llm import get_litellm_params
        from infinidev.engine.llm_client import call_llm

        params = get_litellm_params()
        params["max_tokens"] = max_tokens
        response = call_llm(
            params,
            messages=[
                {"role": "system", "content": BINARY_JUDGES_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            use_json_mode=False,
            thinking_enabled=False,
        )
        content = response.choices[0].message.content
        if not isinstance(content, str):
            return None
        return parse_binary_judges(content)
    except Exception:
        return None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact-match, per-label metrics, and latency percentiles."""
    valid = [record for record in records if record["predicted"] is not None]
    exact = sum(
        set(record["predicted"]) == set(record["expected"])
        for record in valid
    )
    per_label = {}
    for label in METHOD_LABELS:
        true_positive = sum(
            label in record["expected"] and label in record["predicted"]
            for record in valid
        )
        false_positive = sum(
            label not in record["expected"] and label in record["predicted"]
            for record in valid
        )
        false_negative = sum(
            label in record["expected"] and label not in record["predicted"]
            for record in valid
        )
        true_negative = sum(
            label not in record["expected"] and label not in record["predicted"]
            for record in valid
        )
        predicted_positive = true_positive + false_positive
        expected_positive = true_positive + false_negative
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / expected_positive if expected_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "accuracy": (true_positive + true_negative) / len(valid) if valid else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_positive + false_negative,
        }
    latencies = [float(record["latency_ms"]) for record in records]
    gate = all(
        metrics["accuracy"] >= 0.95 and metrics["recall"] >= 0.95
        for metrics in per_label.values()
    )
    return {
        "calls": len(records),
        "valid": len(valid),
        "failures": len(records) - len(valid),
        "exact_match": exact / len(valid) if valid else 0.0,
        "per_label": per_label,
        "gate": {
            "accuracy_target": 0.95,
            "recall_target": 0.95,
            "all_labels_pass": gate,
        },
        "latency_ms": {
            "p50": statistics.median(latencies) if latencies else 0.0,
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0.0),
        },
    }


def evaluate(
    rows: list[dict[str, Any]],
    *,
    max_tokens: int,
    strategy: str = "list",
) -> dict[str, Any]:
    """Run stateless classification calls and return auditable compact records."""
    records = []
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        if strategy == "binary":
            binary_result = _binary_judges_classifier(row["text"], max_tokens=max_tokens)
            predicted = binary_result["operations"] if binary_result else None
            confidence = binary_result["confidences"] if binary_result else None
        elif strategy == "list":
            result = _default_llm_classifier(row["text"], max_tokens=max_tokens)
            predicted = result.operations if result else None
            confidence = result.confidence if result else None
        else:
            raise ValueError(f"unknown classification strategy: {strategy}")
        latency_ms = (time.perf_counter() - started) * 1000
        record = {
            "candidate_id": row["candidate_id"],
            "text_chars": len(row["text"]),
            "expected": list(row["expected"]),
            "predicted": predicted,
            "confidence": confidence,
            "latency_ms": latency_ms,
        }
        records.append(record)
        print(
            f"[{index}/{len(rows)}] {record['candidate_id']} "
            f"expected={record['expected']} predicted={record['predicted']} "
            f"latency_ms={latency_ms:.1f}",
            flush=True,
        )
    return {"summary": summarize(records), "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "tmp" / "task-policy-natural-split-v1",
    )
    parser.add_argument("--partition", default="evaluation")
    parser.add_argument("--per-label", type=int, default=3)
    parser.add_argument("--zero-label", type=int, default=4)
    parser.add_argument("--compound", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--strategy", choices=("list", "binary"), default="list")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.per_label, args.zero_label, args.compound) < 0:
        parser.error("sample counts must not be negative")
    if args.max_tokens < 1:
        parser.error("max tokens must be positive")

    rows = load_reviewed_partition(args.data_root, args.partition)
    sample = select_stratified_sample(
        rows,
        per_label=args.per_label,
        zero_label=args.zero_label,
        compound=args.compound,
    )
    report = evaluate(sample, max_tokens=args.max_tokens, strategy=args.strategy)
    from infinidev.config.settings import settings

    report["provider"] = settings.LLM_PROVIDER
    report["model"] = settings.LLM_MODEL
    report["max_tokens"] = args.max_tokens
    report["strategy"] = args.strategy
    report["data_root"] = str(args.data_root)
    report["partition"] = args.partition
    report["sample"] = {
        "per_label": args.per_label,
        "zero_label": args.zero_label,
        "compound": args.compound,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
