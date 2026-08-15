"""Fine-tune a label-conditioned cross-encoder for task-policy routing.

Each request is paired with every policy definition and scored independently.
Training can use tempered teacher labels, while threshold calibration is loaded
from a separate human-only corpus. Evaluation remains an explicit final phase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any

import numpy as np

from bench.task_policy_encoder_finetune import _example_weights
from bench.task_policy_external_review import ExternalReview
from bench.task_policy_gliclass_finetune import (
    _binary_metrics,
    _load_partition,
    _targets,
    _validate_partitions,
)
from bench.task_policy_multilabel_head import METHOD_LABELS


RUN_VERSION = "task-policy-pairwise-cross-encoder-v1"
ACCURACY_TARGET = 0.95
RECALL_TARGET = 0.95

LABEL_DESCRIPTIONS: dict[str, str] = {
    "bugfix.root_cause": (
        "Bug fix: restore existing software behavior or a previously intended contract that is "
        "currently violated. Exclude new capabilities, pure cleanup, and correct but slow code."
    ),
    "feature.contract_first": (
        "New feature: add observable behavior or a capability the software does not have yet. "
        "Exclude restoring an existing contract and behavior-preserving restructuring."
    ),
    "refactor.preserve_behavior": (
        "Code refactoring: reorganize internal structure while preserving observable behavior. "
        "Exclude bug fixes, new capabilities, and performance changes that alter the contract."
    ),
    "research.evidence_first": (
        "Technical research: gather evidence, compare alternatives, or design an experiment before "
        "deciding. Exclude implementing a decided change and reviewing one existing artifact."
    ),
    "review.read_only": (
        "Read-only review: inspect existing code, a patch, or another artifact and report defects, "
        "risks, or findings without changing it. Exclude requests that authorize fixing findings."
    ),
    "performance.measure_first": (
        "Performance work: measure or improve latency, throughput, memory, CPU, I/O, or cost while "
        "preserving intended semantics. Exclude incorrect output and measurement-free cleanup."
    ),
}

HARD_BOUNDARIES = {
    frozenset(("bugfix.root_cause", "feature.contract_first")),
    frozenset(("bugfix.root_cause", "performance.measure_first")),
    frozenset(("bugfix.root_cause", "review.read_only")),
    frozenset(("feature.contract_first", "refactor.preserve_behavior")),
    frozenset(("performance.measure_first", "refactor.preserve_behavior")),
    frozenset(("research.evidence_first", "review.read_only")),
}


@dataclass(frozen=True)
class PairwiseParameters:
    """Frozen training and evaluation parameters."""

    model_name: str
    head: str = "nli"
    max_length: int = 512
    batch_size: int = 16
    evaluation_batch_size: int = 48
    gradient_accumulation_steps: int = 2
    epochs: int = 3
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    model_label_weight: float = 0.25
    positive_weight: float = 3.0
    hard_negative_weight: float = 2.0
    accuracy_target: float = ACCURACY_TARGET
    recall_target: float = RECALL_TARGET
    seed: int = 41
    bf16: bool = True


def _is_hard_negative(policy: str, positives: frozenset[str]) -> bool:
    """Return whether a negative policy sits on a known confusing boundary."""
    return any(frozenset((policy, positive)) in HARD_BOUNDARIES for positive in positives)


def _pair_weight(
    review: ExternalReview,
    policy: str,
    *,
    example_weight: float,
    positive_weight: float,
    hard_negative_weight: float,
) -> float:
    """Combine provenance and semantic-boundary weighting for one pair."""
    positives = frozenset(review.policies)
    if policy in positives:
        return example_weight * positive_weight
    if _is_hard_negative(policy, positives):
        return example_weight * hard_negative_weight
    return example_weight


class _PairDataset:
    """Lazy Cartesian product of requests and policy descriptions."""

    def __init__(
        self,
        reviews: list[ExternalReview],
        tokenizer: object,
        parameters: PairwiseParameters,
        *,
        training: bool,
    ) -> None:
        self.reviews = reviews
        self.tokenizer = tokenizer
        self.parameters = parameters
        self.training = training
        self.example_weights = _example_weights(reviews, parameters.model_label_weight)

    def __len__(self) -> int:
        return len(self.reviews) * len(METHOD_LABELS)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_index, label_index = divmod(index, len(METHOD_LABELS))
        review = self.reviews[row_index]
        policy = METHOD_LABELS[label_index]
        encoded = self.tokenizer(
            review.text,
            LABEL_DESCRIPTIONS[policy],
            max_length=self.parameters.max_length,
            truncation="only_first",
        )
        positive = float(policy in review.policies)
        encoded["labels"] = positive
        if self.training:
            encoded["pair_weight"] = _pair_weight(
                review,
                policy,
                example_weight=float(self.example_weights[row_index]),
                positive_weight=self.parameters.positive_weight,
                hard_negative_weight=self.parameters.hard_negative_weight,
            )
        return encoded


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    return sorted({
        0.0,
        1.0,
        *(float(value) for value in np.arange(0.025, 0.976, 0.025)),
        *(float(value) for value in scores),
        *(float(np.nextafter(value, np.inf)) for value in scores),
    })


def _select_threshold(
    scores: np.ndarray,
    expected: np.ndarray,
    *,
    accuracy_target: float,
    recall_target: float,
) -> float:
    """Select the best operating point, prioritizing the actual 95/95 gate."""
    best: tuple[tuple[float, ...], float] | None = None
    for threshold in _threshold_candidates(scores):
        metrics = _binary_metrics(scores, expected, threshold)
        accuracy = float(metrics["accuracy"])
        recall = float(metrics["recall"])
        precision = float(metrics["precision"])
        f1 = float(metrics["f1"])
        key = (
            float(accuracy >= accuracy_target and recall >= recall_target),
            min(accuracy / accuracy_target, recall / recall_target),
            min(accuracy, recall),
            f1,
            precision,
            accuracy,
            recall,
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        raise RuntimeError("threshold selection produced no candidates")
    return best[1]


def _calibrate_thresholds(
    scores: np.ndarray,
    expected: np.ndarray,
    parameters: PairwiseParameters,
) -> tuple[float, ...]:
    return tuple(
        _select_threshold(
            scores[:, index],
            expected[:, index],
            accuracy_target=parameters.accuracy_target,
            recall_target=parameters.recall_target,
        )
        for index in range(len(METHOD_LABELS))
    )


def _report(
    scores: np.ndarray,
    expected: np.ndarray,
    thresholds: tuple[float, ...],
    parameters: PairwiseParameters,
) -> dict[str, Any]:
    predicted = scores >= np.asarray(thresholds)[None, :]
    per_label = {
        label: _binary_metrics(scores[:, index], expected[:, index], thresholds[index])
        for index, label in enumerate(METHOD_LABELS)
    }
    return {
        "examples": len(expected),
        "exact_match": float(np.mean(np.all(predicted == expected, axis=1))),
        "micro_accuracy": float(np.mean(predicted == expected)),
        "predicted_cardinality": float(np.mean(predicted.sum(axis=1))),
        "expected_cardinality": float(np.mean(expected.sum(axis=1))),
        "per_label": per_label,
        "gate": {
            "accuracy_target": parameters.accuracy_target,
            "recall_target": parameters.recall_target,
            "all_labels_pass": all(
                float(metrics["accuracy"]) >= parameters.accuracy_target
                and float(metrics["recall"]) >= parameters.recall_target
                for metrics in per_label.values()
            ),
        },
    }


def _checkpoint_key(report: dict[str, Any], parameters: PairwiseParameters) -> tuple[float, ...]:
    metrics = [report["per_label"][label] for label in METHOD_LABELS]
    margins = [
        min(
            float(item["accuracy"]) / parameters.accuracy_target,
            float(item["recall"]) / parameters.recall_target,
        )
        for item in metrics
    ]
    passes = sum(margin >= 1.0 for margin in margins)
    return (
        float(passes == len(METHOD_LABELS)),
        passes / len(METHOD_LABELS),
        min(margins),
        float(np.mean(margins)),
        float(report["exact_match"]),
    )


def _loader(dataset: object, tokenizer: object, *, batch_size: int, shuffle: bool) -> object:
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def _move(batch: dict[str, object], device: object) -> dict[str, object]:
    import torch

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _scores(
    model: object,
    tokenizer: object,
    reviews: list[ExternalReview],
    parameters: PairwiseParameters,
    device: object,
) -> np.ndarray:
    import torch

    dataset = _PairDataset(reviews, tokenizer, parameters, training=False)
    loader = _loader(
        dataset,
        tokenizer,
        batch_size=parameters.evaluation_batch_size,
        shuffle=False,
    )
    chunks = []
    model.eval()  # type: ignore[attr-defined]
    with torch.inference_mode():
        for raw_batch in loader:  # type: ignore[union-attr]
            batch = _move(raw_batch, device)
            batch.pop("labels")
            logits = model(**batch).logits  # type: ignore[operator]
            if parameters.head == "nli":
                positive = logits[:, (0, 2)].float().softmax(dim=-1)[:, 0]
            else:
                positive = logits.squeeze(-1).sigmoid().float()
            chunks.append(positive.cpu().numpy())
    return np.concatenate(chunks).reshape(len(reviews), len(METHOD_LABELS))


def _save(
    model: object,
    tokenizer: object,
    output_dir: Path,
    metadata: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)  # type: ignore[attr-defined]
    tokenizer.save_pretrained(output_dir)  # type: ignore[attr-defined]
    (output_dir / "task_policy_pairwise_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _train(
    model: object,
    tokenizer: object,
    training: list[ExternalReview],
    calibration: list[ExternalReview],
    parameters: PairwiseParameters,
    output_dir: Path,
    device: object,
) -> list[dict[str, Any]]:
    import torch
    from transformers import get_cosine_schedule_with_warmup

    dataset = _PairDataset(training, tokenizer, parameters, training=True)
    loader = _loader(dataset, tokenizer, batch_size=parameters.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),  # type: ignore[attr-defined]
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / parameters.gradient_accumulation_steps)
    total_updates = parameters.epochs * updates_per_epoch
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * parameters.warmup_ratio),
        num_training_steps=total_updates,
    )
    use_bf16 = parameters.bf16 and getattr(device, "type", str(device)) == "cuda"
    best_key: tuple[float, ...] | None = None
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, parameters.epochs + 1):
        model.train()  # type: ignore[attr-defined]
        losses = []
        for batch_index, raw_batch in enumerate(loader, 1):
            batch = _move(raw_batch, device)
            labels = batch.pop("labels").float()
            pair_weights = batch.pop("pair_weight").float()
            with torch.autocast(
                device_type=getattr(device, "type", str(device)),
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                logits = model(**batch).logits  # type: ignore[operator]
                if parameters.head == "nli":
                    # Published mDeBERTa NLI mapping: entailment=0, contradiction=2.
                    binary_logits = logits[:, (0, 2)].float()
                    binary_targets = torch.where(
                        labels.bool(),
                        torch.zeros_like(labels, dtype=torch.long),
                        torch.ones_like(labels, dtype=torch.long),
                    )
                    per_pair = torch.nn.functional.cross_entropy(
                        binary_logits,
                        binary_targets,
                        reduction="none",
                    )
                else:
                    per_pair = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits.squeeze(-1).float(),
                        labels,
                        reduction="none",
                    )
                loss = (per_pair * pair_weights).sum() / pair_weights.sum().clamp(min=1e-8)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
            losses.append(float(loss.detach()))
            (loss / parameters.gradient_accumulation_steps).backward()
            if (
                batch_index % parameters.gradient_accumulation_steps == 0
                or batch_index == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # type: ignore[attr-defined]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batch_index % 250 == 0:
                print(json.dumps({
                    "epoch": epoch,
                    "batch": batch_index,
                    "batches": len(loader),
                    "mean_loss": float(np.mean(losses[-250:])),
                }, sort_keys=True), flush=True)
        calibration_scores = _scores(model, tokenizer, calibration, parameters, device)
        thresholds = _calibrate_thresholds(calibration_scores, _targets(calibration), parameters)
        report = _report(calibration_scores, _targets(calibration), thresholds, parameters)
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "thresholds": dict(zip(METHOD_LABELS, thresholds, strict=True)),
            "calibration": report,
        }
        history.append(record)
        print(json.dumps({
            "epoch": epoch,
            "loss": record["training_loss"],
            "calibration_exact_match": report["exact_match"],
            "gate": report["gate"],
        }, sort_keys=True), flush=True)
        key = _checkpoint_key(report, parameters)
        if best_key is None or key > best_key:
            best_key = key
            _save(
                model,
                tokenizer,
                output_dir,
                {
                    "run_version": RUN_VERSION,
                    "parameters": asdict(parameters),
                    "label_descriptions": LABEL_DESCRIPTIONS,
                    "selected_epoch": epoch,
                    "thresholds": record["thresholds"],
                    "calibration": report,
                },
            )
    return history


def _latency(
    model: object,
    tokenizer: object,
    reviews: list[ExternalReview],
    parameters: PairwiseParameters,
    device: object,
    samples: int,
) -> dict[str, float | int]:
    import torch

    timings = []
    for index in range(samples + 10):
        review = reviews[index % len(reviews)]
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        _scores(model, tokenizer, [review], parameters, device)
        if getattr(device, "type", str(device)) == "cuda":
            torch.cuda.synchronize(device)
        if index >= 10:
            timings.append((time.perf_counter() - started) * 1000)
    return {
        "samples": samples,
        "p50_ms": statistics.median(timings),
        "p95_ms": float(np.percentile(timings, 95)),
        "max_ms": max(timings),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--calibration-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    parser.add_argument("--head", choices=("nli", "binary"), default="nli")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--evaluation-batch-size", type=int, default=48)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--model-label-weight", type=float, default=0.25)
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(41)
    requested_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested_device == "auto":
        requested_device = "cpu"
    device = torch.device(requested_device)
    parameters = PairwiseParameters(
        model_name=args.model,
        head=args.head,
        max_length=args.max_length,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        model_label_weight=args.model_label_weight,
        bf16=not args.no_bf16,
    )
    training = _load_partition(args.training_root, "training")
    calibration = _load_partition(args.calibration_root, "calibration")
    _validate_partitions(training, calibration)
    print(json.dumps({
        "device": str(device),
        "training_examples": len(training),
        "training_pairs": len(training) * len(METHOD_LABELS),
        "calibration_examples": len(calibration),
        "calibration_annotation_kinds": sorted({row.annotation_kind for row in calibration}),
        "evaluation_opened": False,
    }, sort_keys=True), flush=True)
    if {row.annotation_kind for row in calibration} != {"human"}:
        raise ValueError("calibration must contain human annotations only")

    tokenizer = AutoTokenizer.from_pretrained(parameters.model_name)
    if parameters.head == "nli":
        model = AutoModelForSequenceClassification.from_pretrained(parameters.model_name)
        expected_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
        if model.config.id2label != expected_mapping:
            raise ValueError(
                "NLI head requires id2label entailment=0, neutral=1, contradiction=2; "
                f"got {model.config.id2label}"
            )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            parameters.model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
    model.to(device)
    history = _train(
        model,
        tokenizer,
        training,
        calibration,
        parameters,
        args.output_dir,
        device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model = AutoModelForSequenceClassification.from_pretrained(args.output_dir)
    model.to(device)
    frozen = json.loads(
        (args.output_dir / "task_policy_pairwise_config.json").read_text(encoding="utf-8")
    )
    thresholds = tuple(float(frozen["thresholds"][label]) for label in METHOD_LABELS)
    evaluation = _load_partition(args.evaluation_root, "evaluation")
    evaluation_scores = _scores(model, tokenizer, evaluation, parameters, device)
    evaluation_report = _report(evaluation_scores, _targets(evaluation), thresholds, parameters)
    latency = _latency(
        model,
        tokenizer,
        evaluation,
        parameters,
        device,
        args.latency_samples,
    )
    report = {
        "run_version": RUN_VERSION,
        "parameters": asdict(parameters),
        "counts": {
            "training": len(training),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        "selected_epoch": frozen["selected_epoch"],
        "thresholds": frozen["thresholds"],
        "label_descriptions": LABEL_DESCRIPTIONS,
        "history": history,
        "evaluation": evaluation_report,
        "latency": latency,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "selected_epoch": report["selected_epoch"],
        "evaluation_exact_match": evaluation_report["exact_match"],
        "evaluation_gate": evaluation_report["gate"],
        "latency": latency,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
