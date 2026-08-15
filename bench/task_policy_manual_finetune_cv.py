"""Cross-validate end-to-end E5-small adaptation on the manual policy corpus.

This is a development diagnostic. Every fold starts from the published encoder,
selects its checkpoint and thresholds on validation, and predicts family-separated
test rows once. It never reads the future sealed validation or holdout datasets.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import re
import sys
import time
from typing import Any
import zlib

import numpy as np

from bench.contextual_embedding_benchmark import (
    DEFAULT_CONTEXTUAL_MODEL,
    _cardinality_class_weights,
    _cardinality_targets,
    _prediction_metrics,
)
from bench.contextual_task_policy_finetune import _set_trainable_layers
from bench.task_policy_external_review import (
    ExternalReview,
    load_external_candidates,
    load_external_reviews,
)
from bench.task_policy_manual_audit import load_examples
from bench.task_policy_manual_cv import (
    FOLD_ASSIGNMENT_VERSION,
    _examples,
    _predict_independent,
    assign_folds,
    select_independent_thresholds,
    select_joint_thresholds,
)
from bench.task_policy_multilabel_head import METHOD_LABELS, _targets


FINETUNE_CV_VERSION = "manual-task-policy-e5-finetune-cv-v4"
LEXICAL_FEATURE_DIMENSIONS = 2048
_LEXICAL_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _hashed_lexical_features(
    texts: list[str],
    *,
    dimensions: int = LEXICAL_FEATURE_DIMENSIONS,
) -> np.ndarray:
    """Encode bounded title and request n-grams with deterministic feature hashing."""
    if dimensions < 32:
        raise ValueError("lexical feature dimensions must be at least 32")
    matrix = np.zeros((len(texts), dimensions), dtype=np.float32)
    for row_index, text in enumerate(texts):
        normalized = " ".join(text.casefold().split())[:768]
        title = text.casefold().splitlines()[0].strip()[:256] if text.strip() else ""
        tokens = _LEXICAL_TOKEN.findall(normalized)[:96]
        title_tokens = _LEXICAL_TOKEN.findall(title)[:32]
        features = [f"word:{token}" for token in tokens]
        features.extend(
            f"bigram:{left}:{right}"
            for left, right in zip(tokens, tokens[1:])
        )
        features.extend(f"title-word:{token}" for token in title_tokens)
        features.extend(
            f"title-bigram:{left}:{right}"
            for left, right in zip(title_tokens, title_tokens[1:])
        )
        compact_title = " ".join(title_tokens)
        for width in (3, 4, 5):
            features.extend(
                f"title-char-{width}:{compact_title[index:index + width]}"
                for index in range(max(0, len(compact_title) - width + 1))
            )
        for feature in features:
            digest = zlib.crc32(feature.encode("utf-8"))
            column = digest % dimensions
            sign = 1.0 if digest & 0x80000000 else -1.0
            matrix[row_index, column] += sign
        norm = float(np.linalg.norm(matrix[row_index]))
        if norm:
            matrix[row_index] /= norm
    return matrix

@dataclass(frozen=True)
class ManualFinetuneParameters:
    """Fine-tuning parameters fixed before cross-validation."""

    max_length: int = 96
    batch_size: int = 16
    epochs: int = 8
    unfrozen_layers: int = 4
    hidden_size: int = 128
    architecture: str = "shared_mean"
    pooling: str = "mean"
    query_instruction: str | None = None
    load_in_4bit: bool = False
    threshold_calibration: str = "independent"
    encoder_learning_rate: float = 2e-5
    head_learning_rate: float = 5e-4
    lexical_learning_rate: float = 1e-2
    weight_decay: float = 1e-3
    task_loss_weight: float = 0.7
    exclusive_loss_weight: float = 0.7
    cardinality_loss_weight: float = 0.7
    cardinality_balance_power: float = 0.5
    positive_weight_power: float = 0.5
    minimum_method_precision: float = 0.85
    minimum_method_recall: float | None = None
    minimum_method_accuracy: float = 0.95
    lora_rank: int = 0
    lora_alpha: int = 16
    patience: int = 3
    seed: int = 41


def _positive_weights(targets: np.ndarray, power: float) -> np.ndarray:
    """Return tempered inverse-frequency positive weights for binary labels."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("positive_weight_power must be between 0 and 1")
    positives = targets.sum(axis=0)
    negatives = len(targets) - positives
    ratios = negatives / np.maximum(positives, 1)
    return np.power(ratios, power).astype(np.float32)


def _individual_accuracy_checkpoint_key(
    report: dict[str, Any],
    *,
    accuracy_target: float,
) -> tuple[float, ...]:
    """Rank checkpoints by the weakest per-policy binary accuracy."""
    per_label = report["per_label"]
    accuracies = [float(per_label[label]["accuracy"]) for label in METHOD_LABELS]
    precisions = [float(per_label[label]["precision"]) for label in METHOD_LABELS]
    recalls = [float(per_label[label]["recall"]) for label in METHOD_LABELS]
    labels_with_recall = sum(value > 0.0 for value in recalls)
    labels_at_target = sum(
        accuracy >= accuracy_target and recall > 0.0
        for accuracy, recall in zip(accuracies, recalls, strict=True)
    )
    return (
        float(labels_at_target == len(METHOD_LABELS)),
        labels_at_target / len(METHOD_LABELS),
        labels_with_recall / len(METHOD_LABELS),
        min(accuracies),
        float(np.mean(accuracies)),
        min(recalls),
        float(np.mean(recalls)),
        float(np.mean(precisions)),
        float(report["exact_match"]),
    )


def _select_accuracy_thresholds(
    method_scores: np.ndarray,
    task_scores: np.ndarray,
    validation: list[object],
    *,
    accuracy_target: float,
) -> tuple[tuple[float, ...], float, dict[str, Any], float]:
    """Select method and abstention thresholds using validation accuracy only."""
    best = None
    for precision_floor in (0.80, 0.85, 0.90, 0.95, 1.0):
        thresholds, _, _ = select_independent_thresholds(
            method_scores,
            task_scores,
            validation,  # type: ignore[arg-type]
            minimum_method_precision=precision_floor,
        )
        # Independent method thresholds already represent abstention. A second
        # global task gate can only suppress otherwise valid labels and makes
        # one abundant task class dominate all six binary decisions.
        task_threshold = 0.0
        predictions = _predict_independent(
            method_scores,
            task_scores,
            thresholds,
            task_threshold,
        )
        report = _prediction_metrics(validation, predictions)  # type: ignore[arg-type]
        key = _individual_accuracy_checkpoint_key(
            report,
            accuracy_target=accuracy_target,
        )
        if best is None or key > best[0]:
            best = (
                key,
                thresholds,
                task_threshold,
                report,
                precision_floor,
            )
    if best is None:
        raise RuntimeError("accuracy calibration produced no threshold candidate")
    _, thresholds, task_threshold, report, precision_floor = best
    return thresholds, task_threshold, report, precision_floor


def _binary_metrics(
    scores: np.ndarray,
    expected: np.ndarray,
    threshold: float,
) -> tuple[float, float, float, float]:
    """Return accuracy, precision, recall, and F1 for one binary label."""
    predicted = scores >= threshold
    true_positive = int(np.sum(predicted & expected))
    false_positive = int(np.sum(predicted & ~expected))
    false_negative = int(np.sum(~predicted & expected))
    accuracy = float(np.mean(predicted == expected))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return accuracy, precision, recall, f1


def _select_natural_accuracy_threshold(
    scores: np.ndarray,
    expected: np.ndarray,
    *,
    accuracy_target: float,
    minimum_precision: float = 0.85,
    minimum_recall: float = 0.50,
) -> float:
    """Select one threshold from a supported natural calibration label."""
    candidates = sorted({
        0.0,
        1.0,
        *(float(value) for value in np.arange(0.05, 0.951, 0.025)),
        *(float(value) for value in scores),
        *(float(np.nextafter(value, np.inf)) for value in scores),
    })
    best: tuple[tuple[float, ...], float] | None = None
    for threshold in candidates:
        accuracy, precision, recall, f1 = _binary_metrics(scores, expected, threshold)
        satisfies_quality = (
            accuracy >= accuracy_target
            and precision >= minimum_precision
            and recall >= minimum_recall
        )
        satisfies_recall = recall >= minimum_recall
        key = (
            float(satisfies_quality),
            float(satisfies_recall),
            float(recall > 0.0),
            accuracy,
            f1,
            precision,
            recall,
            threshold,
        )
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        raise RuntimeError("natural accuracy calibration produced no threshold")
    return best[1]


def _select_domain_accuracy_thresholds(
    synthetic_method_scores: np.ndarray,
    synthetic_task_scores: np.ndarray,
    synthetic_validation: list[object],
    natural_method_scores: np.ndarray,
    natural_task_scores: np.ndarray,
    natural_validation: list[object],
    *,
    accuracy_target: float,
    minimum_natural_positives: int = 5,
) -> tuple[tuple[float, ...], float, dict[str, Any], float]:
    """Prefer natural thresholds where calibration has real class support.

    Labels absent or too sparse in the natural shard retain their synthetic
    threshold. This prevents a much larger synthetic validation fold from
    silently dominating the domain calibration while avoiding zero-positive
    thresholds for research or review.
    """
    thresholds, _, _, precision_floor = _select_accuracy_thresholds(
        synthetic_method_scores,
        synthetic_task_scores,
        synthetic_validation,
        accuracy_target=accuracy_target,
    )
    selected = list(thresholds)
    natural_targets = _targets(natural_validation).astype(bool)  # type: ignore[arg-type]
    for index in range(len(METHOD_LABELS)):
        expected = natural_targets[:, index]
        positives = int(expected.sum())
        negatives = len(expected) - positives
        if positives < minimum_natural_positives or negatives < minimum_natural_positives:
            continue
        selected[index] = _select_natural_accuracy_threshold(
            natural_method_scores[:, index],
            expected,
            accuracy_target=accuracy_target,
        )
    combined_scores = np.concatenate((synthetic_method_scores, natural_method_scores))
    combined_tasks = np.concatenate((synthetic_task_scores, natural_task_scores))
    combined_examples = synthetic_validation + natural_validation
    predictions = _predict_independent(combined_scores, combined_tasks, selected, 0.0)
    report = _prediction_metrics(combined_examples, predictions)  # type: ignore[arg-type]
    return tuple(selected), 0.0, report, precision_floor


def _domain_checkpoint_key(
    synthetic_report: dict[str, Any],
    natural_report: dict[str, Any],
    *,
    accuracy_target: float,
    minimum_natural_positives: int = 5,
) -> tuple[float, ...]:
    """Rank epochs using natural metrics where their positive support is adequate."""
    selected = []
    for label in METHOD_LABELS:
        natural_metrics = natural_report["per_label"][label]
        metrics = (
            natural_metrics
            if int(natural_metrics["support"]) >= minimum_natural_positives
            else synthetic_report["per_label"][label]
        )
        selected.append(metrics)
    accuracies = [float(metrics["accuracy"]) for metrics in selected]
    precisions = [float(metrics["precision"]) for metrics in selected]
    recalls = [float(metrics["recall"]) for metrics in selected]
    labels_with_recall = sum(value > 0.0 for value in recalls)
    labels_at_target = sum(
        accuracy >= accuracy_target and recall > 0.0
        for accuracy, recall in zip(accuracies, recalls, strict=True)
    )
    return (
        float(labels_at_target == len(METHOD_LABELS)),
        labels_at_target / len(METHOD_LABELS),
        labels_with_recall / len(METHOD_LABELS),
        min(accuracies),
        float(np.mean(accuracies)),
        min(recalls),
        float(np.mean(recalls)),
        float(np.mean(precisions)),
        float(natural_report["exact_match"]),
    )


def _encoded_subset(encoded: dict[str, object], indices: np.ndarray) -> dict[str, object]:
    return {name: value[indices] for name, value in encoded.items()}  # type: ignore[index]


def _resolve_device(device: str) -> object:
    """Resolve an explicit PyTorch device, preferring CUDA for auto."""
    import torch

    if device.casefold() == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        requested = device
    try:
        resolved = torch.device(requested)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"invalid PyTorch device {device!r}: {exc}") from exc
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device!r} was requested, but torch.cuda.is_available() is false"
            )
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {resolved.index} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)"
            )
    return resolved


def _batch_to_device(batch: dict[str, object], device: object) -> dict[str, object]:
    """Move one encoded batch to the selected accelerator."""
    return {
        name: value.to(device)  # type: ignore[attr-defined]
        for name, value in batch.items()
    }


def _format_model_input(text: str, instruction: str | None) -> str:
    """Format one request using the selected embedding model's query convention."""
    if instruction is not None:
        normalized = instruction.strip()
        if not normalized:
            raise ValueError("query_instruction must not be blank")
        return f"Instruct: {normalized}\nQuery:{text}"
    return f"query: {text}"


def _pool_hidden_state(
    hidden_state: object,
    attention_mask: object,
    pooling: str,
) -> object:
    """Pool token states with either encoder mean or decoder last-token pooling."""
    import torch

    mask = attention_mask.bool()  # type: ignore[attr-defined]
    if pooling == "mean":
        expanded = mask[..., None]
        pooled = hidden_state.masked_fill(~expanded, 0.0).sum(dim=1)  # type: ignore[attr-defined]
        return pooled / expanded.sum(dim=1).clamp(min=1)
    if pooling == "last":
        positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
        last_positions = positions.expand_as(mask).masked_fill(~mask, -1).max(dim=1).values
        if bool((last_positions < 0).any()):
            raise ValueError("cannot last-pool an input with no unmasked tokens")
        rows = torch.arange(mask.shape[0], device=mask.device)
        return hidden_state[rows, last_positions]  # type: ignore[index]
    raise ValueError("pooling must be 'mean' or 'last'")


def _lora_target_modules(encoder: object) -> tuple[str, str]:
    """Select the attention projection names exposed by an encoder family."""
    available = {
        name.rsplit(".", 1)[-1]
        for name, _module in encoder.named_modules()  # type: ignore[attr-defined]
    }
    for candidates in (("query", "value"), ("q_proj", "v_proj")):
        if all(candidate in available for candidate in candidates):
            return candidates
    raise ValueError(
        "unsupported encoder attention projections for LoRA; expected "
        "query/value or q_proj/v_proj modules"
    )


def _model_scores(
    model: object,
    encoded: dict[str, object],
    *,
    batch_size: int,
    device: object = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run a token-level model in bounded inference batches."""
    import torch

    model.eval()  # type: ignore[attr-defined]
    method_chunks = []
    task_chunks = []
    example_count = len(encoded["input_ids"])  # type: ignore[arg-type]
    with torch.inference_mode():
        for start in range(0, example_count, batch_size):
            batch = _batch_to_device(
                {
                    name: value[start:start + batch_size]  # type: ignore[index]
                    for name, value in encoded.items()
                },
                device,
            )
            method_logits, task_logits = model(**batch)  # type: ignore[operator]
            method_chunks.append(method_logits.sigmoid().detach().cpu().numpy())
            task_chunks.append(task_logits.sigmoid().detach().cpu().numpy())
    return np.concatenate(method_chunks), np.concatenate(task_chunks)


def _model_outputs(
    model: object,
    encoded: dict[str, object],
    *,
    batch_size: int,
    device: object = "cpu",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Run inference and retain optional cardinality probabilities."""
    import torch

    model.eval()  # type: ignore[attr-defined]
    method_chunks = []
    task_chunks = []
    cardinality_chunks = []
    example_count = len(encoded["input_ids"])  # type: ignore[arg-type]
    with torch.inference_mode():
        for start in range(0, example_count, batch_size):
            batch = _batch_to_device(
                {
                    name: value[start:start + batch_size]  # type: ignore[index]
                    for name, value in encoded.items()
                },
                device,
            )
            outputs = model(**batch)  # type: ignore[operator]
            method_logits, task_logits = outputs[:2]
            method_chunks.append(method_logits.sigmoid().detach().cpu().numpy())
            task_chunks.append(task_logits.sigmoid().detach().cpu().numpy())
            if len(outputs) == 3:
                cardinality_chunks.append(
                    outputs[2].softmax(dim=1).detach().cpu().numpy()
                )
    cardinality_scores = (
        np.concatenate(cardinality_chunks) if cardinality_chunks else None
    )
    return (
        np.concatenate(method_chunks),
        np.concatenate(task_chunks),
        cardinality_scores,
    )


def _predict_cardinality(
    method_scores: np.ndarray,
    cardinality_scores: np.ndarray,
) -> list[tuple[str, ...]]:
    """Select the highest-scoring labels using an explicit 0/1/2/3 prediction."""
    predictions = []
    for row, cardinality_row in zip(method_scores, cardinality_scores, strict=True):
        cardinality = int(np.argmax(cardinality_row))
        ranked = np.argsort(row)[::-1][:cardinality]
        predictions.append(tuple(METHOD_LABELS[int(index)] for index in ranked))
    return predictions


def _consensus_predictions(
    predictions_by_model: list[list[tuple[str, ...]]],
) -> list[tuple[str, ...]]:
    """Return strict-majority labels from equally weighted fold models."""
    if not predictions_by_model:
        raise ValueError("consensus needs at least one model")
    example_count = len(predictions_by_model[0])
    if any(len(predictions) != example_count for predictions in predictions_by_model):
        raise ValueError("consensus model predictions have different lengths")
    label_order = {label: index for index, label in enumerate(METHOD_LABELS)}
    consensus = []
    for index in range(example_count):
        votes = Counter(
            label
            for predictions in predictions_by_model
            for label in predictions[index]
        )
        selected = [
            label
            for label, count in votes.items()
            if count * 2 > len(predictions_by_model)
        ]
        selected.sort(key=lambda label: (-votes[label], label_order[label]))
        consensus.append(tuple(selected[:3]))
    return consensus


def _validate_external_partition(
    training_reviews: list[ExternalReview],
    evaluation_reviews: list[ExternalReview],
) -> None:
    """Reject source-repository leakage across natural train and evaluation rows."""
    training_repos = {review.repo for review in training_reviews}
    evaluation_repos = {review.repo for review in evaluation_reviews}
    overlap = sorted(training_repos & evaluation_repos)
    if overlap:
        raise ValueError(
            "external training and evaluation repositories overlap: "
            + ", ".join(overlap)
        )


def _run_fold(
    *,
    model_name: str,
    encoded: dict[str, object],
    examples: list[object],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    parameters: ManualFinetuneParameters,
    fold: int,
    device: object,
    calibration_encoded: dict[str, object] | None = None,
    calibration_examples: list[object] | None = None,
    external_encoded: dict[str, object] | None = None,
    external_examples: list[object] | None = None,
    prediction_encoded: dict[str, object] | None = None,
    prediction_ids: list[str] | None = None,
) -> tuple[list[tuple[str, ...]], dict[str, Any]]:
    import torch
    from torch import nn
    from transformers import AutoModel

    torch.manual_seed(parameters.seed + fold)
    np.random.seed(parameters.seed + fold)

    class TaskPolicyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            model_kwargs: dict[str, object] = {}
            if parameters.load_in_4bit:
                from transformers import BitsAndBytesConfig

                model_kwargs.update({
                    "attn_implementation": "sdpa",
                    "device_map": {"": device},
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                    ),
                })
            self.encoder = AutoModel.from_pretrained(model_name, **model_kwargs)
            self.encoder.config.use_cache = False
            if parameters.lora_rank:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    prepare_model_for_kbit_training,
                )

                if parameters.load_in_4bit:
                    self.encoder = prepare_model_for_kbit_training(self.encoder)
                target_modules = _lora_target_modules(self.encoder)
                self.encoder = get_peft_model(
                    self.encoder,
                    LoraConfig(
                        r=parameters.lora_rank,
                        lora_alpha=parameters.lora_alpha,
                        lora_dropout=0.05,
                        target_modules=target_modules,
                    ),
                )
            encoder_size = int(self.encoder.config.hidden_size)
            if parameters.architecture == "shared_mean":
                self.body = self._body(encoder_size)
                self.methods = nn.Linear(parameters.hidden_size, len(METHOD_LABELS))
                self.task = nn.Linear(parameters.hidden_size, 1)
            elif parameters.architecture in {
                "label_attention",
                "label_attention_cardinality",
                "label_attention_lexical",
            }:
                self.token_body = self._body(encoder_size)
                self.global_body = self._body(encoder_size)
                self.label_queries = nn.Parameter(
                    torch.empty(len(METHOD_LABELS), parameters.hidden_size)
                )
                self.label_weights = nn.Parameter(
                    torch.empty(len(METHOD_LABELS), parameters.hidden_size)
                )
                self.label_bias = nn.Parameter(torch.zeros(len(METHOD_LABELS)))
                self.task = nn.Linear(parameters.hidden_size, 1)
                if parameters.architecture == "label_attention_lexical":
                    self.lexical_methods = nn.Linear(
                        LEXICAL_FEATURE_DIMENSIONS,
                        len(METHOD_LABELS),
                    )
                    self.lexical_task = nn.Linear(LEXICAL_FEATURE_DIMENSIONS, 1)
                if parameters.architecture == "label_attention_cardinality":
                    self.cardinality = nn.Linear(parameters.hidden_size, 4)
                nn.init.normal_(self.label_queries, std=0.02)
                nn.init.normal_(self.label_weights, std=0.02)
            else:
                raise ValueError(
                    "architecture must be 'shared_mean', 'label_attention', or "
                    "'label_attention_cardinality', or 'label_attention_lexical'"
                )

        def move_head_to(self, selected_device: object) -> None:
            """Move classifier-owned modules without recasting a quantized encoder."""
            for name, module in self.named_children():
                if name != "encoder":
                    module.to(selected_device)
            for name, parameter in tuple(self._parameters.items()):
                if parameter is not None:
                    self._parameters[name] = nn.Parameter(
                        parameter.to(selected_device),
                        requires_grad=parameter.requires_grad,
                    )

        def _body(self, encoder_size: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(encoder_size, parameters.hidden_size),
                nn.LayerNorm(parameters.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
            )

        def forward(self, **batch: object) -> tuple[object, object]:
            lexical_features = batch.pop("lexical_features", None)
            output = self.encoder(**batch).last_hidden_state.float()
            attention_mask = batch["attention_mask"]  # type: ignore[index]
            mask = attention_mask[..., None].bool()
            pooled = _pool_hidden_state(output, attention_mask, parameters.pooling)
            if parameters.architecture == "shared_mean":
                hidden = self.body(pooled)
                return self.methods(hidden), self.task(hidden).squeeze(1)

            token_hidden = self.token_body(output)
            global_hidden = self.global_body(pooled)
            attention_scores = torch.einsum(
                "bth,lh->btl", token_hidden, self.label_queries
            ) / parameters.hidden_size**0.5
            attention_scores = attention_scores.masked_fill(~mask, torch.finfo(
                attention_scores.dtype
            ).min)
            attention = attention_scores.softmax(dim=1)
            label_hidden = torch.einsum("btl,bth->blh", attention, token_hidden)
            label_hidden = label_hidden + global_hidden[:, None, :]
            method_logits = (label_hidden * self.label_weights[None, ...]).sum(dim=-1)
            method_logits = method_logits + self.label_bias
            task_logits = self.task(global_hidden).squeeze(1)
            if parameters.architecture == "label_attention_lexical":
                if lexical_features is None:
                    raise ValueError("label_attention_lexical requires lexical features")
                method_logits = method_logits + self.lexical_methods(lexical_features)
                task_logits = task_logits + self.lexical_task(lexical_features).squeeze(1)
            if parameters.architecture == "label_attention_cardinality":
                return method_logits, task_logits, self.cardinality(global_hidden)
            return method_logits, task_logits

    train_examples = [examples[index] for index in train_indices]
    validation_examples = [examples[index] for index in validation_indices]
    test_examples = [examples[index] for index in test_indices]
    train_x = _encoded_subset(encoded, train_indices)
    validation_x = _encoded_subset(encoded, validation_indices)
    test_x = _encoded_subset(encoded, test_indices)
    train_targets_np = _targets(train_examples)  # type: ignore[arg-type]
    train_targets = torch.from_numpy(train_targets_np.astype(np.float32))
    task_targets_np = np.asarray(
        [bool(example.policies) for example in train_examples], dtype=np.float32
    )
    task_targets = torch.from_numpy(task_targets_np)
    cardinality_targets_np = _cardinality_targets(train_examples)  # type: ignore[arg-type]
    cardinality_targets = torch.from_numpy(cardinality_targets_np)

    model = TaskPolicyModel()
    if parameters.load_in_4bit:
        model.move_head_to(device)
    else:
        model.to(device)
    if parameters.lora_rank:
        trainable_encoder = sum(
            parameter.numel()
            for parameter in model.encoder.parameters()
            if parameter.requires_grad
        )
        total_encoder = sum(parameter.numel() for parameter in model.encoder.parameters())
    else:
        trainable_encoder, total_encoder = _set_trainable_layers(
            model.encoder, parameters.unfrozen_layers
        )
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    lexical_parameter_names = {"lexical_methods", "lexical_task"}
    lexical_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.split(".", 1)[0] in lexical_parameter_names
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
        and name.split(".", 1)[0] not in lexical_parameter_names
    ]
    optimizer_groups = [
        {"params": encoder_parameters, "lr": parameters.encoder_learning_rate},
        {"params": head_parameters, "lr": parameters.head_learning_rate},
    ]
    if lexical_parameters:
        optimizer_groups.append({
            "params": lexical_parameters,
            "lr": parameters.lexical_learning_rate,
        })
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=parameters.weight_decay,
    )
    method_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(
            _positive_weights(train_targets_np, parameters.positive_weight_power)
        ).to(device)
    )
    exclusive_loss = nn.CrossEntropyLoss().to(device)
    task_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(
            _positive_weights(
                task_targets_np.reshape(-1, 1), parameters.positive_weight_power
            )
        ).squeeze(0).to(device)
    )
    cardinality_loss = nn.CrossEntropyLoss(
        weight=torch.from_numpy(
            _cardinality_class_weights(
                cardinality_targets_np,
                power=parameters.cardinality_balance_power,
            ).astype(np.float32)
        ).to(device)
    )
    best_key = (-1.0,) * 9
    best_state = None
    best_thresholds = (0.5,) * len(METHOD_LABELS)
    best_task_threshold = 0.5
    best_precision_floor = parameters.minimum_method_precision
    best_epoch = 0
    best_validation: dict[str, Any] = {}
    best_synthetic_validation: dict[str, Any] = {}
    best_natural_calibration: dict[str, Any] | None = None
    epoch_reports = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(1, parameters.epochs + 1):
        model.train()
        order = torch.randperm(len(train_indices))
        running_loss = 0.0
        for start in range(0, len(order), parameters.batch_size):
            indices = order[start:start + parameters.batch_size]
            batch = _batch_to_device(
                {
                    name: value[indices]  # type: ignore[index]
                    for name, value in train_x.items()
                },
                device,
            )
            method_target_batch = train_targets[indices].to(device)
            task_target_batch = task_targets[indices].to(device)
            cardinality_target_batch = cardinality_targets[indices].to(device)
            outputs = model(**batch)
            method_logits, task_logits = outputs[:2]
            loss = method_loss(method_logits, method_target_batch)
            loss = loss + parameters.task_loss_weight * task_loss(
                task_logits, task_target_batch
            )
            exclusive_mask = cardinality_target_batch == 1
            if bool(exclusive_mask.any()):
                exclusive_targets = method_target_batch[exclusive_mask].argmax(dim=1)
                loss = loss + parameters.exclusive_loss_weight * exclusive_loss(
                    method_logits[exclusive_mask],
                    exclusive_targets,
                )
            if len(outputs) == 3:
                loss = loss + parameters.cardinality_loss_weight * cardinality_loss(
                    outputs[2], cardinality_target_batch
                )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                encoder_parameters + head_parameters + lexical_parameters,
                1.0,
            )
            optimizer.step()
            running_loss += float(loss.detach()) * len(indices)
        method_scores, task_scores, cardinality_scores = _model_outputs(
            model,
            validation_x,
            batch_size=parameters.batch_size,
            device=device,
        )
        selection_method_scores = method_scores
        selection_task_scores = task_scores
        selection_cardinality_scores = cardinality_scores
        selection_examples = validation_examples
        calibration_outputs = None
        if calibration_encoded is not None and calibration_examples is not None:
            calibration_outputs = _model_outputs(
                model,
                calibration_encoded,
                batch_size=parameters.batch_size,
                device=device,
            )
            calibration_method, calibration_task, calibration_cardinality = (
                calibration_outputs
            )
            selection_method_scores = np.concatenate(
                (method_scores, calibration_method)
            )
            selection_task_scores = np.concatenate((task_scores, calibration_task))
            selection_examples = validation_examples + calibration_examples
            if cardinality_scores is not None and calibration_cardinality is not None:
                selection_cardinality_scores = np.concatenate(
                    (cardinality_scores, calibration_cardinality)
                )
        if selection_cardinality_scores is None:
            if parameters.threshold_calibration == "accuracy":
                if calibration_outputs is not None and calibration_examples is not None:
                    (
                        thresholds,
                        task_threshold,
                        validation_report,
                        selected_precision_floor,
                    ) = _select_domain_accuracy_thresholds(
                        method_scores,
                        task_scores,
                        validation_examples,
                        calibration_method,
                        calibration_task,
                        calibration_examples,
                        accuracy_target=parameters.minimum_method_accuracy,
                    )
                else:
                    (
                        thresholds,
                        task_threshold,
                        validation_report,
                        selected_precision_floor,
                    ) = _select_accuracy_thresholds(
                        selection_method_scores,
                        selection_task_scores,
                        selection_examples,
                        accuracy_target=parameters.minimum_method_accuracy,
                    )
            else:
                threshold_selector = (
                    select_joint_thresholds
                    if parameters.threshold_calibration == "joint"
                    else select_independent_thresholds
                )
                thresholds, task_threshold, validation_report = threshold_selector(
                    selection_method_scores,
                    selection_task_scores,
                    selection_examples,  # type: ignore[arg-type]
                    minimum_method_precision=parameters.minimum_method_precision,
                    minimum_method_recall=parameters.minimum_method_recall,
                )
                selected_precision_floor = parameters.minimum_method_precision
        else:
            thresholds = (0.0,) * len(METHOD_LABELS)
            task_threshold = 0.0
            validation_predictions = _predict_cardinality(
                selection_method_scores, selection_cardinality_scores
            )
            validation_report = _prediction_metrics(
                selection_examples, validation_predictions  # type: ignore[arg-type]
            )
            selected_precision_floor = parameters.minimum_method_precision
        synthetic_predictions = (
            _predict_cardinality(method_scores, cardinality_scores)
            if cardinality_scores is not None
            else _predict_independent(
                method_scores,
                task_scores,
                thresholds,
                task_threshold,
            )
        )
        synthetic_report = _prediction_metrics(
            validation_examples,
            synthetic_predictions,  # type: ignore[arg-type]
        )
        natural_report = None
        if calibration_outputs is not None and calibration_examples is not None:
            natural_predictions = (
                _predict_cardinality(calibration_method, calibration_cardinality)
                if calibration_cardinality is not None
                else _predict_independent(
                    calibration_method,
                    calibration_task,
                    thresholds,
                    task_threshold,
                )
            )
            natural_report = _prediction_metrics(
                calibration_examples,
                natural_predictions,  # type: ignore[arg-type]
            )
        key = (
            _domain_checkpoint_key(
                synthetic_report,
                natural_report,
                accuracy_target=parameters.minimum_method_accuracy,
            )
            if natural_report is not None
            else _individual_accuracy_checkpoint_key(
                validation_report,
                accuracy_target=parameters.minimum_method_accuracy,
            )
        )
        epoch_reports.append({
            "epoch": epoch,
            "loss": running_loss / len(train_indices),
            "exact_match": validation_report["exact_match"],
            "macro_f1": validation_report["macro_f1"],
            "micro_precision": validation_report["micro_precision"],
            "micro_recall": validation_report["micro_recall"],
            "false_activations": validation_report["false_activations"],
        })
        if key > best_key:
            best_key = key
            trainable_names = {
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
                if name in trainable_names
            }
            best_thresholds = thresholds
            best_task_threshold = task_threshold
            best_precision_floor = selected_precision_floor
            best_epoch = epoch
            best_validation = validation_report
            best_synthetic_validation = synthetic_report
            best_natural_calibration = natural_report
            stale = 0
        else:
            stale += 1
            if stale >= parameters.patience:
                break
    if best_state is None:
        raise RuntimeError("fine-tuning produced no checkpoint")
    model.load_state_dict(best_state, strict=False)
    method_scores, task_scores, cardinality_scores = _model_outputs(
        model, test_x, batch_size=parameters.batch_size, device=device
    )
    predictions = (
        _predict_cardinality(method_scores, cardinality_scores)
        if cardinality_scores is not None
        else _predict_independent(
            method_scores,
            task_scores,
            best_thresholds,
            best_task_threshold,
        )
    )
    external_evaluation = None
    if external_encoded is not None and external_examples is not None:
        external_method_scores, external_task_scores, external_cardinality_scores = (
            _model_outputs(
                model,
                external_encoded,
                batch_size=parameters.batch_size,
                device=device,
            )
        )
        external_predictions = (
            _predict_cardinality(external_method_scores, external_cardinality_scores)
            if external_cardinality_scores is not None
            else _predict_independent(
                external_method_scores,
                external_task_scores,
                best_thresholds,
                best_task_threshold,
            )
        )
        external_evaluation = {
            "predictions": [list(prediction) for prediction in external_predictions],
            "method_scores": external_method_scores.tolist(),
            "task_scores": external_task_scores.tolist(),
            "metrics": _prediction_metrics(
                external_examples, external_predictions  # type: ignore[arg-type]
            ),
        }
    report = {
        "fold": fold,
        "device": str(device),
        "train": len(train_indices),
        "validation": len(validation_indices),
        "test": len(test_indices),
        "best_epoch": best_epoch,
        "method_thresholds": best_thresholds,
        "task_threshold": best_task_threshold,
        "selected_precision_floor": best_precision_floor,
        "validation_metrics": best_validation,
        "synthetic_validation_metrics": best_synthetic_validation,
        "test_metrics": _prediction_metrics(test_examples, predictions),  # type: ignore[arg-type]
        "epochs": epoch_reports,
        "training_seconds": time.perf_counter() - started,
        "encoder_parameters": {
            "trainable": trainable_encoder,
            "total": total_encoder,
            "trainable_fraction": trainable_encoder / total_encoder,
        },
    }
    if best_natural_calibration is not None:
        report["natural_calibration_metrics"] = best_natural_calibration
    if external_evaluation is not None:
        report["external_evaluation"] = external_evaluation
    if prediction_encoded is not None and prediction_ids is not None:
        prediction_method, prediction_task, prediction_cardinality = _model_outputs(
            model,
            prediction_encoded,
            batch_size=parameters.batch_size,
            device=device,
        )
        frozen_predictions = (
            _predict_cardinality(prediction_method, prediction_cardinality)
            if prediction_cardinality is not None
            else _predict_independent(
                prediction_method,
                prediction_task,
                best_thresholds,
                best_task_threshold,
            )
        )
        report["prediction_only"] = {
            "ids": prediction_ids,
            "predictions": [list(item) for item in frozen_predictions],
            "method_scores": prediction_method.tolist(),
            "task_scores": prediction_task.tolist(),
        }
    del best_state, model
    gc.collect()
    if getattr(device, "type", None) == "cuda":
        torch.cuda.empty_cache()
    return predictions, report


def run_cross_validation(
    *,
    model_name: str = DEFAULT_CONTEXTUAL_MODEL,
    fold_count: int = 5,
    parameters: ManualFinetuneParameters | None = None,
    progress_path: Path | None = None,
    emit_progress: bool = False,
    selected_folds: tuple[int, ...] | None = None,
    training_candidates_path: Path | tuple[Path, ...] | None = None,
    training_reviews_paths: tuple[Path, ...] | None = None,
    calibration_candidates_path: Path | tuple[Path, ...] | None = None,
    calibration_reviews_path: Path | None = None,
    external_candidates_path: Path | tuple[Path, ...] | None = None,
    external_reviews_path: Path | None = None,
    prediction_candidates_path: Path | tuple[Path, ...] | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """Fine-tune a fresh encoder per fold and return out-of-fold predictions."""
    import torch
    from transformers import AutoTokenizer

    parameters = parameters or ManualFinetuneParameters()
    if parameters.load_in_4bit and not parameters.lora_rank:
        raise ValueError("load_in_4bit requires a positive lora_rank for training")
    if parameters.pooling not in {"mean", "last"}:
        raise ValueError("pooling must be 'mean' or 'last'")
    resolved_device = _resolve_device(device)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    rows = load_examples()
    examples = _examples(rows)
    folds = assign_folds(rows, fold_count)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left" if parameters.pooling == "last" else "right",
    )
    if (training_candidates_path is None) != (not training_reviews_paths):
        raise ValueError("training candidates and reviews must be provided together")
    training_reviews = []
    training_examples = []
    if training_candidates_path is not None and training_reviews_paths:
        for reviews_path in training_reviews_paths:
            training_reviews.extend(load_external_reviews(
                training_candidates_path,
                reviews_path,
            ))
        training_ids = [review.candidate_id for review in training_reviews]
        if len(training_ids) != len(set(training_ids)):
            raise ValueError("external training review files contain duplicate candidates")
        training_examples = [review.as_example() for review in training_reviews]
    all_examples = examples + training_examples
    encoded = tokenizer(
        [
            _format_model_input(example.text, parameters.query_instruction)
            for example in all_examples
        ],
        max_length=parameters.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    if parameters.architecture == "label_attention_lexical":
        encoded["lexical_features"] = torch.from_numpy(
            _hashed_lexical_features([example.text for example in all_examples])
        )
    if (external_candidates_path is None) != (external_reviews_path is None):
        raise ValueError("external candidates and reviews must be provided together")
    if (calibration_candidates_path is None) != (calibration_reviews_path is None):
        raise ValueError("calibration candidates and reviews must be provided together")
    calibration_reviews = []
    calibration_examples = None
    calibration_encoded = None
    if calibration_candidates_path is not None and calibration_reviews_path is not None:
        calibration_reviews = load_external_reviews(
            calibration_candidates_path,
            calibration_reviews_path,
        )
        _validate_external_partition(training_reviews, calibration_reviews)
        calibration_examples = [review.as_example() for review in calibration_reviews]
        calibration_encoded = tokenizer(
            [
                _format_model_input(example.text, parameters.query_instruction)
                for example in calibration_examples
            ],
            max_length=parameters.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if parameters.architecture == "label_attention_lexical":
            calibration_encoded["lexical_features"] = torch.from_numpy(
                _hashed_lexical_features(
                    [example.text for example in calibration_examples]
                )
            )
    external_examples = None
    external_encoded = None
    if external_candidates_path is not None and external_reviews_path is not None:
        external_reviews = load_external_reviews(
            external_candidates_path,
            external_reviews_path,
        )
        _validate_external_partition(training_reviews, external_reviews)
        _validate_external_partition(calibration_reviews, external_reviews)
        external_examples = [review.as_example() for review in external_reviews]
        external_encoded = tokenizer(
            [
                _format_model_input(example.text, parameters.query_instruction)
                for example in external_examples
            ],
            max_length=parameters.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if parameters.architecture == "label_attention_lexical":
            external_encoded["lexical_features"] = torch.from_numpy(
                _hashed_lexical_features([example.text for example in external_examples])
            )
    prediction_candidates = []
    prediction_encoded = None
    if prediction_candidates_path is not None:
        prediction_candidates = load_external_candidates(prediction_candidates_path)
        training_repos = {review.repo for review in training_reviews}
        overlap = sorted(training_repos & {item.repo for item in prediction_candidates})
        if overlap:
            raise ValueError(
                "external training and prediction repositories overlap: "
                + ", ".join(overlap)
            )
        prediction_encoded = tokenizer(
            [
                _format_model_input(item.text, parameters.query_instruction)
                for item in prediction_candidates
            ],
            max_length=parameters.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if parameters.architecture == "label_attention_lexical":
            prediction_encoded["lexical_features"] = torch.from_numpy(
                _hashed_lexical_features([item.text for item in prediction_candidates])
            )
    folds_to_run = selected_folds or tuple(range(fold_count))
    if not folds_to_run or any(fold < 0 or fold >= fold_count for fold in folds_to_run):
        raise ValueError("selected_folds must contain valid fold indices")
    predictions: list[tuple[str, ...] | None] = [None] * len(examples)
    training_only_indices = np.arange(
        len(examples),
        len(all_examples),
        dtype=np.int64,
    )
    fold_reports = []
    evaluated_indices = []
    for test_fold in folds_to_run:
        validation_fold = (test_fold + 1) % fold_count
        synthetic_train_indices = np.asarray([
            index
            for index, fold in enumerate(folds)
            if fold not in {test_fold, validation_fold}
        ])
        train_indices = np.concatenate((synthetic_train_indices, training_only_indices))
        validation_indices = np.asarray([
            index for index, fold in enumerate(folds) if fold == validation_fold
        ])
        test_indices = np.asarray([
            index for index, fold in enumerate(folds) if fold == test_fold
        ])
        fold_predictions, fold_report = _run_fold(
            model_name=model_name,
            encoded=encoded,
            examples=all_examples,
            train_indices=train_indices,
            validation_indices=validation_indices,
            test_indices=test_indices,
            parameters=parameters,
            fold=test_fold,
            device=resolved_device,
            calibration_encoded=calibration_encoded,
            calibration_examples=calibration_examples,
            external_encoded=external_encoded,
            external_examples=external_examples,
            prediction_encoded=prediction_encoded,
            prediction_ids=[item.candidate_id for item in prediction_candidates],
        )
        for index, prediction in zip(test_indices, fold_predictions, strict=True):
            predictions[int(index)] = prediction
        evaluated_indices.extend(int(index) for index in test_indices)
        fold_reports.append(fold_report)
        progress = {
            "version": FINETUNE_CV_VERSION,
            "fold_assignment": FOLD_ASSIGNMENT_VERSION,
            "model": model_name,
            "device": str(resolved_device),
            "parameters": asdict(parameters),
            "completed_folds": len(fold_reports),
            "total_folds": fold_count,
            "folds": fold_reports,
        }
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps(progress, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
        if emit_progress:
            print(json.dumps({
                "fold_complete": test_fold,
                "best_epoch": fold_report["best_epoch"],
                "training_seconds": fold_report["training_seconds"],
                "test": fold_report["test_metrics"],
            }, ensure_ascii=False, sort_keys=True), file=sys.stderr, flush=True)
    if selected_folds is None and any(prediction is None for prediction in predictions):
        raise RuntimeError("fine-tune cross-validation left rows without predictions")
    evaluated_indices.sort()
    evaluated_examples = [examples[index] for index in evaluated_indices]
    final_predictions = [
        predictions[index]
        for index in evaluated_indices
        if predictions[index] is not None
    ]
    aggregate = _prediction_metrics(evaluated_examples, final_predictions)
    errors = []
    false_activation_reasons: Counter[str] = Counter()
    evaluated_rows = [rows[index] for index in evaluated_indices]
    for row, expected, predicted in zip(
        evaluated_rows, evaluated_examples, final_predictions, strict=True
    ):
        if set(expected.policies) == set(predicted):
            continue
        if not expected.policies and predicted:
            false_activation_reasons[str(row["uncategorized_reason"])] += 1
        errors.append({
            "id": row["id"],
            "expected": list(expected.policies),
            "predicted": list(predicted),
            "uncategorized_reason": row["uncategorized_reason"],
            "language": row["language"],
            "project_type": row["project_type"],
            "style": row["style"],
            "scenario_family": row["scenario_family"],
        })
    aggregate.pop("error_sample", None)
    result = {
        "version": FINETUNE_CV_VERSION,
        "fold_assignment": FOLD_ASSIGNMENT_VERSION,
        "purpose": "development diagnostic; not sealed-holdout evidence",
        "model": model_name,
        "device": str(resolved_device),
        "parameters": asdict(parameters),
        "examples": len(examples),
        "training_only_examples": len(training_examples),
        "natural_calibration_examples": len(calibration_examples or []),
        "evaluated_examples": len(evaluated_examples),
        "selected_folds": list(folds_to_run),
        "folds": fold_reports,
        "aggregate": aggregate,
        "false_activation_reasons": dict(sorted(false_activation_reasons.items())),
        "errors": errors,
    }
    if external_examples is not None:
        predictions_by_fold = [
            [
                tuple(prediction)
                for prediction in report["external_evaluation"]["predictions"]
            ]
            for report in fold_reports
        ]
        consensus = _consensus_predictions(predictions_by_fold)
        result["external_evaluation"] = {
            "purpose": "development validation; not sealed-holdout evidence",
            "examples": len(external_examples),
            "fold_metrics": [
                report["external_evaluation"]["metrics"] for report in fold_reports
            ],
            "consensus_metrics": _prediction_metrics(
                external_examples,
                consensus,
            ),
        }
    if prediction_candidates:
        predictions_by_fold = [
            [tuple(item) for item in report["prediction_only"]["predictions"]]
            for report in fold_reports
        ]
        result["prediction_only"] = {
            "purpose": "frozen unlabeled predictions; no evaluation labels were loaded",
            "ids": [item.candidate_id for item in prediction_candidates],
            "consensus": [list(item) for item in _consensus_predictions(predictions_by_fold)],
            "folds": [report["prediction_only"] for report in fold_reports],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_CONTEXTUAL_MODEL)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:N, or another torch device string",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--pooling", choices=("mean", "last"), default="mean")
    parser.add_argument("--query-instruction")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--unfrozen-layers", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument(
        "--architecture",
        choices=(
            "shared_mean",
            "label_attention",
            "label_attention_cardinality",
            "label_attention_lexical",
        ),
        default="shared_mean",
    )
    parser.add_argument(
        "--threshold-calibration",
        choices=("independent", "joint", "accuracy"),
        default="independent",
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--lexical-learning-rate", type=float, default=1e-2)
    parser.add_argument("--positive-weight-power", type=float, default=0.5)
    parser.add_argument("--exclusive-loss-weight", type=float, default=0.7)
    parser.add_argument("--cardinality-loss-weight", type=float, default=0.7)
    parser.add_argument("--cardinality-balance-power", type=float, default=0.5)
    parser.add_argument("--minimum-method-precision", type=float, default=0.85)
    parser.add_argument("--minimum-method-recall", type=float)
    parser.add_argument("--minimum-method-accuracy", type=float, default=0.95)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--only-fold", type=int, action="append")
    parser.add_argument("--training-candidates", type=Path, action="append")
    parser.add_argument("--training-reviews", type=Path, action="append")
    parser.add_argument("--calibration-candidates", type=Path, action="append")
    parser.add_argument("--calibration-reviews", type=Path)
    parser.add_argument("--external-candidates", type=Path, action="append")
    parser.add_argument("--external-reviews", type=Path)
    parser.add_argument("--prediction-candidates", type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_cross_validation(
        model_name=args.model,
        fold_count=args.folds,
        parameters=ManualFinetuneParameters(
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            unfrozen_layers=args.unfrozen_layers,
            hidden_size=args.hidden_size,
            architecture=args.architecture,
            pooling=args.pooling,
            query_instruction=args.query_instruction,
            load_in_4bit=args.load_in_4bit,
            threshold_calibration=args.threshold_calibration,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            lexical_learning_rate=args.lexical_learning_rate,
            positive_weight_power=args.positive_weight_power,
            exclusive_loss_weight=args.exclusive_loss_weight,
            cardinality_loss_weight=args.cardinality_loss_weight,
            cardinality_balance_power=args.cardinality_balance_power,
            minimum_method_precision=args.minimum_method_precision,
            minimum_method_recall=args.minimum_method_recall,
            minimum_method_accuracy=args.minimum_method_accuracy,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            patience=args.patience,
            seed=args.seed,
        ),
        progress_path=(
            args.output.with_suffix(args.output.suffix + ".partial")
            if args.output is not None
            else None
        ),
        emit_progress=True,
        selected_folds=(tuple(args.only_fold) if args.only_fold else None),
        training_candidates_path=(
            tuple(args.training_candidates) if args.training_candidates else None
        ),
        training_reviews_paths=(tuple(args.training_reviews) if args.training_reviews else None),
        calibration_candidates_path=(
            tuple(args.calibration_candidates) if args.calibration_candidates else None
        ),
        calibration_reviews_path=args.calibration_reviews,
        external_candidates_path=(
            tuple(args.external_candidates) if args.external_candidates else None
        ),
        external_reviews_path=args.external_reviews,
        prediction_candidates_path=(
            tuple(args.prediction_candidates) if args.prediction_candidates else None
        ),
        device=args.device,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "model": report["model"],
        "device": report["device"],
        "examples": report["examples"],
        "aggregate": report["aggregate"],
        "false_activation_reasons": report["false_activation_reasons"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "FINETUNE_CV_VERSION",
    "ManualFinetuneParameters",
    "LEXICAL_FEATURE_DIMENSIONS",
    "_hashed_lexical_features",
    "_format_model_input",
    "_pool_hidden_state",
    "_lora_target_modules",
    "_resolve_device",
    "_model_scores",
    "_consensus_predictions",
    "_individual_accuracy_checkpoint_key",
    "_select_accuracy_thresholds",
    "_validate_external_partition",
    "_predict_cardinality",
    "_positive_weights",
    "run_cross_validation",
]
