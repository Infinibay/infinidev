"""Train a fixed six-logit encoder classifier on natural task-policy data.

The training command never opens the evaluation partition. Evaluation is a
separate explicit command so architecture and loss selection remain confined
to the training and calibration partitions.
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

from bench.task_policy_gliclass_finetune import (
    ACCURACY_TARGET,
    LABEL_STYLES,
    _binary_metrics,
    _load_partition,
    _positive_weights,
    _prediction_report,
    _targets,
    _validate_partitions,
)
from bench.task_policy_multilabel_head import METHOD_LABELS


RUN_VERSION = "task-policy-fixed-encoder-natural-v2"
QUERY_TOKENS = tuple(
    f"<|task_policy_{label.split('.', 1)[0]}|>" for label in METHOD_LABELS
)


@dataclass(frozen=True)
class EncoderParameters:
    """Hyperparameters selected without reading natural evaluation labels."""

    model_name: str
    architecture: str = "cls"
    loss: str = "weighted_bce"
    sampling: str = "uniform"
    max_length: int = 1024
    batch_size: int = 4
    evaluation_batch_size: int = 12
    gradient_accumulation_steps: int = 8
    epochs: int = 5
    early_stopping_patience: int = 0
    encoder_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    positive_weight_power: float = 0.5
    task_loss_weight: float = 0.4
    exclusive_loss_weight: float = 0.4
    warmup_ratio: float = 0.06
    minimum_precision: float = 0.85
    minimum_recall: float = 0.95
    accuracy_target: float = ACCURACY_TARGET
    model_label_weight: float = 0.5
    minimum_positive_support: int = 1_000
    seed: int = 41
    bf16: bool = True


def _asymmetric_loss(
    logits: object,
    targets: object,
    *,
    gamma_negative: float = 4.0,
    gamma_positive: float = 1.0,
    clip: float = 0.05,
    example_weights: object | None = None,
) -> object:
    """Return ASL for independent sigmoid outputs.

    This follows the official Alibaba-MIIL formulation: easy negatives are
    clipped and down-weighted more aggressively than positives.
    """
    import torch

    positive = torch.sigmoid(logits)
    negative = 1.0 - positive
    if clip > 0:
        negative = (negative + clip).clamp(max=1.0)
    loss = targets * torch.log(positive.clamp(min=1e-8))
    loss = loss + (1.0 - targets) * torch.log(negative.clamp(min=1e-8))
    if gamma_negative > 0 or gamma_positive > 0:
        probability = positive * targets + negative * (1.0 - targets)
        gamma = gamma_positive * targets + gamma_negative * (1.0 - targets)
        loss = loss * torch.pow(1.0 - probability, gamma)
    per_example = -loss.mean(dim=1)
    if example_weights is None:
        return per_example.mean()
    weights = example_weights.to(per_example.device, dtype=per_example.dtype)
    return (per_example * weights).sum() / weights.sum().clamp(min=1e-8)


def _weighted_mean(values: object, weights: object) -> object:
    values = values.float()
    weights = weights.to(values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp(min=1e-8)


def _example_weights(reviews: list[object], model_label_weight: float) -> np.ndarray:
    """Give human labels full weight and temper model labels by confidence."""
    if not 0 < model_label_weight <= 1:
        raise ValueError("model_label_weight must be in (0, 1]")
    values = []
    for review in reviews:
        kind = str(getattr(review, "annotation_kind", "human"))
        confidence = float(getattr(review, "annotation_confidence", 1.0))
        if kind == "human":
            values.append(1.0)
        elif kind == "model":
            if not 0 <= confidence <= 1:
                raise ValueError("model annotation confidence must be in [0, 1]")
            values.append(model_label_weight * confidence)
        else:
            raise ValueError(f"unknown annotation kind: {kind}")
    return np.asarray(values, dtype=np.float32)


def _require_minimum_positive_support(
    reviews: list[object], minimum: int,
) -> dict[str, int]:
    """Refuse candidate training runs with an under-supported category."""
    if minimum < 0:
        raise ValueError("minimum positive support must not be negative")
    support = {
        label: sum(label in review.policies for review in reviews)
        for label in METHOD_LABELS
    }
    missing = {label: count for label, count in support.items() if count < minimum}
    if missing:
        detail = ", ".join(f"{label}={count}" for label, count in missing.items())
        raise ValueError(
            f"training requires at least {minimum} positive examples per category; {detail}"
        )
    return support



def _sampling_probabilities(
    targets: np.ndarray,
    *,
    maximum_weight: float = 5.0,
) -> np.ndarray:
    """Temper rare-label sampling while retaining neutral examples."""
    positives = targets.sum(axis=0)
    label_weights = np.sqrt(len(targets) / np.maximum(positives, 1))
    label_weights = np.minimum(label_weights, maximum_weight)
    row_weights = np.ones(len(targets), dtype=np.float64)
    for index, row in enumerate(targets.astype(bool)):
        if row.any():
            row_weights[index] = float(np.max(label_weights[row]))
    return row_weights / row_weights.sum()


def _mean_pool(hidden: object, attention_mask: object) -> object:
    mask = attention_mask[..., None].bool()
    return hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _last_pool(hidden: object, attention_mask: object) -> object:
    """Pool the final valid token for either left- or right-padded batches."""
    import torch

    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last_positions = positions.masked_fill(~attention_mask.bool(), -1).max(dim=1).values
    if (last_positions < 0).any():
        raise ValueError("cannot pool an input with no valid tokens")
    batch_positions = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_positions, last_positions]


def _early_stopping_reached(epoch: int, best_epoch: int, patience: int) -> bool:
    """Stop after ``patience`` complete epochs without a better checkpoint."""
    return patience > 0 and epoch - best_epoch >= patience


def _select_balanced_threshold(
    scores: np.ndarray,
    expected: np.ndarray,
    *,
    minimum_precision: float,
    minimum_recall: float,
    accuracy_target: float = ACCURACY_TARGET,
) -> float:
    """Choose a threshold by the actual accuracy and recall acceptance gate."""
    candidates = sorted({
        *(float(value) for value in np.arange(0.05, 0.951, 0.025)),
        *(float(value) for value in scores),
        *(float(np.nextafter(value, np.inf)) for value in scores),
    })
    best: tuple[tuple[float, ...], float] | None = None
    for threshold in candidates:
        metrics = _binary_metrics(scores, expected, threshold)
        accuracy = float(metrics["accuracy"])
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        f1 = float(metrics["f1"])
        meets_gate = accuracy >= accuracy_target and recall >= minimum_recall
        key = (
            float(meets_gate),
            min(accuracy / accuracy_target, recall / minimum_recall),
            min(accuracy, recall),
            f1,
            float(precision >= minimum_precision),
            precision,
            accuracy,
            recall,
            -abs(threshold - 0.5),
        )
        if best is None or key > best[0]:
            best = (key, threshold)
    if best is None:
        raise RuntimeError("threshold calibration produced no candidate")
    return best[1]


def _balanced_checkpoint_key(
    report: dict[str, Any],
    *,
    minimum_precision: float,
    minimum_recall: float,
    accuracy_target: float = ACCURACY_TARGET,
) -> tuple[float, ...]:
    """Rank checkpoints by the weakest accuracy/recall gate margin."""
    metrics = [report["per_label"][label] for label in METHOD_LABELS]
    accuracies = [float(item["accuracy"]) for item in metrics]
    precisions = [float(item["precision"]) for item in metrics]
    recalls = [float(item["recall"]) for item in metrics]
    f1_scores = [float(item["f1"]) for item in metrics]
    margins = [
        min(accuracy / accuracy_target, recall / minimum_recall)
        for accuracy, recall in zip(accuracies, recalls, strict=True)
    ]
    passing = sum(
        accuracy >= accuracy_target and recall >= minimum_recall
        for accuracy, recall in zip(accuracies, recalls, strict=True)
    )
    return (
        float(passing == len(METHOD_LABELS)),
        passing / len(METHOD_LABELS),
        min(margins),
        float(np.mean(margins)),
        min(accuracies),
        min(recalls),
        min(f1_scores),
        float(np.mean(f1_scores)),
        float(np.mean(accuracies)),
        float(np.mean(recalls)),
        float(np.mean(np.asarray(precisions) >= minimum_precision)),
        float(np.mean(precisions)),
        float(report["exact_match"]),
    )


def _build_model(
    model_name: str | Path,
    *,
    architecture: str,
    label_queries: object | None = None,
    tokenizer_size: int | None = None,
) -> object:
    import torch
    from torch import nn
    from transformers import AutoModel

    if architecture not in {
        "cls", "mean", "last", "label_attention", "query_tokens", "query2label",
    }:
        raise ValueError(
            "architecture must be 'cls', 'mean', 'last', 'label_attention', or "
            "'query_tokens', or 'query2label'"
        )

    class TaskPolicyEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            if tokenizer_size is not None:
                self.encoder.resize_token_embeddings(tokenizer_size)
            self.encoder.config.use_cache = False
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(0.1)
            self.task = nn.Linear(hidden_size, 1)
            self.architecture = architecture
            if architecture == "query_tokens":
                self.label_weights = nn.Parameter(
                    torch.empty(len(METHOD_LABELS), hidden_size)
                )
                self.label_bias = nn.Parameter(torch.zeros(len(METHOD_LABELS)))
                nn.init.normal_(self.label_weights, std=0.02)
            else:
                self.methods = nn.Linear(hidden_size, len(METHOD_LABELS))
            if architecture in {"label_attention", "query2label"}:
                initial = torch.empty(len(METHOD_LABELS), hidden_size)
                nn.init.normal_(initial, std=0.02)
                if label_queries is not None:
                    initial.copy_(label_queries)
                self.label_queries = nn.Parameter(initial)
                self.label_weights = nn.Parameter(torch.empty_like(initial))
                self.label_bias = nn.Parameter(torch.zeros(len(METHOD_LABELS)))
                nn.init.normal_(self.label_weights, std=0.02)
            if architecture == "query2label":
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=hidden_size,
                    nhead=8,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.label_decoder = nn.TransformerDecoder(
                    decoder_layer,
                    num_layers=2,
                    norm=nn.LayerNorm(hidden_size),
                )

        def forward(self, **batch: object) -> tuple[object, object]:
            import torch.nn.functional as functional

            outputs = self.encoder(**batch).last_hidden_state
            mask = batch["attention_mask"]
            if self.architecture == "query_tokens":
                query_hidden = self.dropout(outputs[:, -len(METHOD_LABELS):])
                method_logits = (
                    query_hidden * self.label_weights[None, ...]
                ).sum(dim=-1) + self.label_bias
                pooled = self.dropout(_mean_pool(
                    outputs[:, :-len(METHOD_LABELS)],
                    mask[:, :-len(METHOD_LABELS)],
                ))
                return method_logits, self.task(pooled).squeeze(1)
            pooled_state = (
                outputs[:, 0]
                if self.architecture == "cls"
                else _last_pool(outputs, mask)
                if self.architecture == "last"
                else _mean_pool(outputs, mask)
            )
            pooled = self.dropout(pooled_state)
            task_logits = self.task(pooled).squeeze(1)
            if self.architecture in {"cls", "mean", "last"}:
                return self.methods(pooled), task_logits
            if self.architecture == "query2label":
                queries = self.label_queries[None, ...].expand(len(outputs), -1, -1)
                label_hidden = self.label_decoder(
                    queries.to(outputs.dtype),
                    outputs,
                    memory_key_padding_mask=~mask.bool(),
                )
                label_hidden = self.dropout(label_hidden)
                method_logits = (
                    label_hidden * self.label_weights[None, ...]
                ).sum(dim=-1) + self.label_bias
                return method_logits, task_logits
            normalized_tokens = functional.normalize(outputs.float(), dim=-1)
            normalized_queries = functional.normalize(self.label_queries.float(), dim=-1)
            attention_scores = torch.einsum(
                "bth,lh->btl", normalized_tokens, normalized_queries
            ) * 10.0
            attention_scores = attention_scores.masked_fill(
                ~mask[..., None].bool(),
                torch.finfo(attention_scores.dtype).min,
            )
            attention = attention_scores.softmax(dim=1).to(outputs.dtype)
            label_hidden = torch.einsum("btl,bth->blh", attention, outputs)
            label_hidden = self.dropout(label_hidden + pooled[:, None, :])
            method_logits = (label_hidden * self.label_weights[None, ...]).sum(dim=-1)
            return method_logits + self.label_bias, task_logits

    return TaskPolicyEncoder()


def _semantic_label_queries(
    model_name: str,
    *,
    tokenizer: object,
    max_length: int = 48,
) -> object:
    """Initialize label attention from concise semantic label embeddings."""
    import torch
    from transformers import AutoModel

    encoder = AutoModel.from_pretrained(model_name)
    encoded = tokenizer(
        [LABEL_STYLES["descriptive"][label] for label in METHOD_LABELS],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoder.eval()
    with torch.inference_mode():
        hidden = encoder(**encoded).last_hidden_state
        queries = _mean_pool(hidden, encoded["attention_mask"]).float()
    del encoder
    return queries


def _encoded_batch(
    tokenizer: object,
    texts: list[str],
    *,
    max_length: int,
    device: object,
    architecture: str = "mean",
) -> dict[str, object]:
    import torch

    content_length = max_length
    if architecture == "query_tokens":
        content_length -= len(QUERY_TOKENS)
        if content_length <= 0:
            raise ValueError("max_length is too short for task-policy query tokens")
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=content_length,
        return_tensors="pt",
    )
    if architecture == "query_tokens":
        if set(encoded) != {"input_ids", "attention_mask"}:
            raise ValueError(
                "query-token architecture requires input_ids and attention_mask only"
            )
        query_ids = tokenizer.convert_tokens_to_ids(list(QUERY_TOKENS))
        if any(token_id == tokenizer.unk_token_id for token_id in query_ids):
            raise ValueError("task-policy query tokens are missing from the tokenizer")
        batch_size = encoded["input_ids"].shape[0]
        queries = torch.tensor(query_ids, dtype=torch.long)[None, :].expand(batch_size, -1)
        encoded["input_ids"] = torch.cat((encoded["input_ids"], queries), dim=1)
        encoded["attention_mask"] = torch.cat((
            encoded["attention_mask"],
            torch.ones_like(queries),
        ), dim=1)
    return {name: value.to(device) for name, value in encoded.items()}


def _score(
    model: object,
    tokenizer: object,
    reviews: list[object],
    *,
    max_length: int,
    batch_size: int,
    device: object,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    method_chunks = []
    task_chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(reviews), batch_size):
            rows = reviews[start:start + batch_size]
            batch = _encoded_batch(
                tokenizer,
                [row.text for row in rows],
                max_length=max_length,
                device=device,
                architecture=model.architecture,
            )
            method_logits, task_logits = model(**batch)
            method_chunks.append(method_logits.sigmoid().float().cpu().numpy())
            task_chunks.append(task_logits.sigmoid().float().cpu().numpy())
    return np.concatenate(method_chunks), np.concatenate(task_chunks)


def _combined_report(
    method_scores: np.ndarray,
    task_scores: np.ndarray,
    expected: np.ndarray,
    parameters: EncoderParameters,
) -> tuple[tuple[float, ...], float, dict[str, Any]]:
    """Calibrate method thresholds and an optional global abstention gate."""
    candidates = (0.0, 0.15, 0.25, 0.35, 0.45, 0.55)
    best: tuple[tuple[float, ...], tuple[float, ...], float, dict[str, Any]] | None = None
    for task_threshold in candidates:
        gated = method_scores.copy()
        gated[task_scores < task_threshold] = 0.0
        thresholds = tuple(
            _select_balanced_threshold(
                gated[:, index], expected[:, index],
                minimum_precision=parameters.minimum_precision,
                minimum_recall=parameters.minimum_recall,
                accuracy_target=parameters.accuracy_target,
            )
            for index in range(len(METHOD_LABELS))
        )
        report = _prediction_report(gated, expected, thresholds)
        report["gate"] = {
            "accuracy_target": parameters.accuracy_target,
            "recall_target": parameters.minimum_recall,
            "all_labels_pass": all(
                float(metrics["accuracy"]) >= parameters.accuracy_target
                and float(metrics["recall"]) >= parameters.minimum_recall
                for metrics in report["per_label"].values()
            ),
        }
        key = _balanced_checkpoint_key(
            report,
            minimum_precision=parameters.minimum_precision,
            minimum_recall=parameters.minimum_recall,
            accuracy_target=parameters.accuracy_target,
        )
        if best is None or key > best[0]:
            best = (key, thresholds, task_threshold, report)
    if best is None:
        raise RuntimeError("task-gate calibration produced no candidate")
    return best[1], best[2], best[3]


def _save_best(
    model: object,
    tokenizer: object,
    output_dir: Path,
    *,
    parameters: EncoderParameters,
    thresholds: tuple[float, ...],
    task_threshold: float,
    epoch: int,
    calibration: dict[str, Any],
) -> None:
    import torch
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_dir = output_dir / "encoder"
    model.encoder.save_pretrained(encoder_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    head_state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if not name.startswith("encoder.")
    }
    save_file(head_state, output_dir / "head.safetensors")
    metadata = {
        "run_version": RUN_VERSION,
        "parameters": asdict(parameters),
        "thresholds": dict(zip(METHOD_LABELS, thresholds, strict=True)),
        "task_threshold": task_threshold,
        "selected_epoch": epoch,
        "calibration": calibration,
    }
    (output_dir / "task_policy_config.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.cuda.empty_cache()


def _load_frozen(output_dir: Path, device: object) -> tuple[object, object, dict[str, Any]]:
    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    metadata = json.loads(
        (output_dir / "task_policy_config.json").read_text(encoding="utf-8")
    )
    parameters = metadata["parameters"]
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    model = _build_model(
        output_dir / "encoder",
        architecture=parameters["architecture"],
        tokenizer_size=len(tokenizer),
    )
    missing, unexpected = model.load_state_dict(
        load_file(output_dir / "head.safetensors"),
        strict=False,
    )
    missing = [name for name in missing if not name.startswith("encoder.")]
    if missing or unexpected:
        raise RuntimeError(f"invalid frozen head; missing={missing}, unexpected={unexpected}")
    model.to(device)
    return model, tokenizer, metadata


def train(args: argparse.Namespace, device: object) -> None:
    import torch
    from torch import nn
    from torch.nn import functional
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite_output_dir:
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}; pass --overwrite-output-dir"
        )
    parameters = EncoderParameters(
        model_name=args.model,
        architecture=args.architecture,
        loss=args.loss,
        sampling=args.sampling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        evaluation_batch_size=args.evaluation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        encoder_learning_rate=args.encoder_learning_rate,
        head_learning_rate=args.head_learning_rate,
        bf16=not args.no_bf16,
        model_label_weight=args.model_label_weight,
        minimum_positive_support=args.minimum_positive_support,
    )
    training = _load_partition(args.data_root, "training")
    calibration = _load_partition(args.data_root, "calibration")
    _require_minimum_positive_support(training, parameters.minimum_positive_support)
    _validate_partitions(training, calibration)
    tokenizer = AutoTokenizer.from_pretrained(parameters.model_name)
    if parameters.architecture == "query_tokens":
        tokenizer.add_special_tokens({"additional_special_tokens": list(QUERY_TOKENS)})
    queries = None
    if parameters.architecture in {"label_attention", "query2label"}:
        queries = _semantic_label_queries(parameters.model_name, tokenizer=tokenizer)
    model = _build_model(
        parameters.model_name,
        architecture=parameters.architecture,
        label_queries=queries,
        tokenizer_size=len(tokenizer),
    )
    model.to(device)
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": parameters.encoder_learning_rate},
            {"params": head_parameters, "lr": parameters.head_learning_rate},
        ],
        weight_decay=parameters.weight_decay,
    )
    batches_per_epoch = math.ceil(len(training) / parameters.batch_size)
    updates_per_epoch = math.ceil(
        batches_per_epoch / parameters.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * parameters.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        round(total_updates * parameters.warmup_ratio),
        total_updates,
    )
    target_matrix = _targets(training).astype(np.float32)
    positive_weights = torch.tensor(
        _positive_weights(target_matrix, parameters.positive_weight_power),
        device=device,
    )
    task_targets = np.asarray([bool(row.policies) for row in training], dtype=np.float32)
    example_weights = _example_weights(training, parameters.model_label_weight)
    task_positive_weight = torch.tensor(
        _positive_weights(task_targets[:, None], parameters.positive_weight_power),
        device=device,
    )
    use_bf16 = parameters.bf16 and getattr(device, "type", None) == "cuda"
    best_key = None
    best_epoch = 0
    history = []
    stopped_early = False
    for epoch in range(1, parameters.epochs + 1):
        model.train()
        generator = np.random.default_rng(parameters.seed + epoch)
        if parameters.sampling == "uniform":
            order = generator.permutation(len(training))
        elif parameters.sampling == "label_balanced":
            order = generator.choice(
                len(training),
                size=len(training),
                replace=True,
                p=_sampling_probabilities(target_matrix),
            )
        else:
            raise ValueError("sampling must be 'uniform' or 'label_balanced'")
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_number, start in enumerate(range(0, len(order), parameters.batch_size), 1):
            indices = order[start:start + parameters.batch_size]
            rows = [training[int(index)] for index in indices]
            batch = _encoded_batch(
                tokenizer,
                [row.text for row in rows],
                max_length=parameters.max_length,
                device=device,
                architecture=parameters.architecture,
            )
            expected = torch.tensor(target_matrix[indices], device=device)
            expected_task = torch.tensor(task_targets[indices], device=device)
            batch_weights = torch.tensor(example_weights[indices], device=device)
            with torch.autocast(
                device_type=getattr(device, "type", str(device)),
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                method_logits, task_logits = model(**batch)
                if parameters.loss == "weighted_bce":
                    method_loss = functional.binary_cross_entropy_with_logits(
                        method_logits.float(),
                        expected,
                        pos_weight=positive_weights,
                        reduction="none",
                    )
                    loss = _weighted_mean(method_loss.mean(dim=1), batch_weights)
                elif parameters.loss == "asymmetric":
                    loss = _asymmetric_loss(
                        method_logits.float(), expected, example_weights=batch_weights
                    )
                else:
                    raise ValueError("loss must be 'weighted_bce' or 'asymmetric'")
                task_loss = functional.binary_cross_entropy_with_logits(
                    task_logits.float(),
                    expected_task,
                    pos_weight=task_positive_weight,
                    reduction="none",
                )
                loss = loss + parameters.task_loss_weight * _weighted_mean(
                    task_loss, batch_weights
                )
                exclusive = expected.sum(dim=1) == 1
                if bool(exclusive.any()):
                    exclusive_loss = functional.cross_entropy(
                        method_logits[exclusive].float(),
                        expected[exclusive].argmax(dim=1),
                        reduction="none",
                    )
                    loss = loss + parameters.exclusive_loss_weight * _weighted_mean(
                        exclusive_loss, batch_weights[exclusive]
                    )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}, batch {batch_number}: {loss}"
                )
            losses.append(float(loss.detach()))
            (loss / parameters.gradient_accumulation_steps).backward()
            if (
                batch_number % parameters.gradient_accumulation_steps == 0
                or start + parameters.batch_size >= len(order)
            ):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        method_scores, task_scores = _score(
            model,
            tokenizer,
            calibration,
            max_length=parameters.max_length,
            batch_size=parameters.evaluation_batch_size,
            device=device,
        )
        thresholds, task_threshold, report = _combined_report(
            method_scores,
            task_scores,
            _targets(calibration),
            parameters,
        )
        key = _balanced_checkpoint_key(
            report,
            minimum_precision=parameters.minimum_precision,
            minimum_recall=parameters.minimum_recall,
            accuracy_target=parameters.accuracy_target,
        )
        record = {
            "epoch": epoch,
            "training_loss": float(np.mean(losses)),
            "calibration": report,
            "thresholds": thresholds,
            "task_threshold": task_threshold,
        }
        history.append(record)
        print(json.dumps({
            "epoch": epoch,
            "loss": record["training_loss"],
            "calibration_exact_match": report["exact_match"],
            "calibration_gate": report["gate"],
        }, sort_keys=True), flush=True)
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            _save_best(
                model,
                tokenizer,
                args.output_dir,
                parameters=parameters,
                thresholds=thresholds,
                task_threshold=task_threshold,
                epoch=epoch,
                calibration=report,
            )
        elif _early_stopping_reached(
            epoch, best_epoch, parameters.early_stopping_patience,
        ):
            stopped_early = True
            print(json.dumps({
                "early_stopping": True,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "patience": parameters.early_stopping_patience,
            }, sort_keys=True), flush=True)
            break
    selection_report = {
        "run_version": RUN_VERSION,
        "parameters": asdict(parameters),
        "counts": {"training": len(training), "calibration": len(calibration)},
        "selected_epoch": best_epoch,
        "completed_epochs": len(history),
        "stopped_early": stopped_early,
        "history": history,
        "evaluation_opened": False,
    }
    (args.output_dir / "selection_report.json").write_text(
        json.dumps(selection_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "checkpoint": str(args.output_dir),
        "selected_epoch": best_epoch,
        "evaluation_opened": False,
    }, sort_keys=True), flush=True)


def _latency(
    model: object,
    tokenizer: object,
    reviews: list[object],
    *,
    parameters: dict[str, Any],
    device: object,
    samples: int,
) -> dict[str, float | int]:
    import torch

    times = []
    for index in range(samples + 10):
        review = reviews[index % len(reviews)]
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        _score(
            model,
            tokenizer,
            [review],
            max_length=parameters["max_length"],
            batch_size=1,
            device=device,
        )
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)
        if index >= 10:
            times.append((time.perf_counter() - started) * 1000)
    return {
        "samples": samples,
        "p50_ms": statistics.median(times),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": max(times),
    }


def evaluate(args: argparse.Namespace, device: object) -> None:
    model, tokenizer, metadata = _load_frozen(args.output_dir, device)
    parameters = metadata["parameters"]
    evaluation = _load_partition(args.data_root, "evaluation")
    method_scores, task_scores = _score(
        model,
        tokenizer,
        evaluation,
        max_length=parameters["max_length"],
        batch_size=parameters["evaluation_batch_size"],
        device=device,
    )
    method_scores[task_scores < float(metadata["task_threshold"])] = 0.0
    thresholds = tuple(float(metadata["thresholds"][label]) for label in METHOD_LABELS)
    report = {
        "run_version": RUN_VERSION,
        "parameters": parameters,
        "evaluation": _prediction_report(method_scores, _targets(evaluation), thresholds),
        "latency": _latency(
            model,
            tokenizer,
            evaluation,
            parameters=parameters,
            device=device,
            samples=args.latency_samples,
        ),
    }
    path = args.output_dir / "evaluation_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(path),
        "exact_match": report["evaluation"]["exact_match"],
        "gate": report["evaluation"]["gate"],
        "latency": report["latency"],
    }, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path.home() / "tmp" / "task-policy-natural-split-v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument(
        "--architecture",
        choices=("cls", "mean", "last", "label_attention", "query_tokens", "query2label"),
        default="cls",
    )
    parser.add_argument("--loss", choices=("weighted_bce", "asymmetric"), default="weighted_bce")
    parser.add_argument(
        "--sampling",
        choices=("uniform", "label_balanced"),
        default="uniform",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-batch-size", type=int, default=12)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="stop after this many non-improving calibration epochs; 0 disables it",
    )
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--model-label-weight", type=float, default=0.5)
    parser.add_argument("--minimum-positive-support", type=int, default=1_000)
    parser.add_argument("--latency-samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite-output-dir", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(
        args.max_length,
        args.batch_size,
        args.evaluation_batch_size,
        args.gradient_accumulation_steps,
        args.epochs,
    ) <= 0:
        raise ValueError("length, batch sizes, accumulation, and epochs must be positive")
    if not 0 < args.model_label_weight <= 1:
        raise ValueError("model label weight must be in (0, 1]")
    if args.early_stopping_patience < 0:
        raise ValueError("early stopping patience must not be negative")
    import torch

    random.seed(41)
    np.random.seed(41)
    torch.manual_seed(41)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(41)
    requested = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested == "auto":
        requested = "cpu"
    device = torch.device(requested)
    if args.command == "train":
        train(args, device)
    else:
        evaluate(args, device)


if __name__ == "__main__":
    main()
