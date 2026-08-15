"""Optional runtime for fine-tuned multi-label task-policy encoders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any


logger = logging.getLogger(__name__)
METHOD_LABELS = (
    "bugfix.root_cause",
    "feature.contract_first",
    "refactor.preserve_behavior",
    "research.evidence_first",
    "review.read_only",
    "performance.measure_first",
)
QUERY_TOKENS = tuple(
    f"<|task_policy_{label.split('.', 1)[0]}|>" for label in METHOD_LABELS
)
SUPPORTED_RUN_VERSIONS = frozenset({"task-policy-fixed-encoder-natural-v2"})
SUPPORTED_ARCHITECTURES = frozenset({"query_tokens", "last"})


@dataclass(frozen=True)
class PolicyScore:
    """One independently calibrated sigmoid output."""

    policy_id: str
    score: float
    threshold: float
    selected: bool


@dataclass(frozen=True)
class EncoderTaskPrediction:
    """Auditable multi-label result from one optional encoder checkpoint."""

    scores: tuple[PolicyScore, ...] = ()
    task_score: float = 0.0
    task_threshold: float = 0.0
    classifier_version: str = ""
    space_id: str | None = None
    abstention_reason: str = ""

    @property
    def selected(self) -> tuple[PolicyScore, ...]:
        return tuple(item for item in self.scores if item.selected)


@dataclass(frozen=True)
class _CheckpointMetadata:
    architecture: str
    max_length: int
    thresholds: tuple[float, ...]
    task_threshold: float
    classifier_version: str
    space_id: str


@dataclass
class _LoadedEncoder:
    metadata: _CheckpointMetadata
    tokenizer: Any
    encoder: Any
    head: dict[str, Any]
    device: Any
    lock: threading.Lock


def _checkpoint_metadata(checkpoint: Path) -> _CheckpointMetadata:
    config_path = checkpoint / "task_policy_config.json"
    head_path = checkpoint / "head.safetensors"
    encoder_path = checkpoint / "encoder"
    if not config_path.is_file() or not head_path.is_file() or not encoder_path.is_dir():
        raise ValueError(
            "checkpoint requires task_policy_config.json, head.safetensors, and encoder/"
        )
    raw = config_path.read_bytes()
    config = json.loads(raw)
    if config.get("run_version") not in SUPPORTED_RUN_VERSIONS:
        raise ValueError("unsupported task-policy encoder run version")
    parameters = config.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("checkpoint parameters are missing")
    architecture = str(parameters.get("architecture", ""))
    if architecture not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"unsupported task-policy encoder architecture: {architecture}")
    max_length = int(parameters.get("max_length", 0))
    if max_length <= len(METHOD_LABELS):
        raise ValueError("checkpoint max_length is invalid")
    threshold_map = config.get("thresholds")
    if not isinstance(threshold_map, dict) or set(threshold_map) != set(METHOD_LABELS):
        raise ValueError("checkpoint thresholds do not match runtime labels")
    thresholds = tuple(float(threshold_map[label]) for label in METHOD_LABELS)
    task_threshold = float(config.get("task_threshold", -1.0))
    if any(not 0 <= value <= 1 for value in (*thresholds, task_threshold)):
        raise ValueError("checkpoint thresholds must be probabilities")
    digest = hashlib.sha256(raw + head_path.read_bytes()).hexdigest()[:16]
    version = f"fine-tuned-qwen-task-policy-{architecture}-{digest}"
    return _CheckpointMetadata(
        architecture=architecture,
        max_length=max_length,
        thresholds=thresholds,
        task_threshold=task_threshold,
        classifier_version=version,
        space_id=f"infinidev/task-policy-encoder:{digest}",
    )


def _resolve_device(torch: Any, requested: str) -> Any:
    value = requested.strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the task-policy encoder but is unavailable")
    return torch.device(value)


@lru_cache(maxsize=2)
def _load_encoder(checkpoint_value: str, requested_device: str) -> _LoadedEncoder:
    import torch
    from safetensors.torch import load_file
    from transformers import AutoModel, AutoTokenizer

    checkpoint = Path(checkpoint_value).expanduser().resolve()
    metadata = _checkpoint_metadata(checkpoint)
    device = _resolve_device(torch, requested_device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    encoder = AutoModel.from_pretrained(
        checkpoint / "encoder",
        dtype=dtype,
        local_files_only=True,
    ).to(device)
    encoder.config.use_cache = False
    encoder.eval()
    head = {
        name: tensor.to(device=device, dtype=dtype)
        for name, tensor in load_file(checkpoint / "head.safetensors").items()
    }
    required = {"task.weight", "task.bias"}
    if metadata.architecture == "query_tokens":
        required.update({"label_weights", "label_bias"})
        token_ids = tokenizer.convert_tokens_to_ids(list(QUERY_TOKENS))
        if any(value == tokenizer.unk_token_id for value in token_ids):
            raise ValueError("checkpoint tokenizer is missing task-policy query tokens")
    else:
        required.update({"methods.weight", "methods.bias"})
    if set(head) != required:
        raise ValueError(
            f"checkpoint head tensors are incompatible: {sorted(set(head) ^ required)}"
        )
    logger.info(
        "Loaded task-policy encoder %s on %s from %s",
        metadata.classifier_version,
        device,
        checkpoint,
    )
    return _LoadedEncoder(
        metadata=metadata,
        tokenizer=tokenizer,
        encoder=encoder,
        head=head,
        device=device,
        lock=threading.Lock(),
    )


def _mean_pool(hidden: Any, mask: Any) -> Any:
    expanded = mask[..., None].bool()
    return hidden.masked_fill(~expanded, 0.0).sum(dim=1) / expanded.sum(dim=1).clamp(min=1)


def _last_pool(hidden: Any, mask: Any) -> Any:
    import torch

    positions = torch.arange(mask.shape[1], device=mask.device)
    last = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
    if bool((last < 0).any()):
        raise ValueError("cannot classify an input with no valid tokens")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def _encoded(runtime: _LoadedEncoder, text: str) -> dict[str, Any]:
    import torch

    metadata = runtime.metadata
    length = metadata.max_length
    if metadata.architecture == "query_tokens":
        length -= len(METHOD_LABELS)
    batch = runtime.tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=length,
        return_tensors="pt",
    )
    if metadata.architecture == "query_tokens":
        query_ids = runtime.tokenizer.convert_tokens_to_ids(list(QUERY_TOKENS))
        queries = torch.tensor([query_ids], dtype=batch["input_ids"].dtype)
        batch["input_ids"] = torch.cat((batch["input_ids"], queries), dim=1)
        batch["attention_mask"] = torch.cat((
            batch["attention_mask"],
            torch.ones_like(queries),
        ), dim=1)
    return {name: value.to(runtime.device) for name, value in batch.items()}


def _logits(runtime: _LoadedEncoder, batch: dict[str, Any]) -> tuple[Any, Any]:
    outputs = runtime.encoder(**batch).last_hidden_state
    mask = batch["attention_mask"]
    if runtime.metadata.architecture == "query_tokens":
        query_hidden = outputs[:, -len(METHOD_LABELS):]
        method_logits = (
            query_hidden * runtime.head["label_weights"][None, ...]
        ).sum(dim=-1) + runtime.head["label_bias"]
        pooled = _mean_pool(
            outputs[:, :-len(METHOD_LABELS)],
            mask[:, :-len(METHOD_LABELS)],
        )
    else:
        pooled = _last_pool(outputs, mask)
        method_logits = (
            pooled @ runtime.head["methods.weight"].T + runtime.head["methods.bias"]
        )
    task_logits = pooled @ runtime.head["task.weight"].T + runtime.head["task.bias"]
    return method_logits, task_logits.squeeze(1)


def classify_task_methods(
    text: str,
    *,
    checkpoint: str,
    device: str = "auto",
) -> EncoderTaskPrediction:
    """Classify zero or more methods, soft-failing if the optional runtime is absent."""
    normalized = " ".join(text.split())
    if not normalized:
        return EncoderTaskPrediction(abstention_reason="request is empty")
    try:
        import torch

        runtime = _load_encoder(checkpoint, device)
        with runtime.lock, torch.inference_mode(), torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
            enabled=runtime.device.type == "cuda",
        ):
            method_logits, task_logits = _logits(runtime, _encoded(runtime, normalized))
            values = method_logits.sigmoid().float().cpu().numpy()[0]
            task_score = float(task_logits.sigmoid().float().cpu().item())
        task_selected = task_score >= runtime.metadata.task_threshold
        scores = tuple(
            PolicyScore(
                policy_id=label,
                score=float(score),
                threshold=threshold,
                selected=task_selected and float(score) >= threshold,
            )
            for label, score, threshold in zip(
                METHOD_LABELS,
                values,
                runtime.metadata.thresholds,
                strict=True,
            )
        )
        selected = any(item.selected for item in scores)
        return EncoderTaskPrediction(
            scores=scores,
            task_score=task_score,
            task_threshold=runtime.metadata.task_threshold,
            classifier_version=runtime.metadata.classifier_version,
            space_id=runtime.metadata.space_id,
            abstention_reason="" if selected else "no calibrated method passed",
        )
    except Exception as exc:
        logger.warning("Task-policy encoder classification failed: %s", exc)
        logger.debug("Task-policy encoder failure", exc_info=True)
        return EncoderTaskPrediction(abstention_reason=str(exc))


def clear_task_policy_encoder_cache() -> None:
    """Release cached model references, primarily for settings changes and tests."""
    _load_encoder.cache_clear()


__all__ = [
    "EncoderTaskPrediction",
    "METHOD_LABELS",
    "PolicyScore",
    "classify_task_methods",
    "clear_task_policy_encoder_cache",
]
