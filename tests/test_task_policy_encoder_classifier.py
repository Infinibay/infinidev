"""Tests for the optional fine-tuned task-policy encoder runtime."""

from __future__ import annotations

import json

import pytest

from infinidev.engine.task_policies.encoder_classifier import (
    METHOD_LABELS,
    _checkpoint_metadata,
)


def _checkpoint(tmp_path, *, architecture: str = "query_tokens"):
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "encoder").mkdir(parents=True)
    (checkpoint / "head.safetensors").write_bytes(b"test-head")
    (checkpoint / "task_policy_config.json").write_text(json.dumps({
        "run_version": "task-policy-fixed-encoder-natural-v2",
        "parameters": {"architecture": architecture, "max_length": 1024},
        "thresholds": {label: 0.5 for label in METHOD_LABELS},
        "task_threshold": 0.4,
    }))
    return checkpoint


def test_checkpoint_metadata_validates_label_order_independently_of_json_order(
    tmp_path,
) -> None:
    metadata = _checkpoint_metadata(_checkpoint(tmp_path))

    assert metadata.architecture == "query_tokens"
    assert metadata.max_length == 1024
    assert metadata.thresholds == (0.5,) * len(METHOD_LABELS)
    assert metadata.classifier_version.startswith("fine-tuned-qwen-task-policy-")
    assert metadata.space_id.startswith("infinidev/task-policy-encoder:")


def test_checkpoint_metadata_rejects_an_unsupported_architecture(tmp_path) -> None:
    with pytest.raises(ValueError, match="unsupported.*architecture"):
        _checkpoint_metadata(_checkpoint(tmp_path, architecture="mean"))


def test_checkpoint_metadata_rejects_missing_label_threshold(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path)
    path = checkpoint / "task_policy_config.json"
    config = json.loads(path.read_text())
    config["thresholds"].pop(METHOD_LABELS[-1])
    path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="thresholds.*labels"):
        _checkpoint_metadata(checkpoint)
