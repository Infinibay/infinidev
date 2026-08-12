"""Runtime tests for the packaged task-policy mini-head."""

from __future__ import annotations

import numpy as np

from infinidev.engine.task_policies.linear_classifier import (
    CLASSIFIER_VERSION,
    METHOD_LABELS,
    _load_head,
    classify_task_method,
)
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


def test_packaged_head_matches_runtime_embedding_space() -> None:
    embedder = get_static_qwen3_embedder()
    assert embedder is not None

    head = _load_head()

    assert head.space_id == embedder.space_id
    assert head.discourse_weights.shape == (embedder.dim + 1, 2)
    assert head.method_weights.shape == (embedder.dim + 1, len(METHOD_LABELS))
    assert np.all(np.isfinite(head.discourse_weights))
    assert np.all(np.isfinite(head.method_weights))


def test_classifier_selects_clear_multilingual_methods() -> None:
    cases = {
        "Simplifica el interior del parser sin cambiar ninguna salida.":
            "refactor.preserve_behavior",
        "Audit the permission broker and report findings without editing files.":
            "review.read_only",
        "Measure the callback latency under load and then make it faster.":
            "performance.measure_first",
    }

    for text, expected in cases.items():
        result = classify_task_method(text)

        assert result.policy_id == expected, text
        assert result.classifier_version == CLASSIFIER_VERSION
        assert result.space_id and result.space_id.startswith(
            "ken/static-qwen3-r512-v2:1024:"
        )


def test_classifier_learns_uncategorized_instead_of_forcing_a_method() -> None:
    result = classify_task_method(
        'El log del parser dice "corrige este error"; explica la frase.'
    )

    assert result.policy_id is None
    assert result.abstention_reason


def test_discourse_gate_rejects_action_words_in_non_task_messages() -> None:
    cases = (
        'The ticket title is "make retries faster"; translate the title only.',
        "Perfecto, ya no necesito que cambies el temporizador.",
        "Explain how retry budgets are normally defined.",
    )

    for text in cases:
        result = classify_task_method(text)

        assert result.policy_id is None, text


def test_hard_bugfix_does_not_become_performance_or_feature() -> None:
    cases = (
        "The lease renewer waits after its last allowed attempt; make it stop at the limit.",
        "El debounce pierde la última edición cuando vence el timer; recupera la garantía existente.",
    )

    for text in cases:
        result = classify_task_method(text)

        assert result.policy_id in {None, "bugfix.root_cause"}, text
