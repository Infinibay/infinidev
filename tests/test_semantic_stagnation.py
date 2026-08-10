"""Semantic stagnation is subordinate to deterministic progress evidence."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.semantic_stagnation import (
    SemanticStagnation,
    detect_semantic_stagnation,
)
from infinidev.engine.loop.step_manager import StepManager


def _record(
    summary: str,
    *,
    step: int = 2,
    edits: int = 0,
    net_change: bool = False,
    tests: tuple[str, ...] = (),
) -> ActionRecord:
    return ActionRecord(
        step_index=step,
        summary=summary,
        successful_edit_count=edits,
        net_workspace_changed=net_change,
        test_outcome_fingerprints=tests,
    )


def test_three_similar_same_step_records_trigger_when_hard_evidence_is_static(
    monkeypatch,
) -> None:
    vectors = np.asarray([
        [1.0, 0.0],
        [0.9, np.sqrt(1.0 - 0.9 ** 2)],
        [0.82, np.sqrt(1.0 - 0.82 ** 2)],
    ], dtype=np.float32)
    monkeypatch.setattr(
        "infinidev.tools.base.embeddings.embed_passages",
        lambda texts: list(vectors),
    )
    history = [
        _record("Investigated the same implementation visitor and its call handling."),
        _record("Reviewed the same implementation visitor and call handling again."),
        _record("Inspected that implementation visitor and its call handling once more."),
    ]

    signal = detect_semantic_stagnation(
        history, minimum_cosine=0.75, strong_cosine=0.95,
    )

    assert signal is not None
    assert signal.step_index == 2


def test_two_near_duplicate_records_trigger_at_strong_threshold(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.tools.base.embeddings.embed_passages",
        lambda texts: [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.91, np.sqrt(1.0 - 0.91 ** 2)], dtype=np.float32),
        ],
    )
    summary = "Inspected the same implementation visitor and call handling once more."

    signal = detect_semantic_stagnation([_record(summary), _record(summary)])

    assert signal is not None
    assert signal.similarities == pytest.approx((0.91,))


def test_two_merely_similar_records_do_not_trigger(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.tools.base.embeddings.embed_passages",
        lambda texts: [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.85, np.sqrt(1.0 - 0.85 ** 2)], dtype=np.float32),
        ],
    )
    summary = "Inspected the same implementation visitor and call handling once more."

    assert detect_semantic_stagnation([_record(summary), _record(summary)]) is None


def test_two_similar_same_step_records_trigger_after_unrelated_prior_step(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.tools.base.embeddings.embed_passages",
        lambda texts: [
            np.asarray([1.0, 0.0], dtype=np.float32),
            np.asarray([0.95, np.sqrt(1.0 - 0.95 ** 2)], dtype=np.float32),
        ],
    )

    signal = detect_semantic_stagnation([
        _record("Completed an earlier discovery phase with useful evidence.", step=1),
        _record("Inspected the implementation without changing it.", step=2),
        _record("The budget ended while another inspection was requested.", step=2),
    ])

    assert signal is not None
    assert signal.step_index == 2


def test_similarity_abstains_after_edit_new_test_or_step_transition(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.tools.base.embeddings.embed_passages",
        lambda texts: [np.asarray([1.0, 0.0], dtype=np.float32)] * 3,
    )
    base = "Investigated the implementation visitor and its call handling in detail."

    assert detect_semantic_stagnation([
        _record(base), _record(base, edits=1, net_change=True), _record(base)
    ]) is None
    assert detect_semantic_stagnation([
        _record(base, tests=("1 failed",)),
        _record(base, tests=("1 failed",)),
        _record(base, tests=("2 passed",)),
    ]) is None
    assert detect_semantic_stagnation([
        _record(base, step=1), _record(base, step=1), _record(base, step=2)
    ]) is None


def test_reverted_edit_calls_do_not_mask_semantic_stagnation(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.tools.base.embeddings.embed_passages",
        lambda texts: [np.asarray([1.0, 0.0], dtype=np.float32)] * len(texts),
    )
    summary = "Inspected the same implementation and restored every attempted edit."

    signal = detect_semantic_stagnation([
        _record(summary, edits=2),
        _record(summary, edits=2),
    ])

    assert signal is not None


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Implement visitor handling", 1),
        ("Add regression tests for visitor handling", 1),
        ("Inspect visitor handling", 0),
        ("Design visitor handling", 0),
        ("Run visitor tests", 0),
    ],
)
def test_step_manager_only_arms_action_phases(monkeypatch, title, expected) -> None:
    monkeypatch.setattr(
        "infinidev.engine.loop.semantic_stagnation.detect_semantic_stagnation",
        lambda history: SemanticStagnation(2, (0.93,)),
    )
    monkeypatch.setattr("infinidev.engine.loop.step_manager._emit_log", lambda *a, **k: None)
    state = SimpleNamespace(
        history=[_record("A sufficiently long repeated implementation summary.")],
        plan=LoopPlan(steps=[PlanStep(index=2, title=title, status="active")]),
        discovery_suppression_steps=0,
    )
    ctx = SimpleNamespace(
        semantic_stagnation_control=True,
        state=state,
        project_id="project",
        agent_id="agent",
    )

    StepManager._arm_semantic_stagnation_control(ctx)

    assert state.discovery_suppression_steps == expected


def test_step_manager_policy_gate_prevents_embedding_call(monkeypatch) -> None:
    called = False

    def detect(history):
        nonlocal called
        called = True

    monkeypatch.setattr(
        "infinidev.engine.loop.semantic_stagnation.detect_semantic_stagnation", detect,
    )
    ctx = SimpleNamespace(semantic_stagnation_control=False)

    StepManager._arm_semantic_stagnation_control(ctx)

    assert called is False
