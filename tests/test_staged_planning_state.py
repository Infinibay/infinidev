"""Regression tests for durable staged-planning state."""

from __future__ import annotations

import pytest

from infinidev.engine.analysis.staged_planning import (
    EvidenceEntry,
    GoalSpec,
    StagedPlanningState,
)


def _state() -> StagedPlanningState:
    return StagedPlanningState(goal=GoalSpec(
        title="Test evidence ledger",
        user_request="Test evidence ledger provenance.",
    ))


def test_equal_summaries_from_different_tasks_do_not_collide() -> None:
    state = _state()
    first = EvidenceEntry(
        kind="task_result",
        summary="done",
        stage_id="stage-1",
        task_id="first",
        details={"task_status": "completed"},
    )
    second = EvidenceEntry(
        kind="task_result",
        summary="done",
        stage_id="stage-1",
        task_id="second",
        details={"task_status": "completed"},
    )

    assert state.add_evidence(first) is True
    assert state.add_evidence(second) is True
    assert [entry.task_id for entry in state.evidence] == ["first", "second"]


def test_retry_status_change_produces_new_evidence() -> None:
    state = _state()
    pending = EvidenceEntry(
        kind="task_result",
        summary="same output",
        stage_id="stage-1",
        task_id="task",
        details={"task_status": "pending"},
    )
    completed = pending.model_copy(update={
        "id": "completed-evidence",
        "details": {"task_status": "completed"},
    })

    assert state.add_evidence(pending) is True
    assert state.add_evidence(completed) is True


def test_identical_evidence_is_still_deduplicated() -> None:
    state = _state()
    first = EvidenceEntry(kind="observation", summary="same fact")
    duplicate = first.model_copy(update={"id": "another-id"})

    assert state.add_evidence(first) is True
    assert state.add_evidence(duplicate) is False
    assert state.evidence == [first]

def test_conflicting_evidence_id_is_rejected() -> None:
    state = _state()
    first = EvidenceEntry(id="evidence-fixed", kind="observation", summary="first")
    conflicting = EvidenceEntry(
        id="evidence-fixed", kind="observation", summary="different"
    )
    state.add_evidence(first)

    with pytest.raises(ValueError, match="evidence id"):
        state.add_evidence(conflicting)
