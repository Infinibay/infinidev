"""Tests for safe sequential collection of model-authored harness feedback."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench.harness_feedback import FeedbackCase
from bench.run_harness_feedback import (
    FeedbackReply,
    FeedbackRunConfig,
    pending_runs,
    run_sequentially,
)


def _case() -> FeedbackCase:
    return FeedbackCase.from_dict(
        {
            "id": "f1",
            "category": "tool_interface",
            "scenario": "a tool failed",
            "visible_artifact": "the schema error named field x",
            "question": "What should be tested?",
            "review_status": "approved",
        }
    )


def _config(interval: float = 2.0) -> FeedbackRunConfig:
    return FeedbackRunConfig(
        model="model",
        model_identity="provider/model@revision",
        min_request_interval_seconds=interval,
    )


def _reply() -> FeedbackReply:
    return FeedbackReply(
        text=(
            '{"no_change_warranted":false,"assessment":"The error is actionable but verbose.",'
            '"friction":"Recovery needs the full schema.","evidence":"Two failed shapes.",'
            '"suggested_change":"Show one minimal valid call.","expected_effect":"Fewer recovery '
            'calls.","risk":"The example may overfit one operation.","experiment":"Pair errors '
            'with and without examples on held-out invalid calls."}'
        )
    )


def test_config_rejects_interval_below_two_seconds() -> None:
    with pytest.raises(ValueError, match="at least 2.0"):
        FeedbackRunConfig.from_dict(
            {"model": "m", "model_identity": "id", "min_request_interval_seconds": 1.99}
        )


def test_runner_is_sequential_paced_and_records_each_result() -> None:
    starts = iter((0.0, 0.5, 2.0, 2.5))
    sleeps = []
    rows = []
    concurrent = 0

    def completion(config, case):
        nonlocal concurrent
        concurrent += 1
        assert concurrent == 1
        reply = _reply()
        concurrent -= 1
        return reply

    count = run_sequentially(
        [(_case(), 0), (_case(), 1)],
        _config(),
        completion,
        rows.append,
        dataset_sha256="dataset",
        monotonic=lambda: next(starts),
        sleep=sleeps.append,
    )

    assert count == 2
    assert sleeps == [0.5]
    assert [row.repetition for row in rows] == [0, 1]
    assert all(row.feedback is not None for row in rows)


def test_pending_runs_are_bound_to_case_hash_and_model_identity() -> None:
    case = _case()
    assert pending_runs([case], 2, [], model_identity="model-a") == [(case, 0), (case, 1)]


def test_direct_cli_help_works() -> None:
    assert Path("bench/run_harness_feedback.py").is_file()
