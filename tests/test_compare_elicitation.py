from __future__ import annotations

import pytest

from bench.compare_elicitation import compare_protocols
from bench.model_behavior import Observation


def _row(protocol: str, answer: str, **kwargs: object) -> Observation:
    return Observation(
        "p1",
        "raw",
        answer,
        kwargs.pop("confidence", None),
        repetition=0,
        model_identity=str(kwargs.pop("model_identity", "provider/model@v1")),
        condition_sha256="raw-hash",
        elicitation_protocol=protocol,
        **kwargs,
    )


def test_compare_protocols_retains_changed_answer_and_self_report() -> None:
    choice = _row("choice_only", "A", response_text='{"answer":"A"}')
    report = _row(
        "self_report",
        "B",
        confidence=0.7,
        decision_criterion="Prefer verification",
        missing_context="test cost",
        response_text='{"answer":"B"}',
    )
    comparison = compare_protocols([choice], [report])
    assert comparison["answer_agreement"] == 0.0
    assert comparison["records"][0]["choice_only_answer"] == "A"
    assert comparison["records"][0]["self_report_answer"] == "B"
    assert comparison["records"][0]["expressed_decision_criterion"] == "Prefer verification"


def test_compare_protocols_rejects_different_model_revision() -> None:
    with pytest.raises(ValueError, match="model identity mismatch"):
        compare_protocols(
            [_row("choice_only", "A")],
            [_row("self_report", "A", model_identity="provider/model@v2")],
        )


def test_compare_protocols_rejects_wrong_input_protocol() -> None:
    with pytest.raises(ValueError, match="expected only choice_only"):
        compare_protocols([_row("self_report", "A")], [])
