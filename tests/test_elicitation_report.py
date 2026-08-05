from __future__ import annotations

import json

from bench.elicitation_report import build_elicitation_report, render_markdown
from bench.model_behavior import Observation, Probe


def _write(path, row: Observation) -> None:
    path.write_text(json.dumps(row.__dict__) + "\n", encoding="utf-8")


def test_report_preserves_changed_actions_criterion_and_context(tmp_path) -> None:
    choice_path = tmp_path / "choice.jsonl"
    report_path = tmp_path / "report.jsonl"
    common = {
        "model_identity": "provider/model@v1",
        "condition_sha256": "raw-hash",
    }
    _write(
        choice_path,
        Observation(
            "p1", "raw", "A", None, response_text='{"answer":"A"}',
            elicitation_protocol="choice_only", **common,
        ),
    )
    _write(
        report_path,
        Observation(
            "p1", "raw", "B", 0.7, response_text='{"answer":"B"}',
            decision_criterion="Prefer verification", missing_context="test cost",
            elicitation_protocol="self_report", **common,
        ),
    )
    probes = {
        "p1": Probe(
            "p1", "verification", "What next?",
            {"A": "Move now.", "B": "Verify first."}, None,
            evaluation_mode="preference",
            choice_effects={"A": {"speed": 1.0}, "B": {"quality": 1.0}},
        )
    }

    result = build_elicitation_report(
        probes, {"Sol": (choice_path, report_path)}
    )
    model = result["models"]["Sol"]
    assert model["changed_choices"] == 1
    assert model["records"][0]["choice_only_action"] == "Move now."
    assert model["records"][0]["self_report_action"] == "Verify first."
    rendered = render_markdown(result)
    assert "Prefer verification" in rendered
    assert "test cost" in rendered
    assert "not privileged access to private reasoning" in rendered
