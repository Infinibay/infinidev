from __future__ import annotations

from bench.model_behavior import Observation, Probe
from bench.repetition_stability_report import build_stability_report, render_markdown


def _probe() -> Probe:
    return Probe(
        "p1", "interaction", "Choose.",
        {"A": "Act now.", "B": "Ask first."}, None,
        evaluation_mode="preference",
        choice_effects={"A": {"speed": 1.0}, "B": {"user_control": 1.0}},
    )


def test_stability_report_retains_each_action_and_marks_instability() -> None:
    probes = {"p1": _probe()}
    report = build_stability_report(
        probes,
        {
            "Sol": [
                Observation("p1", "raw", "A", None, repetition=0),
                Observation("p1", "raw", "A", None, repetition=1),
                Observation("p1", "raw", "B", None, repetition=2),
            ],
            "Terra": [
                Observation("p1", "raw", "A", None, repetition=0),
                Observation("p1", "raw", "A", None, repetition=1),
                Observation("p1", "raw", "A", None, repetition=2),
            ],
        },
    )

    sol = report["models"]["Sol"]
    assert "Repeated isolated choices" in report["interpretation_boundary"]
    assert sol["unstable_probes"] == 1
    assert sol["records"][0]["answer_counts"] == {"A": 2, "B": 1}
    assert sol["records"][0]["repetitions"][2]["selected_action"] == "Ask first."
    assert report["cross_model_modal_agreements"] == 1
    rendered = render_markdown(report)
    assert "p1` — unstable" in rendered
    assert "Repetition 2: **B** — Ask first." in rendered
