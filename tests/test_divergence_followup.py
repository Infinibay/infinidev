from __future__ import annotations

from bench.divergence_followup import build_followup, render_markdown
from bench.model_behavior import Probe
from bench.probe_manifest import manifest_probe_ids


def _probe(probe_id: str, mode: str = "preference") -> Probe:
    return Probe(
        id=probe_id,
        category="interaction",
        prompt="Choose",
        choices={"A": "act", "B": "ask", "C": "compare", "D": "wait"},
        answer="A" if mode == "normative" else None,
        group="family-one",
        evaluation_mode=mode,
    )


def test_followup_selects_only_divergences_and_binds_dataset() -> None:
    probes = {"same": _probe("same"), "different": _probe("different", "normative")}
    comparison = {
        "models": {"Sol": {}, "Terra": {}, "Luna": {}},
        "questions": [
            {"probe_id": "same", "unanimous": True},
            {
                "probe_id": "different",
                "unanimous": False,
                "models": {
                    "Sol": {"selected_key": "A", "selected_action": "act", "raw_response": "A"},
                    "Terra": {"selected_key": "B", "selected_action": "ask", "raw_response": "B"},
                    "Luna": {"selected_key": "A", "selected_action": "act", "raw_response": "A"},
                },
            },
        ],
    }

    manifest, report = build_followup(probes, comparison, dataset_sha256="dataset")

    assert manifest_probe_ids(manifest, probes, dataset_sha256="dataset") == ["different"]
    assert report["probe_count"] == 1
    assert report["normative_count"] == 1
    assert report["complete_rotation_repetitions"] == 4
    assert report["calls_per_model"] == 4
    markdown = render_markdown(report, model_count=3)
    assert "12 total" in markdown
    assert "Sol" in markdown and "Terra" in markdown and "Luna" in markdown


def test_followup_rejects_unknown_probe() -> None:
    comparison = {
        "questions": [
            {
                "probe_id": "missing",
                "unanimous": False,
                "models": {"Sol": {}, "Terra": {}},
            }
        ]
    }

    try:
        build_followup({}, comparison, dataset_sha256="dataset")
    except ValueError as exc:
        assert "unknown probe" in str(exc)
    else:
        raise AssertionError("unknown probes must fail closed")
