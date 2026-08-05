from __future__ import annotations

from bench.model_behavior import Observation, Probe
import json

import pytest

from bench.option_order_report import _load_manifest_probe_ids, build_option_order_report


def test_option_report_compares_canonical_actions_not_provider_letters() -> None:
    probe = Probe("p", "tools", "Choose", {"A": "read", "B": "guess"}, "A")
    fixed = [
        Observation(
            "p",
            "raw",
            "A",
            None,
            repetition=index,
            model_identity="model@v1",
            option_order_protocol="fixed",
            provider_answer="A",
        )
        for index in range(2)
    ]
    balanced = [
        Observation(
            "p",
            "raw",
            "A",
            None,
            repetition=0,
            model_identity="model@v1",
            option_order_protocol="balanced_rotation",
            provider_answer="B",
            choice_mapping={"A": "B", "B": "A"},
        ),
        Observation(
            "p",
            "raw",
            "A",
            None,
            repetition=1,
            model_identity="model@v1",
            option_order_protocol="balanced_rotation",
            provider_answer="A",
            choice_mapping={"A": "A", "B": "B"},
        ),
    ]
    report = build_option_order_report(
        {"p": probe}, {"Model": {"fixed": fixed, "balanced": balanced}}
    )
    record = report["models"]["Model"]["records"][0]
    assert record["modal_relation"] == "same_unique"
    assert record["balanced"]["provider_letter_counts"] == {"A": 1, "B": 1}
    assert record["balanced"]["modal_actions"] == ["read"]


def test_manifest_probe_ids_are_loaded_and_validated(tmp_path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"probes": [{"probe_id": "p2"}, {"probe_id": "p1"}]}),
        encoding="utf-8",
    )
    assert _load_manifest_probe_ids(manifest) == {"p1", "p2"}

    manifest.write_text(
        json.dumps({"probes": [{"probe_id": "p1"}, {"probe_id": "p1"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty and unique"):
        _load_manifest_probe_ids(manifest)
