from __future__ import annotations

import pytest

from bench.model_behavior import Observation, Probe
from bench.profile_adaptation_report import build_profile_adaptation_report


def _probe() -> Probe:
    return Probe(
        "p1",
        "interaction",
        "Choose",
        {"A": "act now", "B": "ask first"},
        None,
        evaluation_mode="preference",
        choice_effects={"A": {"autonomy": 1.0}, "B": {"user_control": 1.0}},
    )


def _rows(answer: str, profile: str = "") -> list[Observation]:
    return [
        Observation(
            "p1",
            "raw",
            answer,
            None,
            repetition=repetition,
            model_identity="provider/model@v1",
            utility_profile=profile,
            utility_profile_sha256=(f"hash-{profile}" if profile else ""),
            elicitation_protocol="choice_only",
        )
        for repetition in range(3)
    ]


def test_report_preserves_actions_and_detects_replicated_profile_change() -> None:
    report = build_profile_adaptation_report(
        {"p1": _probe()},
        {
            "Model": {
                "raw": _rows("A"),
                "fast_autonomy": _rows("A", "fast-autonomy"),
                "quality_control": _rows("B", "quality-control"),
            }
        },
    )
    model = report["models"]["Model"]
    record = model["records"][0]
    assert model["fast_quality_modal_changes"] == 1
    assert record["conditions"]["fast_autonomy"]["modal_actions"] == ["act now"]
    assert record["conditions"]["quality_control"]["modal_actions"] == ["ask first"]


def test_report_rejects_mixed_option_order_protocols() -> None:
    raw = _rows("A")
    fast = _rows("A", "fast-autonomy")
    quality = _rows("B", "quality-control")
    fast[0] = Observation(
        **{
            **fast[0].__dict__,
            "option_order_protocol": "balanced_rotation",
        }
    )
    with pytest.raises(ValueError, match="mix option-order protocols"):
        build_profile_adaptation_report(
            {"p1": _probe()},
            {
                "Model": {
                    "raw": raw,
                    "fast_autonomy": fast,
                    "quality_control": quality,
                }
            },
        )
