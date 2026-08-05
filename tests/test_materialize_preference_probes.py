from __future__ import annotations

from bench.materialize_preference_probes import materialize


def _seed() -> dict[str, object]:
    return {
        "generator": "manual/test@v1",
        "families": [
            {
                "family": "planning-depth",
                "category": "decomposition_and_planning",
                "scenarios": ["Facts in original order.", "Same facts in reverse order."],
                "user_requests": ["Choose for profile.", "Select for profile."],
                "actions": [
                    {"id": "one", "text": "One", "effects": {"speed": 1.0}, "rationale": "Fast."},
                    {"id": "two", "text": "Two", "effects": {"quality": 1.0}, "rationale": "Deep."},
                    {"id": "three", "text": "Three", "effects": {"speed": 0.4, "quality": 0.4}, "rationale": "Balanced."},
                    {"id": "four", "text": "Four", "effects": {"user_control": 1.0}, "rationale": "Controlled."},
                ],
                "gold_rationale": "All four actions satisfy hard requirements and differ only in a legitimate planning trade-off.",
                "hypothesis": "Uses the active profile.",
                "decisive_information": "All actions are safe and feasible.",
                "variant_axis": "Fact and option order.",
                "failure_signal": "Returns the same universal policy for every profile.",
                "calibration_use": "Test profile-conditioned planning guidance.",
                "preference_tradeoff": "Speed versus depth and control.",
            }
        ],
    }


def test_materialize_creates_isolated_reordered_variants() -> None:
    probes = materialize(_seed())
    assert [probe.id for probe in probes] == ["planning-depth-v1", "planning-depth-v2"]
    assert probes[0].group == probes[1].group == "planning-depth"
    assert probes[0].split == probes[1].split
    assert probes[0].choices["A"] == "One"
    assert probes[1].choices["A"] == "Three"
    assert probes[0].choice_effects["A"] == {"speed": 1.0}
    assert probes[1].choice_effects["A"] == {"speed": 0.4, "quality": 0.4}
    assert set(probes[1].analysis["choice_rationales"]) == set("ABCD")
