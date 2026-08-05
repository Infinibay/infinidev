from __future__ import annotations

from bench.counterbalanced_prompt_brief import build_prompt_brief
from bench.model_behavior import Probe, UtilityProfile


def _analysis(*, stable: bool, modes: list[str]) -> dict[str, object]:
    return {
        "models": {"Sol": {}},
        "records": [
            {
                "probe_id": "p",
                "category": "planning",
                "scenario": "Choose",
                "models": {
                    "Sol": {
                        "balanced_modal_keys": modes,
                        "balanced_modal_actions": ["act" if key == "A" else "ask" for key in modes],
                        "balanced_counts": {key: 4 if stable else 2 for key in modes},
                        "exactly_stable": stable,
                        "fixed_to_balanced_relation": "same_unique",
                    }
                },
            }
        ],
    }


def _probe() -> Probe:
    return Probe(
        "p",
        "planning",
        "Choose",
        {"A": "act", "B": "ask"},
        None,
        evaluation_mode="preference",
        choice_effects={"A": {"autonomy": 1.0}, "B": {"autonomy": -1.0}},
    )


def test_stable_raw_action_is_routed_by_explicit_profile() -> None:
    high_autonomy = UtilityProfile("autonomy", {"autonomy": 1.0}, "Act autonomously")
    aligned = build_prompt_brief(
        _analysis(stable=True, modes=["A"]), {"p": _probe()}, model="Sol", profile=high_autonomy
    )
    assert len(aligned["stable_profile_aligned_actions_to_preserve"]) == 1
    assert aligned["stable_profile_conflicts_to_test"] == []

    high_control = UtilityProfile("control", {"autonomy": -1.0}, "Ask before acting")
    conflict = build_prompt_brief(
        _analysis(stable=True, modes=["A"]), {"p": _probe()}, model="Sol", profile=high_control
    )
    assert len(conflict["stable_profile_conflicts_to_test"]) == 1


def test_unstable_prior_never_becomes_stable_candidate() -> None:
    profile = UtilityProfile("autonomy", {"autonomy": 1.0}, "Act autonomously")
    brief = build_prompt_brief(
        _analysis(stable=False, modes=["A", "B"]),
        {"p": _probe()},
        model="Sol",
        profile=profile,
    )
    assert brief["stable_profile_aligned_actions_to_preserve"] == []
    assert brief["stable_profile_conflicts_to_test"] == []
    assert len(brief["unstable_profile_hypotheses"]) == 1
