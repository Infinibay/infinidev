from __future__ import annotations

from bench.model_behavior import Probe
from bench.run_model_behavior import Condition, RunConfig
from bench.run_probe_triage import (
    FamilyTriage,
    TRIAGE_PROTOCOL_VERSION,
    family_packet,
    render_triage_markdown,
    run_family_triage,
    triage_report,
    validate_triage_config,
)


def _family() -> list[Probe]:
    return [
        Probe(
            f"p{index}",
            "interaction",
            f"Scenario {index}",
            {"A": "Act", "B": "Ask"},
            None,
            "family",
            evaluation_mode="preference",
            choice_effects={"A": {"autonomy": 1.0}, "B": {"user_control": 1.0}},
            generator="author",
        )
        for index in (1, 2)
    ]


def test_family_packet_hides_author_fields_and_varies_action_order() -> None:
    first = family_packet("family", _family(), reviewer_identity="reviewer-one")
    second = family_packet("family", _family(), reviewer_identity="reviewer-two")
    assert all("generator" not in item for item in first["variants"])
    assert first["variants"][0]["actions"] != second["variants"][0]["actions"]
    assert first["triage_protocol_version"] == TRIAGE_PROTOCOL_VERSION
    assert "semantic-equivalence" in first["instrument_contract"]["variant_purpose"]
    assert "raw baseline deliberately omits" in first["instrument_contract"]["preference_delivery"]


def test_triage_is_diagnostic_and_report_denies_approval_authority() -> None:
    config = RunConfig("model", "reviewer@v1", (Condition("raw", None),))

    def completion(config: RunConfig, prompt: str) -> str:
        return (
            '{"verdict":"revise","issue_codes":["overlapping_options"],'
            '"affected_probe_ids":["p1"],"summary":"Two actions overlap.",'
            '"suggested_change":"Separate the actions."}'
        )

    row = run_family_triage("family", _family(), config, "dataset", completion)
    assert row.protocol_version == TRIAGE_PROTOCOL_VERSION
    assert row.verdict == "revise"
    report = triage_report([row, row])
    assert report["families"][0]["consensus_issues"] == ["overlapping_options"]
    assert "cannot approve" in report["authority_boundary"]
    markdown = render_triage_markdown(report)
    assert "Two actions overlap" in markdown
    assert "Separate the actions" in markdown
    assert "no numeric threshold approves" in markdown


def test_invalid_triage_is_retained_as_error_not_approval() -> None:
    config = RunConfig("model", "reviewer@v1", (Condition("raw", None),))
    row = run_family_triage(
        "family", _family(), config, "dataset", lambda config, prompt: "not json"
    )
    assert row.error and row.verdict == ""
    assert triage_report([row])["families"][0]["successful_reviews"] == 0


def test_report_rejects_mixed_triage_protocols() -> None:
    config = RunConfig("model", "reviewer@v1", (Condition("raw", None),))
    row = run_family_triage(
        "family",
        _family(),
        config,
        "dataset",
        lambda config, prompt: (
            '{"verdict":"pass","issue_codes":[],"affected_probe_ids":[],'
            '"summary":"Sound family.","suggested_change":""}'
        ),
    )
    legacy = FamilyTriage(**{**row.__dict__, "protocol_version": 1})
    try:
        triage_report([legacy, row])
    except ValueError as exc:
        assert "different protocol versions" in str(exc)
    else:
        raise AssertionError("mixed triage protocols were combined")


def test_triage_config_requires_raw_single_condition_and_two_second_pacing() -> None:
    validate_triage_config(
        RunConfig(
            "model",
            "reviewer@v1",
            (Condition("raw", None),),
            min_request_interval_seconds=2.0,
        )
    )
    invalid = [
        RunConfig(
            "model",
            "reviewer@v1",
            (Condition("raw", None),),
            min_request_interval_seconds=1.99,
        ),
        RunConfig(
            "model",
            "reviewer@v1",
            (Condition("guided", "behave"),),
        ),
        RunConfig(
            "model",
            "reviewer@v1",
            (Condition("one", None), Condition("two", None)),
        ),
    ]
    for config in invalid:
        try:
            validate_triage_config(config)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid triage config was accepted")
