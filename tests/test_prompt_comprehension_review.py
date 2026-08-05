from __future__ import annotations

from dataclasses import replace

import pytest

from bench.prompt_comprehension import COMPREHENSION_FIELDS, ComprehensionCase
from bench.prompt_comprehension_review import (
    REVIEW_CHECKS,
    FamilyReview,
    apply_adjudications,
    blind_packet,
    build_dossier,
    pilot_manifest,
    review_progress,
    review_template,
    select_pilot_cases,
    shard_packets,
)


def _case(identifier: str, relation: str = "anchor") -> ComprehensionCase:
    expected = {field: [] for field in COMPREHENSION_FIELDS}
    expected["objective"] = "Prepare a plan."
    expected["priority_resolution"] = ""
    return ComprehensionCase(
        id=identifier,
        category="planning",
        request=f"Prepare a plan ({identifier}).",
        split="calibration",
        review_status="draft",
        expected=expected,
        family_id="family",
        variant_id=identifier,
        intended_relation=relation,
        research_question_id="execution--planning--planning_before_action",
    )


def _review(cases: list[ComprehensionCase], verdict: str = "accept") -> FamilyReview:
    return FamilyReview(
        family_id="family",
        reviewer="independent-reviewer",
        dataset_sha256="hash",
        verdict=verdict,
        rationale="The variants are natural and isolate the intended difference.",
        checks={field: "pass" for field in REVIEW_CHECKS},
        reconstructions={case.id: dict(case.expected) for case in cases},
    )


def test_blind_packet_hides_authored_interpretation_keys() -> None:
    packet = blind_packet([_case("anchor"), _case("contrast", "contrast")], dataset_sha256="hash")
    assert all("expected" not in family and "authored_keys" not in family for family in packet["families"])
    assert packet["required_interpretation_fields"] == list(COMPREHENSION_FIELDS)


def test_dossier_requires_complete_reconstructions_and_never_auto_approves() -> None:
    cases = [_case("anchor"), _case("contrast", "contrast")]
    review = _review(cases)
    dossier = build_dossier(cases, [review], dataset_sha256="hash")
    assert dossier["ready_for_adjudication"] == ["family"]
    assert "approval" in dossier["approval_boundary"]
    assert "authored_keys" in dossier["families"][0]

    incomplete = replace(review, reconstructions={"anchor": review.reconstructions["anchor"]})
    with pytest.raises(ValueError, match="every family variant"):
        build_dossier(cases, [incomplete], dataset_sha256="hash")


def test_apply_needs_ready_family_and_explicit_adjudication() -> None:
    cases = [_case("anchor"), _case("contrast", "contrast")]
    dossier = build_dossier(cases, [_review(cases)], dataset_sha256="hash")
    adjudication = {
        "family_id": "family",
        "dataset_sha256": "hash",
        "decision": "approve",
        "adjudicator": "human-adjudicator",
        "rationale": "The independent reconstruction agrees with the authored key.",
    }
    rows = apply_adjudications(cases, dossier, [adjudication], dataset_sha256="hash")
    assert {row["review_status"] for row in rows} == {"approved"}
    assert {row["reviewer"] for row in rows} == {"human-adjudicator"}

    blocked = build_dossier(cases, [_review(cases, "revise")], dataset_sha256="hash")
    with pytest.raises(ValueError, match="not ready"):
        apply_adjudications(cases, blocked, [adjudication], dataset_sha256="hash")


def test_pilot_selects_one_family_of_each_kind_per_domain() -> None:
    cases = []
    for domain in ("planning", "implementation"):
        for kind in ("linguistic", "execution"):
            for candidate in ("a", "b"):
                family_id = f"{domain}-{kind}-{candidate}"
                for variant, relation in (("anchor", "anchor"), ("contrast", "contrast")):
                    case = _case(f"{family_id}-{variant}", relation)
                    cases.append(
                        replace(
                            case,
                            family_id=family_id,
                            category=domain,
                            research_question_id=f"{kind}--{candidate}",
                            stimulus_profile={
                                "domain": domain,
                                "study_kind": kind,
                                "phenomenon": candidate if kind == "linguistic" else "execution_policy",
                                "execution_dimension": candidate if kind == "execution" else "none",
                            },
                        )
                    )

    selected, selection = select_pilot_cases(cases, seed="fixed")
    assert len({case.family_id for case in selected}) == 4
    assert len(selected) == 8
    assert len(selection["selected_question_ids"]) == 4
    assert len(selection["selected_research_dimensions"]) == 4
    assert select_pilot_cases(cases, seed="fixed")[1] == selection

    packet = blind_packet(selected, dataset_sha256="hash")
    template = review_template(packet)
    assert len(template) == 4
    assert all(row["verdict"] == "revise" for row in template)
    assert all(set(row["checks"].values()) == {"fail"} for row in template)

    manifest = pilot_manifest(selected, selection, source_dataset_sha256="hash")
    assert manifest["counts"]["families"] == 4
    assert manifest["counts"]["cases"] == 8


def test_shards_preserve_every_family_and_progress_rejects_placeholders() -> None:
    cases = [_case(f"case-{index}") for index in range(4)]
    cases = [replace(case, family_id=f"family-{index}") for index, case in enumerate(cases)]
    packet = blind_packet(cases, dataset_sha256="hash")
    shards = shard_packets(packet, shard_count=2)
    assert [len(shard["families"]) for shard in shards] == [2, 2]
    assert {
        family["family_id"] for shard in shards for family in shard["families"]
    } == {f"family-{index}" for index in range(4)}

    template = review_template(shards[0])
    incomplete = review_progress(shards[0], template)
    assert incomplete["all_complete"] is False
    assert incomplete["counts"]["completed_families"] == 0

    completed = []
    for row in template:
        row["reviewer"] = "independent-reviewer"
        row["rationale"] = "The family isolates its intended semantic relationship."
        for reconstruction in row["reconstructions"].values():
            reconstruction["objective"] = "Prepare a plan."
        completed.append(row)
    progress = review_progress(shards[0], completed)
    assert progress["all_complete"] is True
    assert progress["counts"]["completed_families"] == 2
