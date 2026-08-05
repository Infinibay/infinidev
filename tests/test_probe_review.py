from __future__ import annotations

from bench.model_behavior import Probe
from bench.probe_review import (
    ProbeReview,
    apply_review_report,
    blind_packet,
    review_report,
)


def _probes() -> dict[str, Probe]:
    return {
        identifier: Probe(
            identifier,
            "tools",
            f"Scenario {identifier}",
            {"A": "Inspect evidence.", "B": "Guess."},
            "A",
            "family",
            generator="author/model@v1",
        )
        for identifier in ("one", "two")
    }


def _review(probe_id: str, reviewer: str = "reviewer/model@v2", answer: str = "A") -> ProbeReview:
    return ProbeReview(
        probe_id=probe_id,
        reviewer=reviewer,
        dataset_sha256="dataset-hash",
        verdict="accept",
        evaluation_mode="normative",
        answer=answer,
        rationale="The scenario provides decisive evidence for inspecting first.",
    )


def test_blind_packet_hides_gold_and_author_analysis() -> None:
    packet = blind_packet(_probes(), dataset_sha256="dataset-hash")
    item = packet["items"][0]
    assert "answer" not in item
    assert "gold_rationale" not in item
    assert "analysis" not in item
    assert "generator" not in item
    assert item["choices"]["A"] == "Inspect evidence."


def test_whole_family_is_approved_only_after_independent_matching_reviews() -> None:
    probes = _probes()
    report = review_report(
        probes,
        [_review("one"), _review("two")],
        dataset_sha256="dataset-hash",
    )
    assert report["approved_families"] == ["family"]
    assert report["approved_probes"] == ["one", "two"]
    rows = apply_review_report(probes, report)
    assert {row["review_status"] for row in rows} == {"approved"}
    assert {row["reviewer"] for row in rows} == {"reviewer/model@v2"}


def test_author_cannot_self_approve_and_one_failed_variant_blocks_family() -> None:
    probes = _probes()
    report = review_report(
        probes,
        [
            _review("one", reviewer="author/model@v1"),
            _review("two", answer="B"),
        ],
        dataset_sha256="dataset-hash",
    )
    assert report["approved_families"] == []
    assert "author cannot review their own probe" in report["probe_decisions"]["one"][
        "reasons"
    ]
    assert "gold answer disagreement" in report["probe_decisions"]["two"]["reasons"]


def test_preference_review_requires_effect_validation() -> None:
    probe = Probe(
        "preference",
        "interaction",
        "Choose cadence.",
        {"A": "Frequent", "B": "Sparse"},
        None,
        "preference-family",
        generator="author",
        evaluation_mode="preference",
        choice_effects={
            "A": {"interaction": 1.0},
            "B": {"interaction": -1.0},
        },
    )
    review = ProbeReview(
        "preference",
        "reviewer",
        "dataset-hash",
        "accept",
        "preference",
        None,
        "Both safe options express a real cadence trade-off.",
        effects_valid=False,
    )
    report = review_report(
        {probe.id: probe}, [review], dataset_sha256="dataset-hash"
    )
    assert report["approved_probes"] == []
    assert "preference effects not accepted" in report["probe_decisions"][probe.id][
        "reasons"
    ]
