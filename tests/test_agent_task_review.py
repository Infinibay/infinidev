from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_task_review import (
    apply_review_report,
    build_review_report,
    export_packet,
    render_packet_markdown,
    render_review_template,
)


def _reviews(packet: dict[str, object]) -> list[dict[str, object]]:
    tasks = packet["tasks"]
    assert isinstance(tasks, list)
    return [
        {
            "dataset_sha256": packet["dataset_sha256"],
            "task_id": item["task"]["id"],
            "reviewer_identity": "independent-reviewer",
            "verdict": "approve",
            "rationale": "The verifier and rubric measure the stated task without prompt leakage.",
            "rubric_valid": True,
            "verifier_valid": True,
            "held_out_valid": True,
            "provider_neutral": True,
        }
        for item in tasks
    ]


def test_review_packet_is_candidate_blind_and_binds_fixtures() -> None:
    packet = export_packet(
        Path("bench/agent_task_pilot.tasks.jsonl"), Path("bench/agent_task_fixtures")
    )
    assert packet["candidate_blind"] is True
    assert len(packet["tasks"]) == 6
    rendered = json.dumps(packet)
    assert "system_prompt" not in rendered
    assert "candidate_guidance" not in rendered
    assert all(item["fixture_sha256"] for item in packet["tasks"])
    template = render_review_template(packet)
    assert len(template.splitlines()) == 6
    assert "REPLACE_WITH_approve_revise_or_reject" in template


def test_review_dossier_embeds_fixture_and_preflight_without_candidate_guidance() -> None:
    preflight = json.loads(Path("bench/agent_task_pilot.preflight.json").read_text())
    packet = export_packet(
        Path("bench/agent_task_pilot.tasks.jsonl"),
        Path("bench/agent_task_fixtures"),
        preflight,
    )
    markdown = render_packet_markdown(packet)
    assert "# Candidate-blind agent task review" in markdown
    assert "requirements.md" in markdown
    assert "Pristine exit: `1`" in markdown
    assert "Reference exit: `0`" in markdown
    assert "quality-control-verification" not in markdown
    assert "quality-control-explanation-depth" not in markdown


def test_complete_blind_review_can_materialize_approved_copy() -> None:
    tasks_path = Path("bench/agent_task_pilot.tasks.jsonl")
    packet = export_packet(tasks_path, Path("bench/agent_task_fixtures"))
    report = build_review_report(packet, _reviews(packet))
    source, approved = apply_review_report(tasks_path, report)
    assert source == tasks_path.read_text(encoding="utf-8")
    assert approved.count('"review_status":"approved"') == 6
    assert approved.count('"reviewer":"independent-reviewer"') == 6


def test_review_gate_rejects_incomplete_or_failed_reviews() -> None:
    tasks_path = Path("bench/agent_task_pilot.tasks.jsonl")
    packet = export_packet(tasks_path, Path("bench/agent_task_fixtures"))
    reviews = _reviews(packet)
    with pytest.raises(ValueError, match="incomplete"):
        build_review_report(packet, reviews[:-1])

    reviews[0]["verdict"] = "revise"
    report = build_review_report(packet, reviews)
    assert report["all_approved"] is False
    with pytest.raises(ValueError, match="not blind and fully approved"):
        apply_review_report(tasks_path, report)
