from __future__ import annotations

import json

from bench.task_policy_synthetic_generation import (
    PROJECTS,
    accept_blind_agreements,
    build_specs,
    generate_corpus,
    messages_for_specs,
    parse_generated,
)


def test_build_specs_uses_24_projects_and_requested_support() -> None:
    specs = build_specs({"performance": 96, "review": 96}, seed=7)

    assert len(PROJECTS) == 24
    assert len(specs) == 192
    assert sum("performance" in row["policies"] for row in specs) >= 96
    assert sum("review" in row["policies"] for row in specs) >= 96
    assert len({row["project"] for row in specs}) == 24
    assert {row["natural_language"] for row in specs} >= {"English", "Spanish", "Portuguese"}
    assert {row["length"] for row in specs} == {
        "short: 25-60 words", "medium: 90-180 words", "long: 240-450 words",
    }


def test_build_specs_can_generate_only_single_label_boundaries() -> None:
    specs = build_specs(
        {"research": 30, "review": 30}, seed=9, single_label_only=True,
    )

    assert all(row["policies"] == [row["primary_policy"]] for row in specs)


def test_generation_prompt_does_not_request_labels_in_output() -> None:
    spec = build_specs({"refactor": 1}, seed=3)[0]
    messages = messages_for_specs([spec])
    parsed = parse_generated(
        '[{"candidate_id":"%s","issue_text":"Reorganize the parser while preserving outputs."}]'
        % spec["candidate_id"],
        [spec["candidate_id"]],
    )

    assert set(json.loads(messages[1]["content"])[0]) >= {"policies", "length", "style"}
    assert set(parsed[0]) == {"candidate_id", "issue_text"}


def test_generate_corpus_is_resumable(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    targets = tmp_path / "targets.jsonl"

    def request(messages: list[dict[str, str]]) -> tuple[str, dict]:
        briefs = json.loads(messages[1]["content"])
        rows = [
            {"candidate_id": row["candidate_id"], "issue_text": f"Unique request {row['scenario_nonce']}"}
            for row in briefs
        ]
        return json.dumps(rows), {"response_model": "fake", "response_id": "one"}

    first = generate_corpus(
        candidates, targets, targets={"research": 3}, seed=4,
        request=request, workers=2, batch_size=2, max_attempts=1,
    )
    second = generate_corpus(
        candidates, targets, targets={"research": 3}, seed=4,
        request=request, workers=2, batch_size=2, max_attempts=1,
    )

    assert first["generated"] == 3
    assert second["generated"] == 0
    assert second["already_completed"] == 3


def test_accept_requires_exact_blind_agreement_confidence_and_diversity(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    targets = tmp_path / "targets.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    accepted_candidates = tmp_path / "accepted-candidates.jsonl"
    accepted_reviews = tmp_path / "accepted-reviews.jsonl"
    candidate_rows = [
        {
            "candidate_id": f"c{index}", "issue_text": text,
            "source": {"repo": "synthetic-projects/a", "natural_language": "English", "programming_language": "Rust"},
        }
        for index, text in enumerate((
            "Measure representative latency and reduce the parser p95 below forty milliseconds.",
            "Measure representative latency and reduce the parser p95 below forty milliseconds please.",
            "Inspect the supplied parser patch and report correctness findings without editing files.",
            "Compare parser libraries and return an evidence-backed recommendation.",
        ))
    ]
    target_rows = [
        {"candidate_id": "c0", "primary_policy": "performance", "policies": ["performance"]},
        {"candidate_id": "c1", "primary_policy": "performance", "policies": ["performance"]},
        {"candidate_id": "c2", "primary_policy": "review", "policies": ["review"]},
        {"candidate_id": "c3", "primary_policy": "research", "policies": ["research"]},
    ]
    proposal_rows = [
        {"candidate_id": "c0", "proposal_status": "model_reviewed", "policies": ["performance"], "confidence": 0.95},
        {"candidate_id": "c1", "proposal_status": "model_reviewed", "policies": ["performance"], "confidence": 0.95},
        {"candidate_id": "c2", "proposal_status": "model_reviewed", "policies": ["research"], "confidence": 0.95},
        {"candidate_id": "c3", "proposal_status": "model_reviewed", "policies": ["research"], "confidence": 0.7},
    ]
    for path, rows in ((candidates, candidate_rows), (targets, target_rows), (proposals, proposal_rows)):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    report = accept_blind_agreements(
        candidates, targets, proposals, accepted_candidates, accepted_reviews,
        minimum_confidence=0.85, similarity_threshold=0.7,
    )

    assert report["accepted"] == 1
    assert report["rejection_reasons"] == {
        "label_disagreement": 1, "low_confidence": 1, "near_duplicate": 1,
    }
    review = json.loads(accepted_reviews.read_text(encoding="utf-8").splitlines()[0])
    assert review["annotation"]["kind"] == "model"
    assert review["annotation"]["provenance"] == "synthetic_generation_plus_blind_model_agreement"
