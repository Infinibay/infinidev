from __future__ import annotations

import json

import pytest

from bench.task_policy_minimax_proposals import (
    BATCH_PROMPT_VERSION,
    classify_batch_with_split,
    classify_candidate,
    generate_proposals,
    load_candidates,
    messages_for_batch,
    parse_batch_decisions,
)


def _candidate(candidate_id: str, issue_text: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "issue_text": issue_text,
        "source": {"repo": "owner/repo", "programming_language": "Python"},
    }


def test_load_candidates_rejects_duplicate_ids(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    row = _candidate("same", "Fix the documented behavior.")
    first.write_text(json.dumps(row) + "\n", encoding="utf-8")
    second.write_text(json.dumps(row) + "\n", encoding="utf-8")

    try:
        load_candidates([first, second])
    except ValueError as exc:
        assert "duplicate candidate" in str(exc)
    else:
        raise AssertionError("duplicate candidate was accepted")


def test_load_candidates_preserves_unicode_line_separator_inside_json(tmp_path) -> None:
    source = tmp_path / "unicode.jsonl"
    row = _candidate("unicode", "First paragraph.\u2028Second paragraph.")
    source.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    loaded = load_candidates([source])

    assert loaded[0]["issue_text"] == row["issue_text"]


def test_classify_candidate_emits_machine_review_metadata() -> None:
    row = {
        "candidate_id": "one",
        "issue_text": "The documented endpoint returns stale state; fix it.",
        "source_path": "queue.jsonl",
        "source_index": 0,
    }

    def request(messages: list[dict[str, str]]) -> tuple[str, dict]:
        assert "upstream_category_hint" not in json.dumps(messages)
        return (
            '{"policies":["bugfix"],"uncategorized_reason":null,'
            '"confidence":0.96,"rationale":"It restores documented behavior."}',
            {"response_id": "response-1", "response_model": "MiniMax-M3"},
        )

    result = classify_candidate(row, request=request, model="MiniMax-M3", max_attempts=1)

    assert result["proposal_status"] == "model_reviewed"
    assert result["reviewer_kind"] == "model"
    assert result["policies"] == ["bugfix"]
    assert result["notes"] == "It restores documented behavior."
    assert "issue_text" not in result


def test_batch_prompt_and_parser_preserve_one_decision_per_candidate() -> None:
    rows = [
        {"candidate_id": "one", "issue_text": "Fix stale results."},
        {"candidate_id": "two", "issue_text": "Add an export command."},
    ]
    messages = messages_for_batch(rows)
    parsed = parse_batch_decisions(
        json.dumps([
            {
                "candidate_id": "one",
                "policies": ["bugfix"],
                "uncategorized_reason": None,
                "confidence": 0.9,
                "rationale": "It restores existing behavior.",
            },
            {
                "candidate_id": "two",
                "policies": ["feature"],
                "uncategorized_reason": None,
                "confidence": 0.8,
                "rationale": "It adds a new capability.",
            },
        ]),
        ["one", "two"],
    )

    assert "upstream_category_hint" not in json.dumps(messages)
    assert [candidate_id for candidate_id, _decision in parsed] == ["one", "two"]
    assert parsed[1][1].policies == ("feature",)


def test_batch_parser_recovers_singleton_id_but_rejects_multirow_id_changes() -> None:
    decision = {
        "candidate_id": "model-typo",
        "policies": ["bugfix"],
        "uncategorized_reason": None,
        "confidence": 0.9,
        "rationale": "It restores existing behavior.",
    }

    recovered = parse_batch_decisions(json.dumps([decision]), ["source-id"])
    assert recovered[0][0] == "source-id"

    second = {**decision, "candidate_id": "second"}
    with pytest.raises(ValueError, match="candidate IDs or order"):
        parse_batch_decisions(json.dumps([decision, second]), ["first", "second"])


def test_invalid_batch_is_split_without_losing_valid_rows() -> None:
    rows = [
        {
            "candidate_id": "one",
            "issue_text": "Fix stale results.",
            "source_path": "q",
            "source_index": 0,
        },
        {
            "candidate_id": "two",
            "issue_text": "Add export.",
            "source_path": "q",
            "source_index": 1,
        },
    ]

    def request(messages: list[dict[str, str]]) -> tuple[str, dict]:
        requested = json.loads(messages[-1]["content"])
        if len(requested) > 1:
            return "invalid", {}
        item = requested[0]
        return json.dumps([{
            "candidate_id": item["candidate_id"],
            "policies": ["bugfix" if item["candidate_id"] == "one" else "feature"],
            "uncategorized_reason": None,
            "confidence": 0.9,
            "rationale": "The contract determines the category.",
        }]), {}

    proposals, failed = classify_batch_with_split(
        rows, request=request, model="MiniMax-M3", max_attempts=1
    )

    assert failed == 0
    assert [row["candidate_id"] for row in proposals] == ["one", "two"]
    assert all(row["batch_size"] == 1 for row in proposals)


def test_generate_proposals_is_resumable_and_leaves_failures_pending(tmp_path) -> None:
    source = tmp_path / "queue.jsonl"
    output = tmp_path / "proposals.jsonl"
    source.write_text(
        "\n".join([
            json.dumps(_candidate("one", "Add a new export command.")),
            json.dumps(_candidate("two", "Update only the installation guide.")),
        ]) + "\n",
        encoding="utf-8",
    )
    output.write_text(json.dumps({
        "candidate_id": "one",
        "prompt_version": BATCH_PROMPT_VERSION,
    }) + "\n", encoding="utf-8")

    def invalid_request(_messages: list[dict[str, str]]) -> tuple[str, dict]:
        return "not-json", {}

    report = generate_proposals(
        [source],
        output,
        request=invalid_request,
        model="MiniMax-M3",
        workers=1,
        batch_size=2,
        max_attempts=1,
        limit=None,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report == {
        "source": 2,
        "already_completed": 1,
        "attempted": 1,
        "reviewed": 0,
        "failed": 1,
    }
    assert rows == [{"candidate_id": "one", "prompt_version": BATCH_PROMPT_VERSION}]
