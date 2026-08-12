"""Tests for WildChat natural-request candidate extraction."""

from __future__ import annotations

import hashlib
import json

import pytest

from bench.wildchat_candidate_sampler import (
    SOURCE_REVISION,
    candidate_from_row,
    exclusion_sets,
    first_user_utterance,
    select_candidates,
    selection_hint,
    write_candidate_queue,
)


def _row(
    conversation_id: str,
    text: str,
    *,
    language: str = "English",
    toxic: bool = False,
    redacted: bool = False,
) -> dict[str, object]:
    return {
        "conversation_id": conversation_id,
        "model": "gpt-4",
        "timestamp": "2024-01-01T00:00:00Z",
        "turn": 1,
        "language": language,
        "toxic": toxic,
        "redacted": redacted,
        "conversation": [
            {
                "role": "user",
                "content": text,
                "language": language,
                "toxic": toxic,
                "redacted": redacted,
            },
            {"role": "assistant", "content": "response"},
        ],
    }


def test_first_user_utterance_skips_empty_and_non_user_messages() -> None:
    conversation = [
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": " "},
        {"role": "user", "content": "Review this Python function."},
    ]

    assert first_user_utterance(conversation)["content"] == "Review this Python function."


def test_candidate_preserves_provenance_but_does_not_assign_policy() -> None:
    candidate = candidate_from_row(_row("one", "Please review this Python function for races."))

    assert candidate["candidate_id"] == "wildchat:one:0"
    assert candidate["source"]["conversation_id"] == "one"
    assert candidate["source"]["dataset_revision"] == SOURCE_REVISION
    assert candidate["source"]["upstream_selection_hint"] == "review_signal"
    assert candidate["manual_review"]["policies"] is None
    assert candidate["manual_review"]["status"] == "unreviewed"


@pytest.mark.parametrize(
    "row",
    [
        _row("toxic", "Write Python code for this task.", toxic=True),
        _row("redacted", "Debug this JavaScript error.", redacted=True),
        _row("unrelated", "Recommend a good soup recipe."),
    ],
)
def test_candidate_rejects_unsafe_or_non_programming_requests(row: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        candidate_from_row(row)


def test_selection_hints_are_coarse_acquisition_signals() -> None:
    assert selection_hint("Please audit this SQL query") == "review_signal"
    assert selection_hint("Compare Python queue libraries") == "research_signal"
    assert selection_hint("Why does this React code rerender?") == "question_signal"


def test_selection_is_deterministic_deduplicated_and_diverse() -> None:
    rows = [
        _row("review-en", "Please review this Python function.", language="English"),
        _row("review-es", "Review este código Python por favor.", language="Spanish"),
        _row("fix-en", "Fix the crash in this Python script.", language="English"),
        _row("fix-es", "Debug el error de este código Python.", language="Spanish"),
        _row("duplicate", "FIX THE CRASH IN THIS PYTHON SCRIPT.", language="English"),
    ]

    first, report = select_candidates(rows, limit=4, max_per_language=2, seed=19)
    second, _ = select_candidates(rows, limit=4, max_per_language=2, seed=19)

    assert first == second
    assert len(first) == 4
    assert report["selected_languages"] == {"english": 2, "spanish": 2}
    assert report["rejected"]["duplicate_text"] == 1


def test_selection_excludes_prior_conversations(tmp_path) -> None:
    prior = tmp_path / "prior.jsonl"
    prior.write_text(
        json.dumps(candidate_from_row(_row("old", "Review this Python code carefully."))) + "\n",
        encoding="utf-8",
    )
    candidate_ids, conversation_ids = exclusion_sets([prior])
    selected, report = select_candidates(
        [
            _row("old", "Fix this Python code now."),
            _row("fresh", "Fix this Python code instead."),
        ],
        limit=2,
        max_per_language=2,
        seed=19,
        excluded_candidate_ids=candidate_ids,
        excluded_conversation_ids=conversation_ids,
    )

    assert [item["candidate_id"] for item in selected] == ["wildchat:fresh:0"]
    assert report["rejected"]["excluded_conversation"] == 1


def test_write_candidate_queue_records_privacy_and_license_boundary(tmp_path) -> None:
    candidate = candidate_from_row(_row("one", "Review this Python function carefully."))
    output = tmp_path / "candidates.jsonl"

    manifest_path = write_candidate_queue(
        output,
        [candidate],
        {"selected": 1},
        scan_limit=10,
        selection_limit=1,
        max_per_language=5,
        seed=19,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifact"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest["source"]["revision"] == SOURCE_REVISION
    assert manifest["source"]["license"] == "ODC-BY-1.0"
    assert manifest["privacy_filters"]["conversation_redacted_must_be_false"] is True
    assert manifest["review_contract"]["selection_hint_is_policy_label"] is False
    assert manifest["review_contract"]["source_text_remains_external"] is True
    assert (
        manifest["review_contract"]["reviewed_annotations_license"]
        == "CC-BY-4.0 AND ODC-By-1.0"
    )
