"""Tests for near-duplicate-family-disjoint external data splits."""

from __future__ import annotations

from bench.external_candidate_family_split import (
    candidate_families,
    split_development_queue_reserve,
)


def _candidate(identifier: str, text: str, source_group: str | None = None) -> dict[str, object]:
    return {
        "candidate_id": identifier,
        "source": {"repo": source_group or identifier},
        "issue_text": text,
    }


def _ids(rows: list[dict[str, object]]) -> set[str]:
    return {str(row["candidate_id"]) for row in rows}


def test_candidate_families_groups_long_prompt_variants_and_short_templates() -> None:
    shared_code = " ".join(f"token{index}" for index in range(80))
    rows = [
        _candidate("long-a", f"Please fix this code {shared_code} ending alpha"),
        _candidate("long-b", f"Can you repair this code {shared_code} ending beta"),
        _candidate("sport-a", "write a script about alpha versus beta"),
        _candidate("sport-b", "write a script about gamma versus delta"),
        _candidate("other", "Explain why a database transaction may deadlock under contention"),
    ]

    families = candidate_families(rows)
    identifier_sets = [_ids(family) for family in families]

    assert {"long-a", "long-b"} in identifier_sets
    assert {"sport-a", "sport-b"} in identifier_sets
    assert {"other"} in identifier_sets


def test_candidate_families_keeps_same_source_group_atomic() -> None:
    rows = [
        _candidate("turn-a", "Create a Python parser for invoices", "conversation-one"),
        _candidate("turn-b", "Now make the output sortable", "conversation-one"),
        _candidate("other", "Review this Rust unsafe block", "conversation-two"),
    ]

    families = candidate_families(rows)

    assert {"turn-a", "turn-b"} in [_ids(family) for family in families]


def test_split_excludes_entire_reviewed_family_from_reserve() -> None:
    shared_code = " ".join(f"statement{index}" for index in range(70))
    rows = [
        _candidate("seen", f"Fix the game {shared_code} red cards"),
        _candidate("unseen-variant", f"Improve the game {shared_code} black cards"),
        _candidate("fresh-a", "Implement a Rust parser for a compact binary header"),
        _candidate("fresh-b", "Compare current Python HTTP client libraries using benchmarks"),
        _candidate("fresh-c", "Why does this SQL query return duplicate rows"),
    ]

    development, queue, reserve, report = split_development_queue_reserve(
        rows,
        reviewed_ids=frozenset({"seen"}),
        reserve_target=1,
        queue_partitions=2,
        seed=9,
    )

    assert {"seen", "unseen-variant"} <= _ids(development)
    assert not ({"seen", "unseen-variant"} & _ids(reserve))
    assigned = _ids(development) | _ids(reserve) | {
        identifier for block in queue for identifier in _ids(block)
    }
    assert assigned == {str(row["candidate_id"]) for row in rows}
    assert report["multi_member_families"] >= 1
