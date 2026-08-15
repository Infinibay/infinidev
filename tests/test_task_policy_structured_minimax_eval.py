from __future__ import annotations

import json

import pytest

from bench.task_policy_structured_minimax_eval import (
    map_work_items,
    messages_for_batch,
    parse_structured_batch,
)


def _item(effect: str, relation: str, operation: str, acceptance: str) -> dict:
    return {
        "effect": effect,
        "relation": relation,
        "operation": operation,
        "acceptance": acceptance,
        "evidence_scope": "none",
        "explicit_no_product_change": False,
        "evidence_quote": "exact request span",
    }


def test_mapper_unions_independent_work_items() -> None:
    row = {
        "request_state": "actionable",
        "items": [
            _item("product_change", "existing_contract", "restore", "contract_restored"),
            _item(
                "product_change", "internal_structure", "reorganize",
                "behavior_unchanged",
            ),
        ],
    }

    assert map_work_items(row) == ["bugfix", "refactor"]


def test_mapper_does_not_force_unsupported_or_conflicting_work() -> None:
    unsupported = {
        "request_state": "actionable",
        "items": [_item("product_change", "other", "other", "other")],
    }
    conflicting = {"request_state": "conflicting", "items": []}

    assert map_work_items(unsupported) == []
    assert map_work_items(conflicting) == []


def test_parser_is_strict_and_prompt_hides_gold_labels() -> None:
    item = _item("read_only_assessment", "bounded_artifact", "assess", "finding_quality")
    payload = [{"candidate_id": "one", "request_state": "actionable", "items": [item]}]
    parsed = parse_structured_batch(json.dumps(payload), ["one"])
    messages = messages_for_batch([
        {"candidate_id": "one", "text": "Inspect this patch only.", "expected": ("review",)},
    ])

    assert map_work_items(parsed[0]) == ["review"]
    assert "expected" not in messages[1]["content"]
    assert "review" not in messages[1]["content"].casefold()


def test_parser_rejects_extra_keys() -> None:
    payload = [{
        "candidate_id": "one", "request_state": "actionable", "items": [], "labels": [],
    }]

    with pytest.raises(ValueError, match="invalid keys"):
        parse_structured_batch(json.dumps(payload), ["one"])
