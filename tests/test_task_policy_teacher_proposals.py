from __future__ import annotations

import json

import pytest

from bench.task_policy_teacher_proposals import (
    messages_for_request,
    parse_teacher_decision,
)


def test_teacher_prompt_uses_only_request_text() -> None:
    messages = messages_for_request("The documented endpoint returns stale state; fix it.")

    assert messages[-1]["content"].endswith("fix it.\n</user_request>")
    assert "upstream_category_hint" not in json.dumps(messages)
    assert "Normal diagnosis before a fix is not research" in messages[0]["content"]


def test_parse_teacher_decision_accepts_label_and_zero_label() -> None:
    labeled = parse_teacher_decision(
        '{"policies":["bugfix"],"uncategorized_reason":null,'
        '"confidence":0.91,"rationale":"It restores an existing contract."}'
    )
    empty = parse_teacher_decision(
        'answer: {"policies":[],"uncategorized_reason":"unsupported_method",'
        '"confidence":0.8,"rationale":"Only documentation is requested."}'
    )

    assert labeled.policies == ("bugfix",)
    assert empty.uncategorized_reason == "unsupported_method"


@pytest.mark.parametrize(
    "payload",
    [
        {"policies": ["review", "feature"], "uncategorized_reason": None},
        {"policies": ["unknown"], "uncategorized_reason": None},
        {"policies": [], "uncategorized_reason": None},
        {"policies": ["bugfix"], "uncategorized_reason": "ambiguous_method"},
    ],
)
def test_parse_teacher_decision_rejects_unsafe_or_incomplete_labels(payload: dict) -> None:
    payload.update({"confidence": 0.8, "rationale": "A rationale."})

    with pytest.raises(ValueError):
        parse_teacher_decision(json.dumps(payload))
