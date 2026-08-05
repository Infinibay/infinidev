from __future__ import annotations

import json

import pytest

from bench.prompt_comprehension import (
    ComprehensionCase,
    ComprehensionCondition,
    comprehension_messages,
    parse_comprehension_reply,
)


def _case() -> ComprehensionCase:
    return ComprehensionCase.from_dict(
        {
            "id": "ambiguous-migration",
            "category": "requirements",
            "request": "Migrate the database safely. Do not deploy.",
            "split": "validation",
            "review_status": "approved",
            "expected": {
                "objective": "migration plan",
                "deliverables": [],
                "constraints": ["do not deploy"],
                "user_owned_decisions": [],
                "authorized_actions": ["inspect and plan"],
                "unauthorized_actions": ["deploy"],
                "verification": [],
                "ambiguities": ["target schema"],
                "stop_conditions": ["deployment requires authorization"],
                "conflicts": [],
                "priority_resolution": "",
                "interpretation_risks": ["Mistaking planning for deployment authority"],
            },
        }
    )


def test_raw_comprehension_is_one_user_message_without_system_or_history() -> None:
    messages = comprehension_messages(_case(), ComprehensionCondition("raw"))

    assert [message["role"] for message in messages] == ["user"]
    assert "Migrate the database safely" in messages[0]["content"]
    assert "Do not execute" in messages[0]["content"]


def test_condition_layers_cannot_replace_objective_or_context() -> None:
    with pytest.raises(ValueError, match="forbidden prompt responsibilities"):
        ComprehensionCondition.from_value("bad", {"objective": "different task"})

    condition = ComprehensionCondition.from_value(
        "full",
        {"behavior_prompt": "Be concise.", "execution_policy_prompt": "Inspect first."},
    )
    system = condition.system_prompt()
    assert system is not None
    assert "<behavior-layer" in system
    assert "<execution-policy-layer" in system


def test_parsing_retains_free_reconstruction_and_structured_fields() -> None:
    payload = {
        "understanding": "Prepare a migration but do not deploy it.",
        "objective": "Prepare the migration.",
        "deliverables": [],
        "constraints": ["Do not deploy."],
        "user_owned_decisions": [],
        "authorized_actions": ["Inspect files."],
        "unauthorized_actions": ["Deploy."],
        "verification": ["Run migration tests."],
        "ambiguities": ["Target schema is unknown."],
        "stop_conditions": ["Ask before deployment."],
        "conflicts": [],
        "priority_resolution": "",
        "interpretation_risks": ["Deploying despite the prohibition."],
        "confidence": 0.9,
    }

    parsed = parse_comprehension_reply(f"```json\n{json.dumps(payload)}\n```")

    assert parsed["understanding"].startswith("Prepare")
    assert parsed["unauthorized_actions"] == ["Deploy."]
