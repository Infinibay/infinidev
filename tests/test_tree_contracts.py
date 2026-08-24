"""Contract tests for Tree engine pseudo-tool payloads."""

from infinidev.engine.tree.context import INIT_TREE_SCHEMA, RESOLVE_NODE_SCHEMA
from infinidev.engine.tree.contracts import (
    ResolveNodeInput,
    validate_resolution_evidence,
    validate_resolve_node,
    validate_tool_payload,
)


def _valid_payload() -> dict:
    return {
        "state": "solvable",
        "confidence": "high",
        "summary": "Observed implementation is viable",
    }


def test_valid_resolution_is_normalized() -> None:
    payload, error = validate_resolve_node({
        **_valid_payload(),
        "summary": "  Observed implementation is viable  ",
    })

    assert error == ""
    assert payload is not None
    assert payload.summary == "Observed implementation is viable"


def test_missing_required_resolution_fields_are_rejected() -> None:
    payload, error = validate_resolve_node({})

    assert payload is None
    assert "state" in error
    assert "confidence" in error
    assert "summary" in error


def test_blank_resolution_summary_is_rejected() -> None:
    payload, error = validate_resolve_node({**_valid_payload(), "summary": "   "})

    assert payload is None
    assert "summary" in error


def test_hypothesis_requires_explicit_speculative_content() -> None:
    payload, error = validate_resolve_node({
        **_valid_payload(),
        "state": "hypothesis",
    })

    assert payload is None
    assert "hypothesis_content" in error


def test_tool_facts_require_observed_tool_evidence() -> None:
    payload, error = validate_resolve_node({
        **_valid_payload(),
        "new_facts": [{"content": "The endpoint exists"}],
    })

    assert error == ""
    assert payload is not None
    evidence_error = validate_resolution_evidence(payload, {})

    assert "requires evidence and source_tool" in evidence_error


def test_user_supplied_fact_does_not_require_a_fabricated_tool_call() -> None:
    payload, error = validate_resolve_node({
        **_valid_payload(),
        "new_facts": [{
            "content": "The deployment target is staging",
            "source_kind": "user",
            "confidence": "high",
        }],
    })

    assert error == ""
    assert payload is not None
    assert validate_resolution_evidence(payload, {}) == ""


def test_reasoning_fact_can_be_recorded_without_external_evidence() -> None:
    payload, error = validate_resolve_node({
        **_valid_payload(),
        "new_facts": [{
            "content": "Option A dominates when latency is the only objective",
            "source_kind": "reasoning",
            "confidence": "medium",
        }],
    })

    assert error == ""
    assert payload is not None
    assert validate_resolution_evidence(payload, {}) == ""


def test_high_confidence_reasoning_requires_a_rationale() -> None:
    payload, error = validate_resolve_node({
        **_valid_payload(),
        "new_facts": [{
            "content": "Option A is always superior",
            "source_kind": "reasoning",
            "confidence": "high",
        }],
    })

    assert error == ""
    assert payload is not None
    assert "supporting rationale" in validate_resolution_evidence(payload, {})


def test_generic_schema_validation_checks_nested_shapes() -> None:
    error = validate_tool_payload(
        {"root_problem": "Inspect", "sub_problems": [{"logic": "AND"}]},
        [INIT_TREE_SCHEMA],
        "init_tree",
    )

    assert error == "arguments.sub_problems[0] is missing required fields: problem"


def test_generic_schema_validation_rejects_unknown_resolve_fields() -> None:
    error = validate_tool_payload(
        {**_valid_payload(), "invented": True},
        [RESOLVE_NODE_SCHEMA],
        "resolve_node",
    )

    assert error == "arguments has unknown fields: invented"


def test_public_resolve_schema_matches_runtime_contract() -> None:
    public = RESOLVE_NODE_SCHEMA["function"]["parameters"]
    runtime = ResolveNodeInput.model_json_schema()

    assert set(public["properties"]) == set(runtime["properties"])
    assert set(public["required"]) == set(runtime["required"])
    assert public["additionalProperties"] is False
    fact_schema = public["properties"]["new_facts"]["items"]
    assert fact_schema["additionalProperties"] is False
    assert set(fact_schema["required"]) == {"content"}
    assert fact_schema["properties"]["source_kind"]["enum"] == [
        "tool", "user", "reasoning",
    ]
    assert public["properties"]["state"]["enum"] == runtime["properties"]["state"]["enum"]
    assert public["properties"]["confidence"]["enum"] == (
        runtime["properties"]["confidence"]["enum"]
    )
