"""Validated payload contracts for Tree engine pseudo-tools."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class _StrictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResolveFactInput(_StrictPayload):
    content: str = Field(min_length=1)
    # Tool-backed facts keep quote-level validation. User-supplied premises and
    # conceptual derivations can be recorded explicitly without fabricating a
    # tool call; evidence then carries an optional quote or rationale.
    evidence: str = ""
    source_tool: str = ""
    source_kind: Literal["tool", "user", "reasoning"] = "tool"
    confidence: Literal["high", "medium", "low"] = "medium"


class AnsweredQuestionInput(_StrictPayload):
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    answered_by_tool: str | None = None


class SubproblemQuestionInput(_StrictPayload):
    content: str = Field(min_length=1)
    question_type: Literal["pivot", "informational"] = "informational"


class ResolveSubproblemInput(_StrictPayload):
    problem: str = Field(min_length=1)
    logic: Literal["AND", "OR"] = "AND"
    questions: list[SubproblemQuestionInput] = Field(default_factory=list)


class ResolveBlockerInput(_StrictPayload):
    description: str = Field(min_length=1)
    blocker_type: Literal["api", "library", "permission", "infra", "unknown"] = "unknown"
    workaround: str | None = None


class ResolveNodeInput(_StrictPayload):
    """Runtime contract matching the public ``resolve_node`` pseudo-tool."""

    state: Literal[
        "solvable",
        "unsolvable",
        "mitigable",
        "needs_decision",
        "needs_experiment",
        "hypothesis",
    ]
    confidence: Literal["high", "medium", "low"]
    summary: str = Field(min_length=1)
    hypothesis_content: str | None = None
    new_facts: list[ResolveFactInput] = Field(default_factory=list)
    answered_questions: list[AnsweredQuestionInput] = Field(default_factory=list)
    new_sub_problems: list[ResolveSubproblemInput] = Field(default_factory=list)
    new_constraints: list[str] = Field(default_factory=list)
    new_blockers: list[ResolveBlockerInput] = Field(default_factory=list)
    problem_reformulated: str | None = None
    discard_reason: str | None = None

    @model_validator(mode="after")
    def _validate_hypothesis(self) -> "ResolveNodeInput":
        if self.state == "hypothesis" and not self.hypothesis_content:
            raise ValueError("hypothesis state requires hypothesis_content")
        if self.state != "hypothesis" and self.hypothesis_content:
            raise ValueError("hypothesis_content requires hypothesis state")
        return self


def validate_tool_payload(
    value: Any,
    schemas: list[dict[str, Any]],
    expected_tool: str,
) -> str:
    """Validate the JSON-Schema subset used by Tree pseudo-tools."""
    for tool_schema in schemas:
        function = tool_schema.get("function", {})
        if function.get("name") == expected_tool:
            parameters = function.get("parameters", {})
            return _schema_value_error(value, parameters, path="arguments")
    return ""


def _schema_value_error(value: Any, schema: dict[str, Any], *, path: str) -> str:
    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return f"{path} must be an object"
        required = schema.get("required", [])
        missing = [name for name in required if name not in value]
        if missing:
            return f"{path} is missing required fields: {', '.join(missing)}"
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                return f"{path} has unknown fields: {', '.join(extras)}"
        for name, child in properties.items():
            if name in value:
                error = _schema_value_error(
                    value[name], child, path=f"{path}.{name}"
                )
                if error:
                    return error
    elif expected == "array":
        if not isinstance(value, list):
            return f"{path} must be an array"
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            error = _schema_value_error(
                item, item_schema, path=f"{path}[{index}]"
            )
            if error:
                return error
    elif expected == "string" and not isinstance(value, str):
        return f"{path} must be a string"
    elif expected == "boolean" and not isinstance(value, bool):
        return f"{path} must be a boolean"
    elif expected == "integer" and (
        not isinstance(value, int) or isinstance(value, bool)
    ):
        return f"{path} must be an integer"
    elif expected == "number" and (
        not isinstance(value, (int, float)) or isinstance(value, bool)
    ):
        return f"{path} must be a number"

    allowed = schema.get("enum")
    if allowed is not None and value not in allowed:
        choices = ", ".join(repr(choice) for choice in allowed)
        return f"{path} must be one of: {choices}"
    return ""


def validate_resolution_evidence(
    payload: ResolveNodeInput,
    observed_tool_outputs: dict[str, list[str]],
) -> str:
    """Validate support in proportion to each fact's declared source."""
    for index, fact in enumerate(payload.new_facts):
        if fact.source_kind == "tool":
            if not fact.evidence or not fact.source_tool:
                return (
                    f"Invalid resolve_node evidence: new_facts[{index}] with "
                    "source_kind='tool' requires evidence and source_tool."
                )
            quote = " ".join(fact.evidence.split())
            outputs = observed_tool_outputs.get(fact.source_tool, [])
            if not any(quote in " ".join(output.split()) for output in outputs):
                return (
                    f"Invalid resolve_node evidence: new_facts[{index}].evidence "
                    f"was not observed in output from {fact.source_tool!r}."
                )
        elif (
            fact.source_kind == "reasoning"
            and fact.confidence == "high"
            and not fact.evidence
        ):
            return (
                f"Invalid resolve_node evidence: new_facts[{index}] cannot claim "
                "high-confidence reasoning without a supporting rationale."
            )
    return ""


def validate_resolve_node(value: Any) -> tuple[ResolveNodeInput | None, str]:
    """Validate a resolution payload and return concise model-facing feedback."""
    try:
        return ResolveNodeInput.model_validate(value), ""
    except ValidationError as exc:
        details: list[str] = []
        for issue in exc.errors(include_url=False)[:3]:
            location = ".".join(str(part) for part in issue["loc"]) or "payload"
            details.append(f"{location}: {issue['msg']}")
        return None, "Invalid resolve_node arguments: " + "; ".join(details)
