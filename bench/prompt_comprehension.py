#!/usr/bin/env python3
"""Contracts for isolated, provider-neutral prompt-comprehension studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from infinidev.engine.prompt_layers import PromptLayer, PromptLayerKind, compose_layers


COMPREHENSION_FIELDS = (
    "objective",
    "deliverables",
    "constraints",
    "user_owned_decisions",
    "authorized_actions",
    "unauthorized_actions",
    "verification",
    "ambiguities",
    "stop_conditions",
    "conflicts",
    "priority_resolution",
    "interpretation_risks",
)

RESPONSE_INSTRUCTION = """Do not execute the request. Report only what you understand it to mean.
Return one JSON object with exactly these fields:
{
  "understanding": "free reconstruction in your own words",
  "objective": "primary requested outcome",
  "deliverables": ["explicit deliverable"],
  "constraints": ["explicit constraint"],
  "user_owned_decisions": ["decision whose decisive priority belongs to the user"],
  "authorized_actions": ["action the request authorizes"],
  "unauthorized_actions": ["material action it does not authorize"],
  "verification": ["evidence needed before claiming completion"],
  "ambiguities": ["unresolved ambiguity, or empty list"],
  "stop_conditions": ["condition requiring a pause or final handoff"],
  "conflicts": ["instructions or facts that conflict, or empty list"],
  "priority_resolution": "which instruction wins and why, or empty string",
  "interpretation_risks": ["plausible misreading or unsupported inference"],
  "confidence": 0.0
}
Use concise externally reportable conclusions. Do not provide hidden chain-of-thought."""


@dataclass(frozen=True)
class ComprehensionCase:
    """One reviewed request-under-test and its hidden interpretation key."""

    id: str
    category: str
    request: str
    split: str
    review_status: str
    expected: dict[str, object]
    tags: tuple[str, ...] = ()
    problem_id: str = ""
    research_question_id: str = ""
    family_id: str = ""
    variant_id: str = "anchor"
    intended_relation: str = "anchor"
    stimulus_profile: dict[str, str] | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ComprehensionCase:
        expected = value.get("expected")
        expected_fields = dict(expected) if isinstance(expected, dict) else {}
        expected_fields.setdefault("conflicts", [])
        expected_fields.setdefault("priority_resolution", "")
        expected_fields.setdefault("interpretation_risks", [])
        raw_tags = value.get("tags", [])
        case = cls(
            id=str(value.get("id", "")).strip(),
            category=str(value.get("category", "")).strip(),
            request=str(value.get("request", "")).strip(),
            split=str(value.get("split", "")).strip(),
            review_status=str(value.get("review_status", "draft")).strip(),
            expected=expected_fields,
            tags=tuple(str(tag) for tag in raw_tags) if isinstance(raw_tags, list) else (),
            problem_id=str(value.get("problem_id", "")).strip(),
            research_question_id=str(value.get("research_question_id", "")).strip(),
            family_id=str(value.get("family_id", value.get("id", ""))).strip(),
            variant_id=str(value.get("variant_id", "anchor")).strip(),
            intended_relation=str(value.get("intended_relation", "anchor")).strip(),
            stimulus_profile=(
                {str(key): str(item) for key, item in value["stimulus_profile"].items()}
                if isinstance(value.get("stimulus_profile"), dict)
                else None
            ),
        )
        if not case.id or not case.category or not case.request:
            raise ValueError("comprehension case needs id, category, and request")
        if case.split not in {"calibration", "validation"}:
            raise ValueError(f"unsupported comprehension split: {case.split}")
        if case.review_status not in {"draft", "approved", "rejected"}:
            raise ValueError("invalid comprehension review_status")
        if not case.family_id or not case.variant_id:
            raise ValueError("comprehension case needs family_id and variant_id")
        if case.intended_relation not in {"anchor", "equivalent", "contrast", "adversarial"}:
            raise ValueError("invalid intended_relation")
        missing = set(COMPREHENSION_FIELDS) - set(case.expected)
        if missing:
            raise ValueError(f"comprehension key lacks fields: {sorted(missing)}")
        return case


@dataclass(frozen=True)
class ComprehensionCondition:
    """Behavior and execution-policy shells; never an objective replacement."""

    name: str
    behavior_prompt: str | None = None
    execution_policy_prompt: str | None = None

    @classmethod
    def from_value(cls, name: str, value: object) -> ComprehensionCondition:
        if value is None:
            return cls(name)
        if not isinstance(value, dict):
            raise ValueError("comprehension condition must be null or an object")
        unknown = set(value) - {"behavior_prompt", "execution_policy_prompt"}
        if unknown:
            raise ValueError(f"condition contains forbidden prompt responsibilities: {sorted(unknown)}")
        behavior = value.get("behavior_prompt")
        execution = value.get("execution_policy_prompt")
        if behavior is not None and not isinstance(behavior, str):
            raise ValueError("behavior_prompt must be text or null")
        if execution is not None and not isinstance(execution, str):
            raise ValueError("execution_policy_prompt must be text or null")
        if execution and not behavior:
            raise ValueError("execution-policy condition requires the same behavior shell")
        return cls(name, behavior, execution)

    @property
    def sha256(self) -> str:
        value = {
            "behavior_prompt": self.behavior_prompt,
            "execution_policy_prompt": self.execution_policy_prompt,
        }
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def system_prompt(self) -> str | None:
        layers = []
        if self.behavior_prompt:
            layers.append(
                PromptLayer(
                    PromptLayerKind.BEHAVIOR,
                    self.behavior_prompt,
                    "comprehension-condition",
                )
            )
        if self.execution_policy_prompt:
            layers.append(
                PromptLayer(
                    PromptLayerKind.EXECUTION_POLICY,
                    self.execution_policy_prompt,
                    "comprehension-condition",
                )
            )
        return compose_layers(layers) or None


@dataclass(frozen=True)
class ComprehensionObservation:
    """Raw response plus parsed understanding, without reducing it to one score."""

    case_id: str
    category: str
    condition: str
    condition_sha256: str
    model_identity: str
    response_text: str
    parsed: dict[str, object]
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str = ""
    dataset_sha256: str = ""


def load_cases(path: Path) -> list[ComprehensionCase]:
    """Load and validate a JSONL case catalog with unique IDs."""
    cases = [
        ComprehensionCase.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("comprehension dataset contains duplicate IDs")
    return cases


def comprehension_messages(
    case: ComprehensionCase, condition: ComprehensionCondition
) -> list[dict[str, str]]:
    """Build one fresh conversation; raw has exactly one user message."""
    messages: list[dict[str, str]] = []
    if system_prompt := condition.system_prompt():
        messages.append({"role": "system", "content": system_prompt})
    user_content = (
        "<request-under-test>\n"
        f"{case.request}\n"
        "</request-under-test>\n\n"
        f"{RESPONSE_INSTRUCTION}"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def parse_comprehension_reply(text: str) -> dict[str, object]:
    """Extract a complete structured record while retaining free reconstruction."""
    decoder = json.JSONDecoder()
    payload: object = None
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(payload, dict):
        raise ValueError("response did not contain a JSON object")
    if set(payload) != {"understanding", "confidence", *COMPREHENSION_FIELDS}:
        raise ValueError("response fields do not match comprehension contract")
    if not isinstance(payload["understanding"], str) or not payload["understanding"].strip():
        raise ValueError("understanding reconstruction is empty")
    if not isinstance(payload["objective"], str) or not payload["objective"].strip():
        raise ValueError("objective is empty")
    list_fields = set(COMPREHENSION_FIELDS[1:]) - {"priority_resolution"}
    for field in list_fields:
        value = payload[field]
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"comprehension field must be a list of strings: {field}")
    if not isinstance(payload["priority_resolution"], str):
        raise ValueError("priority_resolution must be text")
    confidence = payload["confidence"]
    if not isinstance(confidence, (int, float)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return dict(payload)


def append_observation(path: Path, observation: ComprehensionObservation) -> None:
    """Durably append a completed isolated call."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(observation), ensure_ascii=False) + "\n")
        stream.flush()
