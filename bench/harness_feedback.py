#!/usr/bin/env python3
"""Validate and report model-authored hypotheses about harness friction."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


FEEDBACK_CATEGORIES = frozenset(
    {
        "context_delivery",
        "tool_interface",
        "planning_protocol",
        "verification_protocol",
        "user_interaction",
        "prompt_clarity",
        "prompt_overload",
        "error_recovery",
        "completion_protocol",
    }
)


@dataclass(frozen=True)
class FeedbackCase:
    """One reviewed situation in which the model may critique the harness."""

    id: str
    category: str
    scenario: str
    visible_artifact: str
    question: str
    split: str = "calibration"
    review_status: str = "draft"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FeedbackCase:
        case = cls(
            id=str(value.get("id", "")).strip(),
            category=str(value.get("category", "")).strip(),
            scenario=str(value.get("scenario", "")).strip(),
            visible_artifact=str(value.get("visible_artifact", "")).strip(),
            question=str(value.get("question", "")).strip(),
            split=str(value.get("split", "calibration")).strip(),
            review_status=str(value.get("review_status", "draft")).strip(),
        )
        if not all((case.id, case.scenario, case.visible_artifact, case.question)):
            raise ValueError("feedback case is missing a required field")
        if case.category not in FEEDBACK_CATEGORIES:
            raise ValueError(f"unsupported feedback category: {case.category}")
        if case.split not in {"calibration", "validation"}:
            raise ValueError("feedback case split must be calibration or validation")
        if case.review_status not in {"draft", "approved", "rejected"}:
            raise ValueError("feedback case review_status is invalid")
        return case

    @property
    def sha256(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def rendered_prompt(self) -> str:
        return (
            f"Scenario:\n{self.scenario}\n\n"
            f"Visible artifact:\n{self.visible_artifact}\n\n"
            f"Question:\n{self.question}\n\n"
            "Return one JSON object with exactly these fields:\n"
            '{"no_change_warranted":false,"assessment":"brief observable assessment",'
            '"friction":"one concrete friction or empty string","evidence":"artifact-grounded '
            'evidence or empty string","suggested_change":"smallest proposed change or empty '
            'string","expected_effect":"observable expected effect or empty string","risk":"most '
            'important regression risk or empty string","experiment":"paired falsifiable test or '
            'empty string"}\n'
            "Do not provide hidden chain-of-thought. Treat your proposal as a hypothesis, not a "
            "directive. Set no_change_warranted=true when the artifact does not justify a change."
        )


@dataclass(frozen=True)
class HarnessFeedback:
    """One structured, externally reportable model hypothesis."""

    no_change_warranted: bool
    assessment: str
    friction: str
    evidence: str
    suggested_change: str
    expected_effect: str
    risk: str
    experiment: str

    _FIELDS = frozenset(
        {
            "no_change_warranted",
            "assessment",
            "friction",
            "evidence",
            "suggested_change",
            "expected_effect",
            "risk",
            "experiment",
        }
    )

    @classmethod
    def from_text(cls, text: str) -> HarnessFeedback:
        return cls.from_mapping(_first_json_object(text))

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> HarnessFeedback:
        if set(value) != set(cls._FIELDS):
            raise ValueError(f"feedback fields must be exactly {sorted(cls._FIELDS)}")
        if not isinstance(value.get("no_change_warranted"), bool):
            raise ValueError("feedback no_change_warranted must be boolean")
        feedback = cls(
            no_change_warranted=bool(value["no_change_warranted"]),
            assessment=str(value.get("assessment", "")).strip(),
            friction=str(value.get("friction", "")).strip(),
            evidence=str(value.get("evidence", "")).strip(),
            suggested_change=str(value.get("suggested_change", "")).strip(),
            expected_effect=str(value.get("expected_effect", "")).strip(),
            risk=str(value.get("risk", "")).strip(),
            experiment=str(value.get("experiment", "")).strip(),
        )
        if not feedback.assessment:
            raise ValueError("feedback assessment cannot be empty")
        hypothesis_fields = (
            feedback.friction,
            feedback.evidence,
            feedback.suggested_change,
            feedback.expected_effect,
            feedback.risk,
            feedback.experiment,
        )
        if feedback.no_change_warranted:
            if any(hypothesis_fields):
                raise ValueError("no-change feedback must not smuggle a change hypothesis")
        elif any(not field for field in hypothesis_fields):
            raise ValueError("change feedback needs friction, evidence, change, effect, risk, and experiment")
        return feedback


@dataclass(frozen=True)
class FeedbackObservation:
    """Provenance-bound raw feedback from one isolated model call."""

    case_id: str
    case_sha256: str
    model_identity: str
    repetition: int
    response_text: str
    feedback: HarnessFeedback | None
    dataset_sha256: str = ""
    latency_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FeedbackObservation:
        raw_feedback = value.get("feedback")
        feedback = None
        if isinstance(raw_feedback, dict):
            feedback = HarnessFeedback.from_mapping(raw_feedback)
        row = cls(
            case_id=str(value.get("case_id", "")).strip(),
            case_sha256=str(value.get("case_sha256", "")).strip(),
            model_identity=str(value.get("model_identity", "")).strip(),
            repetition=int(value.get("repetition", 0)),
            response_text=str(value.get("response_text", "")),
            feedback=feedback,
            dataset_sha256=str(value.get("dataset_sha256", "")).strip(),
            latency_seconds=(
                float(value["latency_seconds"])
                if value.get("latency_seconds") is not None
                else None
            ),
            input_tokens=(
                int(value["input_tokens"]) if value.get("input_tokens") is not None else None
            ),
            output_tokens=(
                int(value["output_tokens"])
                if value.get("output_tokens") is not None
                else None
            ),
            error=str(value.get("error", "")).strip(),
        )
        if not all((row.case_id, row.case_sha256, row.model_identity)):
            raise ValueError("feedback observation is missing provenance")
        if row.repetition < 0:
            raise ValueError("feedback repetition cannot be negative")
        if row.latency_seconds is not None and row.latency_seconds < 0:
            raise ValueError("feedback latency cannot be negative")
        if any(value is not None and value < 0 for value in (row.input_tokens, row.output_tokens)):
            raise ValueError("feedback token counts cannot be negative")
        if bool(row.feedback) == bool(row.error):
            raise ValueError("feedback observation needs exactly one feedback or error outcome")
        return row


def _first_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("feedback response did not contain a JSON object")


def load_cases(path: Path) -> list[FeedbackCase]:
    return [
        FeedbackCase.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_observations(path: Path) -> list[FeedbackObservation]:
    return [
        FeedbackObservation.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_feedback_report(
    cases: Iterable[FeedbackCase], observations: Iterable[FeedbackObservation]
) -> dict[str, object]:
    """Preserve concrete hypotheses and experiments without ranking suggestions as truth."""
    case_list = list(cases)
    by_id = {case.id: case for case in case_list}
    if len(by_id) != len(case_list):
        raise ValueError("feedback cases contain duplicate IDs")
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    identities: set[str] = set()
    seen: set[tuple[str, str, int]] = set()
    for row in observations:
        case = by_id.get(row.case_id)
        if case is None:
            raise ValueError(f"feedback observation has unknown case: {row.case_id}")
        if row.case_sha256 != case.sha256:
            raise ValueError(f"feedback case hash mismatch: {row.case_id}")
        key = (row.case_id, row.model_identity, row.repetition)
        if key in seen:
            raise ValueError(f"duplicate feedback observation: {key}")
        seen.add(key)
        identities.add(row.model_identity)
        grouped[case.category].append(
            {
                "case": asdict(case),
                "model_identity": row.model_identity,
                "repetition": row.repetition,
                "raw_response": row.response_text,
                "dataset_sha256": row.dataset_sha256,
                "latency_seconds": row.latency_seconds,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "feedback": asdict(row.feedback) if row.feedback else None,
                "error": row.error,
                "interpretation": (
                    "Unverified model-authored hypothesis; implement only after independent review "
                    "and a preregistered paired experiment."
                    if row.feedback and not row.feedback.no_change_warranted
                    else "No change proposed by this observation."
                ),
            }
        )
    return {
        "interpretation_boundary": (
            "Model feedback is an observable product hypothesis, not privileged access to model "
            "cognition and not evidence that the suggested change helps. Raw responses remain "
            "primary; consensus only prioritizes experiments."
        ),
        "model_identities": sorted(identities),
        "case_count": len(case_list),
        "observation_count": len(seen),
        "categories": {key: grouped[key] for key in sorted(grouped)},
    }


def render_markdown(report: Mapping[str, object]) -> str:
    lines = ["# Harness feedback hypotheses", "", str(report["interpretation_boundary"])]
    categories = report.get("categories", {})
    if isinstance(categories, dict):
        for category, records in categories.items():
            lines.extend(["", f"## {category}"])
            for record in records if isinstance(records, list) else []:
                case = record["case"]
                lines.extend(
                    [
                        "",
                        f"### `{case['id']}` — `{record['model_identity']}` repetition {record['repetition']}",
                        "",
                        f"Visible artifact: {case['visible_artifact']}",
                        f"Raw response: `{record['raw_response']}`",
                        f"Parsed feedback: `{record['feedback']}`",
                        f"Interpretation: {record['interpretation']}",
                    ]
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("observations", type=Path)
    parser.add_argument("output_markdown", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    report = build_feedback_report(load_cases(args.cases), load_observations(args.observations))
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
