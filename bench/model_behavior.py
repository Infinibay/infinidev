#!/usr/bin/env python3
"""Run and analyze reproducible behavioral probes for model harnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


UTILITY_AXES = frozenset(
    {
        "autonomy",
        "interaction",
        "user_control",
        "speed",
        "quality",
        "cost_efficiency",
        "caution",
    }
)


@dataclass(frozen=True)
class Probe:
    """One observable-behavior test with an optional perturbation family."""

    id: str
    category: str
    prompt: str
    choices: dict[str, str]
    answer: str | None
    group: str | None = None
    tags: tuple[str, ...] = ()
    scenario: str | None = None
    user_request: str | None = None
    split: str = "calibration"
    review_status: str = "draft"
    gold_rationale: str = ""
    reviewer: str = ""
    generator: str = ""
    analysis: dict[str, object] = field(default_factory=dict)
    evaluation_mode: str = "normative"
    choice_effects: dict[str, dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Probe:
        required = ("id", "category", "choices")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"probe is missing required fields: {', '.join(missing)}")
        prompt = value.get("prompt")
        scenario = value.get("scenario")
        user_request = value.get("user_request")
        if prompt is None and (scenario is None or user_request is None):
            raise ValueError("probe needs prompt or both scenario and user_request")
        choices = value["choices"]
        if not isinstance(choices, dict) or len(choices) < 2:
            raise ValueError("probe choices must be an object with at least two entries")
        normalized_choices = {str(key).upper(): str(text) for key, text in choices.items()}
        evaluation_mode = str(value.get("evaluation_mode", "normative"))
        if evaluation_mode not in {"normative", "preference"}:
            raise ValueError("probe evaluation_mode must be normative or preference")
        raw_answer = value.get("answer")
        answer = str(raw_answer).upper() if raw_answer not in {None, ""} else None
        if evaluation_mode == "normative":
            if answer is None:
                raise ValueError("normative probe needs an answer")
            if answer not in normalized_choices:
                raise ValueError(f"probe answer {answer!r} is not present in choices")
        elif answer is not None:
            raise ValueError("preference probe must not define a universal answer")
        choice_effects = _choice_effects(
            value.get("choice_effects", {}), normalized_choices, evaluation_mode
        )
        tags = value.get("tags", ())
        if not isinstance(tags, (list, tuple)):
            raise ValueError("probe tags must be an array")
        group = value.get("group")
        split = str(value.get("split", "calibration"))
        if split not in {"calibration", "validation"}:
            raise ValueError("probe split must be calibration or validation")
        review_status = str(value.get("review_status", "draft"))
        if review_status not in {"draft", "approved", "rejected"}:
            raise ValueError("probe review_status must be draft, approved, or rejected")
        analysis = value.get("analysis", {})
        if not isinstance(analysis, dict):
            raise ValueError("probe analysis must be an object")
        return cls(
            id=str(value["id"]),
            category=str(value["category"]),
            prompt=str(prompt) if prompt is not None else "",
            choices=normalized_choices,
            answer=answer,
            group=str(group) if group is not None else None,
            tags=tuple(str(tag) for tag in tags),
            scenario=str(scenario) if scenario is not None else None,
            user_request=str(user_request) if user_request is not None else None,
            split=split,
            review_status=review_status,
            gold_rationale=str(value.get("gold_rationale", "")),
            reviewer=str(value.get("reviewer", "")),
            generator=str(value.get("generator", "")),
            analysis={str(key): item for key, item in analysis.items()},
            evaluation_mode=evaluation_mode,
            choice_effects=choice_effects,
        )

    def rendered_prompt(self) -> str:
        """Render only the scenario, request, and offered actions."""
        if self.prompt:
            question = self.prompt
        else:
            question = (
                f"Scenario:\n{self.scenario}\n\n"
                f"User request:\n{self.user_request}\n\n"
                "What should the agent do next?"
            )
        choices = "\n".join(f"{key}) {text}" for key, text in self.choices.items())
        return f"{question}\n\n{choices}"


@dataclass(frozen=True)
class UtilityProfile:
    """User-specific behavioral objective used only on preference probes."""

    name: str
    weights: dict[str, float]
    description: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> UtilityProfile:
        name = str(value.get("name", "")).strip()
        raw_weights = value.get("weights")
        if not name or not isinstance(raw_weights, dict) or not raw_weights:
            raise ValueError("utility profile needs name and non-empty weights")
        weights: dict[str, float] = {}
        for raw_axis, raw_weight in raw_weights.items():
            axis = str(raw_axis)
            if axis not in UTILITY_AXES:
                raise ValueError(f"unknown utility axis: {axis}")
            weight = float(raw_weight)
            if not math.isfinite(weight) or not -1.0 <= weight <= 1.0:
                raise ValueError("utility weights must be finite and between -1 and 1")
            if weight:
                weights[axis] = weight
        if not weights:
            raise ValueError("utility profile needs at least one non-zero weight")
        return cls(name, weights, str(value.get("description", "")))

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            {"name": self.name, "weights": self.weights, "description": self.description},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def rendered(self) -> str:
        """Return a compact explicit preference context for model evaluation."""
        weights = ", ".join(
            f"{axis}={weight:+.2f}" for axis, weight in sorted(self.weights.items())
        )
        description = f" ({self.description})" if self.description else ""
        return f"{self.name}{description}: {weights}"

    def rendered_for_model(self) -> str:
        """Return the natural-language user preference without exposing score weights."""
        if not self.description.strip():
            raise ValueError("preference runs need a natural-language profile description")
        return self.description.strip()


@dataclass(frozen=True)
class Observation:
    """A model's externally observable response to one probe."""

    probe_id: str
    condition: str
    answer: str
    confidence: float | None
    latency_seconds: float | None = None
    tool_calls: int | None = None
    error: str | None = None
    repetition: int = 0
    model_identity: str = ""
    condition_sha256: str = ""
    response_text: str = ""
    decision_criterion: str = ""
    missing_context: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    utility_profile: str = ""
    utility_profile_sha256: str = ""
    elicitation_protocol: str = "self_report"
    option_order_protocol: str = "fixed"
    provider_answer: str = ""
    choice_mapping: dict[str, str] = field(default_factory=dict)
    presentation_id: str = ""
    dataset_sha256: str = ""
    manifest_sha256: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Observation:
        raw_confidence = value.get("confidence")
        confidence = float(raw_confidence) if raw_confidence is not None else None
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("observation confidence must be between 0 and 1")
        return cls(
            probe_id=str(value["probe_id"]),
            condition=str(value["condition"]),
            answer=str(value["answer"]).upper(),
            confidence=confidence,
            latency_seconds=_optional_float(value.get("latency_seconds")),
            tool_calls=_optional_int(value.get("tool_calls")),
            error=str(value["error"]) if value.get("error") else None,
            repetition=int(value.get("repetition", 0)),
            model_identity=str(value.get("model_identity", "")),
            condition_sha256=str(value.get("condition_sha256", "")),
            response_text=str(value.get("response_text", "")),
            decision_criterion=str(value.get("decision_criterion", "")),
            missing_context=str(value.get("missing_context", "")),
            input_tokens=_optional_int(value.get("input_tokens")),
            output_tokens=_optional_int(value.get("output_tokens")),
            utility_profile=str(value.get("utility_profile", "")),
            utility_profile_sha256=str(value.get("utility_profile_sha256", "")),
            elicitation_protocol=str(value.get("elicitation_protocol", "self_report")),
            option_order_protocol=str(value.get("option_order_protocol", "fixed")),
            provider_answer=str(value.get("provider_answer", value.get("answer", ""))).upper(),
            choice_mapping={
                str(key).upper(): str(item).upper()
                for key, item in dict(value.get("choice_mapping", {})).items()
            },
            presentation_id=str(value.get("presentation_id", "")),
            dataset_sha256=str(value.get("dataset_sha256", "")),
            manifest_sha256=str(value.get("manifest_sha256", "")),
        )


def _choice_effects(
    value: object, choices: Mapping[str, str], evaluation_mode: str
) -> dict[str, dict[str, float]]:
    if evaluation_mode == "normative":
        if value not in ({}, None):
            raise ValueError("normative probe must not define choice_effects")
        return {}
    if not isinstance(value, dict) or set(value) != set(choices):
        raise ValueError("preference probe needs choice_effects for every choice")
    normalized: dict[str, dict[str, float]] = {}
    for raw_choice, raw_effects in value.items():
        if not isinstance(raw_effects, dict) or not raw_effects:
            raise ValueError("each preference choice needs non-empty effects")
        effects: dict[str, float] = {}
        for raw_axis, raw_effect in raw_effects.items():
            axis = str(raw_axis)
            if axis not in UTILITY_AXES:
                raise ValueError(f"unknown utility axis: {axis}")
            effect = float(raw_effect)
            if not math.isfinite(effect) or not -1.0 <= effect <= 1.0:
                raise ValueError("choice effects must be finite and between -1 and 1")
            effects[axis] = effect
        normalized[str(raw_choice).upper()] = effects
    return normalized


def choice_utility(probe: Probe, answer: str, profile: UtilityProfile) -> float:
    """Return normalized linear utility for one preference-sensitive choice."""
    if probe.evaluation_mode != "preference":
        raise ValueError("choice utility is defined only for preference probes")
    if answer not in probe.choices:
        raise ValueError(f"answer {answer!r} is not present in probe choices")
    denominator = sum(abs(weight) for weight in profile.weights.values())
    return sum(
        weight * probe.choice_effects[answer].get(axis, 0.0)
        for axis, weight in profile.weights.items()
    ) / denominator


def preference_regret(probe: Probe, answer: str, profile: UtilityProfile) -> float:
    """Return utility lost relative to the best offered choice for this profile."""
    selected = choice_utility(probe, answer, profile)
    best = max(choice_utility(probe, choice, profile) for choice in probe.choices)
    return best - selected


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield validated JSON objects from a JSONL file with useful line errors."""
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_probes(path: Path) -> dict[str, Probe]:
    """Load probes and reject duplicate identities."""
    probes: dict[str, Probe] = {}
    for value in read_jsonl(path):
        probe = Probe.from_dict(value)
        if probe.id in probes:
            raise ValueError(f"duplicate probe id: {probe.id}")
        probes[probe.id] = probe
    if not probes:
        raise ValueError("probe dataset is empty")
    return probes


def load_observations(path: Path) -> list[Observation]:
    """Load model observations from JSONL."""
    return [Observation.from_dict(value) for value in read_jsonl(path)]


def brier_score(correct: bool, confidence: float) -> float:
    """Return binary Brier score for confidence assigned to the selected answer."""
    return (confidence - float(correct)) ** 2


def expected_calibration_error(
    outcomes: Sequence[tuple[bool, float]], bins: int = 10
) -> float:
    """Compute equal-width expected calibration error."""
    if not outcomes:
        return math.nan
    buckets: list[list[tuple[bool, float]]] = [[] for _ in range(bins)]
    for correct, confidence in outcomes:
        index = min(int(confidence * bins), bins - 1)
        buckets[index].append((correct, confidence))
    total = len(outcomes)
    return sum(
        len(bucket) / total
        * abs(
            sum(float(correct) for correct, _ in bucket) / len(bucket)
            - sum(confidence for _, confidence in bucket) / len(bucket)
        )
        for bucket in buckets
        if bucket
    )


def summarize(
    probes: dict[str, Probe],
    observations: Iterable[Observation],
    utility_profile: UtilityProfile | None = None,
) -> dict[str, dict[str, float | int]]:
    """Aggregate performance, calibration, robustness, cost, and failures by condition."""
    observation_rows = list(observations)
    protocols = {row.elicitation_protocol for row in observation_rows}
    if len(protocols) > 1:
        raise ValueError(
            "one summary cannot mix elicitation protocols; analyze each protocol separately"
        )
    by_condition: dict[str, list[Observation]] = defaultdict(list)
    for observation in observation_rows:
        if observation.probe_id not in probes:
            raise ValueError(f"unknown probe id in observations: {observation.probe_id}")
        by_condition[observation.condition].append(observation)

    result: dict[str, dict[str, float | int]] = {}
    for condition, rows in sorted(by_condition.items()):
        successful = [row for row in rows if row.error is None]
        normative_rows = [
            row for row in successful
            if probes[row.probe_id].evaluation_mode == "normative"
        ]
        preference_rows = [
            row for row in successful
            if probes[row.probe_id].evaluation_mode == "preference"
        ]
        correctness = [row.answer == probes[row.probe_id].answer for row in normative_rows]
        confidence_outcomes = [
            (row.answer == probes[row.probe_id].answer, row.confidence)
            for row in normative_rows
            if row.confidence is not None
        ]
        utilities = (
            [
                choice_utility(probes[row.probe_id], row.answer, utility_profile)
                for row in preference_rows
            ]
            if utility_profile else []
        )
        regrets = (
            [
                preference_regret(probes[row.probe_id], row.answer, utility_profile)
                for row in preference_rows
            ]
            if utility_profile else []
        )
        groups: dict[str, list[Observation]] = defaultdict(list)
        for row in successful:
            group = probes[row.probe_id].group
            if group:
                groups[group].append(row)
        complete_groups = [
            group_rows for group_rows in groups.values()
            if len(group_rows) > 1
            and all(
                probes[row.probe_id].evaluation_mode == "normative"
                for row in group_rows
            )
        ]
        robust = sum(
            all(row.answer == probes[row.probe_id].answer for row in group_rows)
            for group_rows in complete_groups
        )
        latencies = [row.latency_seconds for row in successful if row.latency_seconds is not None]
        tool_calls = [row.tool_calls for row in successful if row.tool_calls is not None]
        result[condition] = {
            "attempted": len(rows),
            "errors": len(rows) - len(successful),
            "normative_n": len(normative_rows),
            "preference_n": len(preference_rows),
            "accuracy": _mean([float(correct) for correct in correctness]),
            "confidence_n": len(confidence_outcomes),
            "brier": _mean([brier_score(*outcome) for outcome in confidence_outcomes]),
            "ece": expected_calibration_error(confidence_outcomes),
            "perturbation_success": (
                robust / len(complete_groups) if complete_groups else math.nan
            ),
            "mean_latency_seconds": _mean(latencies),
            "mean_tool_calls": _mean(tool_calls),
            "mean_preference_utility": _mean(utilities),
            "mean_preference_regret": _mean(regrets),
        }
    return result


def _mean(values: Sequence[float | int]) -> float:
    return sum(values) / len(values) if values else math.nan


def paired_comparison(
    probes: dict[str, Probe],
    observations: Iterable[Observation],
    baseline: str,
    utility_profile: UtilityProfile | None = None,
) -> dict[str, dict[str, int | float]]:
    """Compare each condition with a baseline on identical successful probes."""
    indexed = {
        (row.condition, row.probe_id, row.repetition): row
        for row in observations
        if row.error is None
    }
    conditions = sorted({condition for condition, _, _ in indexed} - {baseline})
    output: dict[str, dict[str, int | float]] = {}
    for condition in conditions:
        shared = sorted(
            (probe_id, repetition)
            for probe_id in probes
            for repetition in {
                key_repetition
                for key_condition, key_probe, key_repetition in indexed
                if key_condition == baseline and key_probe == probe_id
            }
            if (condition, probe_id, repetition) in indexed
        )
        wins = losses = ties = 0
        utility_wins = utility_losses = utility_ties = 0
        utility_deltas: list[float] = []
        for probe_id, repetition in shared:
            probe = probes[probe_id]
            baseline_answer = indexed[(baseline, probe_id, repetition)].answer
            candidate_answer = indexed[(condition, probe_id, repetition)].answer
            if probe.evaluation_mode == "normative":
                base_correct = baseline_answer == probe.answer
                candidate_correct = candidate_answer == probe.answer
                if candidate_correct and not base_correct:
                    wins += 1
                elif base_correct and not candidate_correct:
                    losses += 1
                else:
                    ties += 1
            elif utility_profile:
                delta = choice_utility(
                    probe, candidate_answer, utility_profile
                ) - choice_utility(probe, baseline_answer, utility_profile)
                utility_deltas.append(delta)
                if delta > 1e-12:
                    utility_wins += 1
                elif delta < -1e-12:
                    utility_losses += 1
                else:
                    utility_ties += 1
            else:
                utility_ties += 1
        normative_n = wins + losses + ties
        preference_n = utility_wins + utility_losses + utility_ties
        output[condition] = {
            "paired_n": normative_n,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "accuracy_delta": (
                (wins - losses) / normative_n if normative_n else math.nan
            ),
            "mcnemar_exact_p": mcnemar_exact_p(wins, losses),
            "preference_paired_n": preference_n,
            "utility_wins": utility_wins,
            "utility_losses": utility_losses,
            "utility_ties": utility_ties,
            "mean_utility_delta": _mean(utility_deltas),
            "utility_sign_exact_p": mcnemar_exact_p(
                utility_wins, utility_losses
            ),
        }
    return output


def mcnemar_exact_p(wins: int, losses: int) -> float:
    """Two-sided exact McNemar p-value for discordant paired outcomes."""
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index) for index in range(min(wins, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def _json_safe(value: object) -> object:
    """Replace non-finite metrics with JSON null recursively."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path, help="versioned probe dataset in JSONL format")
    parser.add_argument("observations", type=Path, help="model observations in JSONL format")
    parser.add_argument("--baseline", help="condition used for paired comparisons")
    parser.add_argument(
        "--utility-profile",
        type=Path,
        help="user preference profile used to score preference probes",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    probes = load_probes(args.probes)
    observations = load_observations(args.observations)
    utility_profile = (
        UtilityProfile.from_dict(
            json.loads(args.utility_profile.read_text(encoding="utf-8"))
        )
        if args.utility_profile else None
    )
    if any(probe.evaluation_mode == "preference" for probe in probes.values()):
        if utility_profile is None:
            raise ValueError("preference probes require --utility-profile")
        mismatched = [
            row.probe_id for row in observations
            if row.utility_profile_sha256
            and row.utility_profile_sha256 != utility_profile.sha256
        ]
        if mismatched:
            raise ValueError("observations contain a different utility profile")
    report: dict[str, object] = {
        "dataset_sha256": hashlib.sha256(args.probes.read_bytes()).hexdigest(),
        "observations_sha256": hashlib.sha256(args.observations.read_bytes()).hexdigest(),
        "model_identities": sorted({row.model_identity for row in observations if row.model_identity}),
        "condition_hashes": {
            row.condition: row.condition_sha256
            for row in observations
            if row.condition_sha256
        },
        "conditions": summarize(probes, observations, utility_profile),
    }
    if utility_profile:
        report["utility_profile"] = {
            "name": utility_profile.name,
            "sha256": utility_profile.sha256,
            "weights": utility_profile.weights,
        }
    if args.baseline:
        report["paired_vs_baseline"] = paired_comparison(
            probes, observations, args.baseline, utility_profile
        )
    rendered = json.dumps(
        _json_safe(report), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
