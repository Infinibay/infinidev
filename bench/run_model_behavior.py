#!/usr/bin/env python3
"""Run behavioral probes against prompt conditions through LiteLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping

from bench.model_behavior import (
    Observation,
    Probe,
    UtilityProfile,
    load_observations,
    load_probes,
)
from bench.probe_manifest import file_sha256, manifest_probe_ids

logger = logging.getLogger(__name__)

MIN_REQUEST_INTERVAL_SECONDS = 2.0

CHOICE_ONLY_INSTRUCTION = """Respond with one JSON object: {"answer":"A"}.
`answer` must be one offered option. Do not add other fields."""

SELF_REPORT_INSTRUCTION = """Respond to the multiple-choice question above.
Use one JSON object with exactly these fields:
{"answer":"A","confidence":0.75,"decision_criterion":"brief criterion",
 "missing_context":"brief missing context or empty string"}
`answer` must be one offered option. `confidence` must be between 0 and 1.
Give a brief externally reportable criterion, not hidden chain-of-thought."""


@dataclass(frozen=True)
class Condition:
    """One immutable system-prompt candidate."""

    name: str
    system_prompt: str | None

    @property
    def sha256(self) -> str:
        raw = self.system_prompt if self.system_prompt is not None else "<no-system-message>"
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class RunConfig:
    """Model route and generation parameters shared by all conditions."""

    model: str
    model_identity: str
    conditions: tuple[Condition, ...]
    temperature: float = 0.0
    max_tokens: int = 300
    api_base: str | None = None
    seed: int = 0
    utility_profile: UtilityProfile | None = None
    min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS
    elicitation_protocol: str = "choice_only"
    provider: str = ""
    reasoning_effort: str | None = None
    option_order_protocol: str = "fixed"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RunConfig:
        raw_conditions = value.get("conditions")
        if not isinstance(raw_conditions, dict) or not raw_conditions:
            raise ValueError("run config needs a non-empty conditions object")
        conditions = tuple(
            Condition(str(name), _condition_prompt(raw))
            for name, raw in raw_conditions.items()
        )
        model = str(value.get("model", "")).strip()
        identity = str(value.get("model_identity", "")).strip()
        if not model or not identity:
            raise ValueError("run config needs model and immutable model_identity")
        min_interval = float(
            value.get("min_request_interval_seconds", MIN_REQUEST_INTERVAL_SECONDS)
        )
        if min_interval < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError(
                "min_request_interval_seconds must be at least "
                f"{MIN_REQUEST_INTERVAL_SECONDS:.1f}"
            )
        protocol = str(value.get("elicitation_protocol", "choice_only"))
        if protocol not in {"choice_only", "self_report"}:
            raise ValueError("elicitation_protocol must be choice_only or self_report")
        option_order_protocol = str(value.get("option_order_protocol", "fixed"))
        if option_order_protocol not in {"fixed", "balanced_rotation"}:
            raise ValueError("option_order_protocol must be fixed or balanced_rotation")
        raw_effort = value.get("reasoning_effort")
        reasoning_effort = str(raw_effort).strip() if raw_effort else None
        if reasoning_effort not in {None, "low", "medium", "high", "xhigh", "max", "ultra"}:
            raise ValueError("unsupported reasoning_effort")
        return cls(
            model=model,
            model_identity=identity,
            conditions=conditions,
            temperature=float(value.get("temperature", 0.0)),
            max_tokens=int(value.get("max_tokens", 300)),
            api_base=str(value["api_base"]) if value.get("api_base") else None,
            seed=int(value.get("seed", 0)),
            utility_profile=(
                UtilityProfile.from_dict(raw_profile)
                if isinstance((raw_profile := value.get("utility_profile")), dict)
                else None
            ),
            min_request_interval_seconds=min_interval,
            elicitation_protocol=protocol,
            provider=str(value.get("provider", "")).strip(),
            reasoning_effort=reasoning_effort,
            option_order_protocol=option_order_protocol,
        )


@dataclass(frozen=True)
class ModelReply:
    """Provider-neutral completion content and accounting."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


Completion = Callable[[RunConfig, Condition, Probe], ModelReply]


def _condition_prompt(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("system_prompt"), str):
        return str(value["system_prompt"])
    raise ValueError("each condition must be a prompt string or system_prompt object")


def parse_model_reply(
    text: str, choices: Mapping[str, str], protocol: str = "self_report"
) -> dict[str, object]:
    """Extract and validate the first JSON object in a provider response."""
    payload = _first_json_object(text)
    answer = str(payload.get("answer", "")).upper().strip()
    if answer not in choices:
        raise ValueError(f"answer must be one of {sorted(choices)}")
    confidence: float | None = None
    if protocol == "self_report":
        confidence = float(payload.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
    return {
        "answer": answer,
        "confidence": confidence,
        "decision_criterion": (
            str(payload.get("decision_criterion", "")) if protocol == "self_report" else ""
        ),
        "missing_context": (
            str(payload.get("missing_context", "")) if protocol == "self_report" else ""
        ),
    }


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
    raise ValueError("response did not contain a JSON object")


def probe_messages(
    config: RunConfig, condition: Condition, probe: Probe
) -> list[dict[str, str]]:
    """Build a fresh, isolated conversation containing only the current probe."""
    profile_context = (
        "\n\nUser requirements and preferences for this scenario:\n"
        f"{config.utility_profile.rendered_for_model()}"
        if probe.evaluation_mode == "preference" and config.utility_profile
        else ""
    )
    messages: list[dict[str, str]] = []
    if condition.system_prompt is not None:
        messages.append({"role": "system", "content": condition.system_prompt})
    messages.append(
        {
            "role": "user",
            "content": (
                f"{probe.rendered_prompt()}{profile_context}\n\n"
                f"{_response_instruction(config.elicitation_protocol)}"
            ),
        }
    )
    return messages


def _response_instruction(protocol: str) -> str:
    return CHOICE_ONLY_INSTRUCTION if protocol == "choice_only" else SELF_REPORT_INSTRUCTION


def run_one(
    probe: Probe,
    condition: Condition,
    config: RunConfig,
    repetition: int,
    completion: Completion,
    *,
    dataset_sha256: str = "",
    manifest_sha256: str = "",
) -> Observation:
    """Run one probe and convert provider or validation failures to observations."""
    started = time.monotonic()
    try:
        presented_probe, choice_mapping, presentation_id = present_probe(
            probe, repetition, config.option_order_protocol, config.seed
        )
        reply = completion(config, condition, presented_probe)
        parsed = parse_model_reply(
            reply.text, presented_probe.choices, config.elicitation_protocol
        )
        provider_answer = str(parsed["answer"])
        canonical_answer = choice_mapping[provider_answer]
        return Observation(
            probe_id=probe.id,
            condition=condition.name,
            answer=canonical_answer,
            confidence=(
                float(parsed["confidence"])
                if parsed["confidence"] is not None
                else None
            ),
            latency_seconds=time.monotonic() - started,
            repetition=repetition,
            model_identity=config.model_identity,
            condition_sha256=condition.sha256,
            response_text=reply.text,
            decision_criterion=str(parsed["decision_criterion"]),
            missing_context=str(parsed["missing_context"]),
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
            utility_profile=(
                config.utility_profile.name if config.utility_profile else ""
            ),
            utility_profile_sha256=(
                config.utility_profile.sha256 if config.utility_profile else ""
            ),
            elicitation_protocol=config.elicitation_protocol,
            option_order_protocol=config.option_order_protocol,
            provider_answer=provider_answer,
            choice_mapping=choice_mapping,
            presentation_id=presentation_id,
            dataset_sha256=dataset_sha256,
            manifest_sha256=manifest_sha256,
        )
    except Exception as exc:  # provider failures are dataset outcomes
        logger.warning("Probe %s/%s failed: %s", probe.id, condition.name, exc)
        return Observation(
            probe_id=probe.id,
            condition=condition.name,
            answer="",
            confidence=None,
            latency_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
            repetition=repetition,
            model_identity=config.model_identity,
            condition_sha256=condition.sha256,
            utility_profile=(
                config.utility_profile.name if config.utility_profile else ""
            ),
            utility_profile_sha256=(
                config.utility_profile.sha256 if config.utility_profile else ""
            ),
            elicitation_protocol=config.elicitation_protocol,
            option_order_protocol=config.option_order_protocol,
            dataset_sha256=dataset_sha256,
            manifest_sha256=manifest_sha256,
        )


def present_probe(
    probe: Probe,
    repetition: int,
    protocol: str,
    seed: int,
) -> tuple[Probe, dict[str, str], str]:
    """Return the displayed choices and their canonical action-key mapping."""
    labels = list(probe.choices)
    if protocol == "fixed":
        mapping = {label: label for label in labels}
        return probe, mapping, "fixed"
    if protocol != "balanced_rotation":
        raise ValueError(f"unsupported option order protocol: {protocol}")
    digest = hashlib.sha256(f"{seed}:{probe.id}".encode()).digest()
    offset = (int.from_bytes(digest[:4], "big") + repetition) % len(labels)
    canonical_order = labels[offset:] + labels[:offset]
    mapping = dict(zip(labels, canonical_order, strict=True))
    displayed_choices = {
        displayed: probe.choices[canonical]
        for displayed, canonical in mapping.items()
    }
    return (
        replace(probe, choices=displayed_choices),
        mapping,
        f"balanced_rotation:{offset}",
    )


def pending_runs(
    probes: Iterable[Probe],
    conditions: Iterable[Condition],
    repetitions: int,
    completed: Iterable[Observation],
    utility_profile_sha256: str = "",
    elicitation_protocol: str = "self_report",
    option_order_protocol: str = "fixed",
    dataset_sha256: str = "",
    manifest_sha256: str = "",
) -> list[tuple[Probe, Condition, int]]:
    """Return runs not already present with the same condition hash."""
    done = {
        (
            row.probe_id,
            row.condition,
            row.condition_sha256,
            row.utility_profile_sha256,
            row.elicitation_protocol,
            row.option_order_protocol,
            row.dataset_sha256,
            row.manifest_sha256,
            row.repetition,
        )
        for row in completed
    }
    return [
        (probe, condition, repetition)
        for probe in probes
        for condition in conditions
        for repetition in range(repetitions)
        if (
            probe.id,
            condition.name,
            condition.sha256,
            utility_profile_sha256,
            elicitation_protocol,
            option_order_protocol,
            dataset_sha256,
            manifest_sha256,
            repetition,
        ) not in done
    ]


def select_probes(
    probes: Iterable[Probe],
    *,
    split: str | None = None,
    categories: Iterable[str] = (),
    probe_ids: Iterable[str] = (),
    evaluation_mode: str | None = None,
    include_drafts: bool = False,
) -> list[Probe]:
    """Select executable probes, excluding unapproved content by default."""
    selected = list(probes)
    if not include_drafts:
        selected = [probe for probe in selected if probe.review_status == "approved"]
    if split:
        selected = [probe for probe in selected if probe.split == split]
    category_set = set(categories)
    if category_set:
        selected = [probe for probe in selected if probe.category in category_set]
    probe_id_set = set(probe_ids)
    if probe_id_set:
        selected = [probe for probe in selected if probe.id in probe_id_set]
    if evaluation_mode:
        selected = [
            probe for probe in selected if probe.evaluation_mode == evaluation_mode
        ]
    return selected


def validate_preference_context(
    probes: Iterable[Probe],
    config: RunConfig,
    *,
    allow_unprofiled: bool = False,
) -> None:
    """Require an explicit opt-in before observing unprofiled preference priors."""
    if not any(probe.evaluation_mode == "preference" for probe in probes):
        return
    if config.utility_profile is None and not allow_unprofiled:
        raise ValueError(
            "preference probes require run config utility_profile or explicit "
            "--allow-unprofiled-preferences for a non-scored raw-prior baseline"
        )


def validate_option_order_protocol(
    probes: Iterable[Probe], config: RunConfig, repetitions: int
) -> None:
    """Require complete Latin rotations rather than partially balanced claims."""
    if config.option_order_protocol != "balanced_rotation":
        return
    incompatible = [
        probe.id for probe in probes if repetitions % len(probe.choices) != 0
    ]
    if incompatible:
        raise ValueError(
            "balanced_rotation requires repetitions divisible by each probe's choice count; "
            f"incompatible probes: {', '.join(incompatible)}"
        )


def select_probe_checkpoint(probes: list[Probe], max_probes: int) -> list[Probe]:
    """Limit a campaign by whole probes so every selected repetition can complete."""
    if max_probes < 0:
        raise ValueError("max_probes cannot be negative")
    return probes[:max_probes] if max_probes else probes


def validate_run_checkpoint(config: RunConfig, *, max_runs: int, max_probes: int) -> None:
    """Prevent individual-call truncation from breaking counterbalanced cycles."""
    if max_runs < 0:
        raise ValueError("max_runs cannot be negative")
    if max_runs and max_probes:
        raise ValueError("use either max_runs or max_probes, not both")
    if max_runs and config.option_order_protocol == "balanced_rotation":
        raise ValueError(
            "balanced_rotation cannot use max_runs; use max_probes to retain complete cycles"
        )


def litellm_completion(config: RunConfig, condition: Condition, probe: Probe) -> ModelReply:
    """Call the configured model through LiteLLM without storing credentials."""
    import litellm

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": probe_messages(config, condition, probe),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "num_retries": 0,
        "caching": False,
    }
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if config.provider:
        from infinidev.config.llm import apply_provider_transport

        apply_provider_transport(kwargs, config.provider)
    # Importing infinidev.config.llm installs the shared subscription response
    # boundary. Resolve completion only after that import: retaining a function
    # reference captured beforehand bypasses its forced-streaming repair.
    response = litellm.completion(**kwargs)
    text = str(response.choices[0].message.content or "")
    usage = getattr(response, "usage", None)
    return ModelReply(
        text=text,
        input_tokens=_usage_value(usage, "prompt_tokens"),
        output_tokens=_usage_value(usage, "completion_tokens"),
    )


def _usage_value(usage: object, key: str) -> int | None:
    value = getattr(usage, key, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(key)
    return int(value) if value is not None else None


def append_observation(path: Path, observation: Observation) -> None:
    """Durably append one completed call so interrupted runs can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(observation), ensure_ascii=False) + "\n")
        stream.flush()


def is_rate_limit_error(error: str | None) -> bool:
    """Recognize common provider rate-limit failures without provider imports."""
    normalized = (error or "").lower()
    return any(
        marker in normalized
        for marker in ("ratelimit", "rate limit", "status code: 429", "status_code=429")
    )


def run_sequentially(
    work: Iterable[tuple[Probe, Condition, int]],
    config: RunConfig,
    completion: Completion,
    record: Callable[[Observation], None],
    *,
    dataset_sha256: str = "",
    manifest_sha256: str = "",
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    """Run exactly one request at a time, paced, and stop on rate limiting."""
    if config.min_request_interval_seconds < MIN_REQUEST_INTERVAL_SECONDS:
        raise ValueError(
            "min_request_interval_seconds must be at least "
            f"{MIN_REQUEST_INTERVAL_SECONDS:.1f}"
        )
    completed = 0
    last_started: float | None = None
    for probe, condition, repetition in work:
        now = monotonic()
        if last_started is not None:
            remaining = config.min_request_interval_seconds - (now - last_started)
            if remaining > 0:
                sleep(remaining)
        last_started = monotonic()
        observation = run_one(
            probe,
            condition,
            config,
            repetition,
            completion,
            dataset_sha256=dataset_sha256,
            manifest_sha256=manifest_sha256,
        )
        record(observation)
        completed += 1
        if is_rate_limit_error(observation.error):
            logger.error(
                "Rate limit encountered at %s/%s; stopping without automatic retries",
                probe.id,
                condition.name,
            )
            break
    return completed


@contextmanager
def exclusive_behavior_run() -> Iterator[None]:
    """Prevent any two behavior-study provider calls from overlapping on this host."""
    from infinidev.engine.subscription_safety import subscription_single_flight

    with subscription_single_flight():
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probes", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("calibration", "validation"))
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--probe-id", action="append", default=[])
    parser.add_argument(
        "--manifest",
        type=Path,
        help="run exactly the probe ids frozen in a dataset-bound manifest",
    )
    parser.add_argument("--evaluation-mode", choices=("normative", "preference"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument(
        "--max-probes",
        type=int,
        default=0,
        help="limit by whole probes while retaining every condition and repetition",
    )
    parser.add_argument(
        "--include-drafts",
        action="store_true",
        help="run unapproved drafts for exploratory analysis",
    )
    parser.add_argument(
        "--allow-unprofiled-preferences",
        action="store_true",
        help=(
            "observe preference choices without a user profile as a non-scored raw-prior "
            "baseline"
        ),
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")

    config = RunConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    dataset_sha256 = file_sha256(args.probes)
    manifest_sha256 = file_sha256(args.manifest) if args.manifest else ""
    try:
        validate_run_checkpoint(
            config, max_runs=args.max_runs, max_probes=args.max_probes
        )
    except ValueError as exc:
        parser.error(str(exc))
    probe_catalog = load_probes(args.probes)
    manifest_ids: list[str] = []
    if args.manifest:
        if args.probe_id or args.category or args.split or args.evaluation_mode:
            parser.error("--manifest cannot be combined with probe selection filters")
        try:
            manifest_ids = manifest_probe_ids(
                json.loads(args.manifest.read_text(encoding="utf-8")),
                probe_catalog,
                dataset_sha256=dataset_sha256,
            )
        except (ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
    probes = select_probes(
        probe_catalog.values(),
        split=args.split,
        categories=args.category,
        probe_ids=manifest_ids or args.probe_id,
        evaluation_mode=args.evaluation_mode,
        include_drafts=args.include_drafts,
    )
    if manifest_ids and [probe.id for probe in probes] != manifest_ids:
        by_id = {probe.id: probe for probe in probes}
        missing = [probe_id for probe_id in manifest_ids if probe_id not in by_id]
        if missing:
            parser.error(
                "manifest probes were filtered out; drafts require --include-drafts: "
                + ", ".join(missing)
            )
        probes = [by_id[probe_id] for probe_id in manifest_ids]
    if not probes:
        parser.error("no probes matched; drafts require --include-drafts")
    probes = select_probe_checkpoint(probes, args.max_probes)
    try:
        validate_preference_context(
            probes,
            config,
            allow_unprofiled=args.allow_unprofiled_preferences,
        )
        validate_option_order_protocol(probes, config, args.repetitions)
    except ValueError as exc:
        parser.error(str(exc))
    completed = load_observations(args.output) if args.output.exists() else []
    work = pending_runs(
        probes,
        config.conditions,
        args.repetitions,
        completed,
        config.utility_profile.sha256 if config.utility_profile else "",
        config.elicitation_protocol,
        config.option_order_protocol,
        dataset_sha256,
        manifest_sha256,
    )
    random.Random(config.seed).shuffle(work)
    if args.max_runs > 0:
        work = work[: args.max_runs]
    logger.info(
        "Running %d calls sequentially with %.2fs minimum start interval",
        len(work),
        config.min_request_interval_seconds,
    )
    with exclusive_behavior_run():
        completed_count = run_sequentially(
            work,
            config,
            litellm_completion,
            lambda observation: append_observation(args.output, observation),
            dataset_sha256=dataset_sha256,
            manifest_sha256=manifest_sha256,
        )
    logger.info("Completed %d/%d calls", completed_count, len(work))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
