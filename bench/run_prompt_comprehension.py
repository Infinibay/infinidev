#!/usr/bin/env python3
"""Run isolated prompt-comprehension cases sequentially through LiteLLM."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import getpass
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

from bench.prompt_comprehension import (
    ComprehensionCase,
    ComprehensionCondition,
    ComprehensionObservation,
    append_observation,
    comprehension_messages,
    load_cases,
    parse_comprehension_reply,
)
from bench.prompt_comprehension_campaign_audit import tuple_id

logger = logging.getLogger(__name__)
DEFAULT_INTERVAL_SECONDS = 1.0
MIN_INTERVAL_SECONDS = 0.75


@dataclass(frozen=True)
class ComprehensionRunConfig:
    """Pinned provider route and immutable identity for one raw-only baseline."""

    provider: str
    model: str
    revision: str
    model_identity: str
    manifest_id: str
    dataset_sha256: str
    dataset_stage: Literal["approved", "exploratory_draft"]
    conditions: tuple[ComprehensionCondition, ...]
    min_request_interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    max_tokens: int = 1800
    temperature: float = 0.0
    reasoning_effort: str | None = None
    api_base: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ComprehensionRunConfig:
        raw_conditions = value.get("conditions")
        if not isinstance(raw_conditions, dict) or not raw_conditions:
            raise ValueError("comprehension config needs conditions")
        provider = str(value.get("provider", "")).strip()
        model = str(value.get("model", "")).strip()
        revision = str(value.get("revision", "")).strip()
        identity = str(value.get("model_identity", "")).strip()
        manifest_id = str(value.get("manifest_id", "")).strip()
        approved_digest = str(value.get("approved_dataset_sha256", "")).strip()
        exploratory_digest = str(value.get("dataset_sha256", "")).strip()
        if approved_digest and exploratory_digest:
            raise ValueError(
                "manifest cannot mix approved_dataset_sha256 with dataset_sha256"
            )
        if approved_digest:
            dataset_sha256 = approved_digest
            dataset_stage: Literal["approved", "exploratory_draft"] = "approved"
        else:
            dataset_sha256 = exploratory_digest
            dataset_stage_value = str(value.get("dataset_stage", "")).strip()
            if dataset_stage_value != "exploratory_draft":
                raise ValueError(
                    "unapproved dataset needs dataset_stage=exploratory_draft"
                )
            dataset_stage = "exploratory_draft"
        if not all((provider, model, revision, identity)):
            raise ValueError(
                "comprehension config needs explicit provider, model, revision, and model_identity"
            )
        if not manifest_id:
            raise ValueError("comprehension config needs an explicit new manifest_id")
        if len(dataset_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in dataset_sha256
        ):
            raise ValueError("dataset SHA-256 must be a lowercase digest")
        interval = float(value.get("min_request_interval_seconds", DEFAULT_INTERVAL_SECONDS))
        if interval < MIN_INTERVAL_SECONDS:
            raise ValueError(f"minimum interval is {MIN_INTERVAL_SECONDS} seconds")
        if set(raw_conditions) != {"raw"} or raw_conditions["raw"] is not None:
            raise ValueError("baseline condition must be exactly raw with no system prompt")
        conditions = (ComprehensionCondition.from_value("raw", None),)
        return cls(
            provider=provider,
            model=model,
            revision=revision,
            model_identity=identity,
            manifest_id=manifest_id,
            dataset_sha256=dataset_sha256,
            dataset_stage=dataset_stage,
            conditions=conditions,
            min_request_interval_seconds=interval,
            max_tokens=int(value.get("max_tokens", 1800)),
            temperature=float(value.get("temperature", 0.0)),
            reasoning_effort=(str(value["reasoning_effort"]) if value.get("reasoning_effort") else None),
            api_base=(str(value["api_base"]) if value.get("api_base") else None),
        )


@dataclass(frozen=True)
class ProviderReply:
    text: str
    input_tokens: int | None
    output_tokens: int | None


FailureType = Literal["provider_error", "parse_error"]


@dataclass(frozen=True)
class AttemptResult:
    """One terminal provider attempt and its operational failure category."""

    observation: ComprehensionObservation
    failure_type: FailureType | None


Completion = Callable[
    [ComprehensionRunConfig, ComprehensionCondition, ComprehensionCase], ProviderReply
]
AttemptRecorder = Callable[[AttemptResult], None]


@dataclass(frozen=True)
class BaselinePlan:
    """Byte-verified inputs and already terminal tuples for one baseline run."""

    config: ComprehensionRunConfig
    cases: tuple[ComprehensionCase, ...]
    dataset_sha256: str
    manifest_sha256: str
    terminal_tuple_ids: frozenset[str]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_line(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_create(path: Path, value: object) -> None:
    """Exclusively create and sync a JSON claim before a campaign may make calls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_canonical_json_line(value))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _durable_append(path: Path, value: object) -> None:
    """Append one complete JSONL row and force it to stable storage before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    with path.open("ab") as stream:
        stream.write(_canonical_json_line(value))
        stream.flush()
        os.fsync(stream.fileno())
    if not existed:
        _fsync_directory(path.parent)


def _claim_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f".{manifest_path.name}.prompt-comprehension-claim.json")


def _read_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"{label} is not valid readable UTF-8 JSON") from err
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _manifest_claim(
    manifest_path: Path,
    ledger_path: Path,
    config: ComprehensionRunConfig,
    manifest_sha256: str,
    dataset_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "manifest_id": config.manifest_id,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "dataset_sha256": dataset_sha256,
        "ledger_path": str(ledger_path.resolve()),
    }


def _ledger_row(
    result: AttemptResult,
    config: ComprehensionRunConfig,
    *,
    manifest_sha256: str,
    dataset_sha256: str,
) -> dict[str, object]:
    observation = result.observation
    model_key = (config.provider, config.model, config.revision, config.model_identity)
    return {
        "tuple_id": tuple_id(observation.case_id, model_key, manifest_sha256, dataset_sha256),
        "case_id": observation.case_id,
        "condition": observation.condition,
        "dataset_sha256": dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "provider": config.provider,
        "model": config.model,
        "revision": config.revision,
        "model_identity": config.model_identity,
        "status": "failure" if result.failure_type else "success",
        "terminal": True,
        "failure": (
            {"type": result.failure_type, "message": observation.error}
            if result.failure_type
            else None
        ),
        "observation": asdict(observation),
    }


def append_attempt(
    path: Path,
    result: AttemptResult,
    config: ComprehensionRunConfig,
    *,
    manifest_sha256: str,
    dataset_sha256: str,
) -> None:
    """Durably persist one typed terminal attempt before execution can continue."""
    _durable_append(
        path,
        _ledger_row(
            result,
            config,
            manifest_sha256=manifest_sha256,
            dataset_sha256=dataset_sha256,
        ),
    )


def _load_terminal_tuple_ids(
    ledger_path: Path,
    config: ComprehensionRunConfig,
    cases: tuple[ComprehensionCase, ...],
    *,
    manifest_sha256: str,
    dataset_sha256: str,
) -> frozenset[str]:
    if not ledger_path.exists():
        return frozenset()
    data = ledger_path.read_bytes()
    if not data:
        return frozenset()
    if not data.endswith(b"\n"):
        raise ValueError("attempt ledger is truncated")
    model_key = (config.provider, config.model, config.revision, config.model_identity)
    expected = {
        tuple_id(case.id, model_key, manifest_sha256, dataset_sha256) for case in cases
    }
    observed: set[str] = set()
    for number, raw_line in enumerate(data.splitlines(), start=1):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as err:
            raise ValueError(f"attempt ledger row {number} is invalid JSON") from err
        if not isinstance(row, dict):
            raise ValueError(f"attempt ledger row {number} must be an object")
        stored_id = row.get("tuple_id")
        if not isinstance(stored_id, str) or stored_id not in expected:
            raise ValueError("attempt ledger contains an extra or mixed-identity tuple")
        if stored_id in observed:
            raise ValueError("attempt ledger contains a duplicate terminal tuple")
        if (
            row.get("terminal") is not True
            or row.get("condition") != "raw"
            or row.get("dataset_sha256") != dataset_sha256
            or row.get("manifest_sha256") != manifest_sha256
            or (
                row.get("provider"),
                row.get("model"),
                row.get("revision"),
                row.get("model_identity"),
            )
            != model_key
        ):
            raise ValueError("attempt ledger mixes campaign identities")
        status = row.get("status")
        failure = row.get("failure")
        if status == "success":
            if failure not in (None, {}):
                raise ValueError("successful ledger tuple contains failure metadata")
        elif status == "failure":
            if not isinstance(failure, dict) or failure.get("type") not in {
                "provider_error",
                "parse_error",
            }:
                raise ValueError("failed ledger tuple lacks a terminal failure type")
        else:
            raise ValueError("attempt ledger tuple is not terminally typed")
        observed.add(stored_id)
    return frozenset(observed)


def preflight_baseline(
    cases_path: Path,
    manifest_path: Path,
    ledger_path: Path,
    *,
    resume: bool,
    split: str = "validation",
) -> BaselinePlan:
    """Validate immutable raw-only inputs and claim a new manifest before provider use."""
    dataset_bytes = cases_path.read_bytes()
    manifest_bytes = manifest_path.read_bytes()
    dataset_sha256 = _sha256(dataset_bytes)
    manifest_sha256 = _sha256(manifest_bytes)
    config = ComprehensionRunConfig.from_dict(_read_object(manifest_path, "manifest"))
    if dataset_sha256 != config.dataset_sha256:
        raise ValueError("dataset bytes do not match the manifest SHA-256")
    all_cases = tuple(load_cases(cases_path))
    expected_status = "approved" if config.dataset_stage == "approved" else "draft"
    if not all_cases or any(case.review_status != expected_status for case in all_cases):
        raise ValueError(
            f"{config.dataset_stage} dataset must contain only {expected_status} cases"
        )
    if split not in {"all", "calibration", "validation"}:
        raise ValueError(f"unsupported baseline split: {split}")
    cases = all_cases if split == "all" else tuple(
        case for case in all_cases if case.split == split
    )
    if not cases:
        raise ValueError("baseline selection contains no approved cases")

    claim_path = _claim_path(manifest_path)
    expected_claim = _manifest_claim(
        manifest_path,
        ledger_path,
        config,
        manifest_sha256,
        dataset_sha256,
    )
    if resume:
        if not claim_path.is_file():
            raise ValueError("resume requires an existing durable manifest claim")
        if _read_object(claim_path, "manifest claim") != expected_claim:
            raise ValueError("manifest claim changed or belongs to another campaign")
    else:
        if ledger_path.exists():
            raise ValueError("new campaign requires a new ledger path")
        try:
            _durable_create(claim_path, expected_claim)
        except FileExistsError as err:
            raise ValueError("manifest is not new; use resume for its claimed campaign") from err

    terminal = _load_terminal_tuple_ids(
        ledger_path,
        config,
        cases,
        manifest_sha256=manifest_sha256,
        dataset_sha256=dataset_sha256,
    )
    return BaselinePlan(config, cases, dataset_sha256, manifest_sha256, terminal)


def _usage(usage: object, key: str) -> int | None:
    value = getattr(usage, key, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(key)
    return int(value) if value is not None else None


def litellm_completion(
    config: ComprehensionRunConfig,
    condition: ComprehensionCondition,
    case: ComprehensionCase,
) -> ProviderReply:
    import litellm

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": comprehension_messages(case, condition),
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "num_retries": 0,
        "caching": False,
    }
    if config.api_base:
        kwargs["api_base"] = config.api_base
    if config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    from infinidev.config.llm import apply_provider_transport

    apply_provider_transport(kwargs, config.provider)
    response = litellm.completion(**kwargs)
    usage = getattr(response, "usage", None)
    return ProviderReply(
        str(response.choices[0].message.content or ""),
        _usage(usage, "prompt_tokens"),
        _usage(usage, "completion_tokens"),
    )


def external_api_completion(api_key: str) -> Completion:
    """Build a stateless direct completion route for MiniMax or GLM."""
    from bench.run_prompt_comprehension_review import glm_request, minimax_request, qwen_request

    def complete(
        config: ComprehensionRunConfig,
        condition: ComprehensionCondition,
        case: ComprehensionCase,
    ) -> ProviderReply:
        prompt = comprehension_messages(case, condition)[-1]["content"]
        if config.provider == "minimax":
            text, metadata = minimax_request(
                endpoint=config.api_base or "https://api.minimax.io/v1/chat/completions",
                api_key=api_key,
                model=config.model,
                prompt=prompt,
                timeout=180.0,
                max_completion_tokens=config.max_tokens,
            )
        elif config.provider == "glm":
            text, metadata = glm_request(
                endpoint=(
                    config.api_base
                    or "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"
                ),
                api_key=api_key,
                model=config.model,
                prompt=prompt,
                timeout=180.0,
                max_completion_tokens=config.max_tokens,
                reasoning_effort=config.reasoning_effort or "high",
            )
        elif config.provider == "qwen":
            if not config.api_base:
                raise ValueError("Qwen requires an explicit approved endpoint")
            text, metadata = qwen_request(
                endpoint=config.api_base,
                api_key=api_key,
                model=config.model,
                prompt=prompt,
                timeout=180.0,
                max_completion_tokens=config.max_tokens,
                temperature=config.temperature,
            )
        else:
            raise ValueError(f"unsupported direct provider: {config.provider}")
        usage = metadata.get("usage")
        usage_map = usage if isinstance(usage, dict) else {}
        input_tokens = usage_map.get("prompt_tokens", usage_map.get("input_tokens"))
        output_tokens = usage_map.get("completion_tokens", usage_map.get("output_tokens"))
        return ProviderReply(
            text,
            int(input_tokens) if input_tokens is not None else None,
            int(output_tokens) if output_tokens is not None else None,
        )

    return complete


def run_one(
    config: ComprehensionRunConfig,
    condition: ComprehensionCondition,
    case: ComprehensionCase,
    completion: Completion,
    *,
    dataset_sha256: str,
) -> AttemptResult:
    """Return exactly one terminal result with provider and parser failures separated."""
    started = time.monotonic()
    try:
        reply = completion(config, condition, case)
    except Exception as exc:
        observation = ComprehensionObservation(
            case.id,
            case.category,
            condition.name,
            condition.sha256,
            config.model_identity,
            "",
            {},
            time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
            dataset_sha256=dataset_sha256,
        )
        return AttemptResult(observation, "provider_error")
    try:
        parsed = parse_comprehension_reply(reply.text)
    except ValueError as exc:
        observation = ComprehensionObservation(
            case.id,
            case.category,
            condition.name,
            condition.sha256,
            config.model_identity,
            reply.text,
            {},
            time.monotonic() - started,
            reply.input_tokens,
            reply.output_tokens,
            error=f"{type(exc).__name__}: {exc}",
            dataset_sha256=dataset_sha256,
        )
        return AttemptResult(observation, "parse_error")
    observation = ComprehensionObservation(
        case.id,
        case.category,
        condition.name,
        condition.sha256,
        config.model_identity,
        reply.text,
        parsed,
        time.monotonic() - started,
        reply.input_tokens,
        reply.output_tokens,
        dataset_sha256=dataset_sha256,
    )
    return AttemptResult(observation, None)


def run_sequentially(
    config: ComprehensionRunConfig,
    cases: list[ComprehensionCase] | tuple[ComprehensionCase, ...],
    completion: Completion,
    record: AttemptRecorder,
    *,
    dataset_sha256: str,
    manifest_sha256: str,
    terminal_tuple_ids: frozenset[str] = frozenset(),
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    max_cases: int | None = None,
) -> int:
    """Run pending tuples sequentially, persisting each before stopping on a 429."""
    completed = 0
    last_started: float | None = None
    model_key = (config.provider, config.model, config.revision, config.model_identity)
    for case in cases:
        for condition in config.conditions:
            planned_id = tuple_id(case.id, model_key, manifest_sha256, dataset_sha256)
            if planned_id in terminal_tuple_ids:
                continue
            now = monotonic()
            if last_started is not None:
                remaining = config.min_request_interval_seconds - (now - last_started)
                if remaining > 0:
                    sleep(remaining)
            last_started = monotonic()
            result = run_one(
                config,
                condition,
                case,
                completion,
                dataset_sha256=dataset_sha256,
            )
            record(result)
            completed += 1
            if result.failure_type == "provider_error":
                return completed
            if max_cases is not None and completed >= max_cases:
                return completed
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("ledger", type=Path)
    parser.add_argument(
        "--split",
        choices=("all", "calibration", "validation"),
        default="validation",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args()
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive")
    try:
        plan = preflight_baseline(
            args.cases,
            args.manifest,
            args.ledger,
            resume=args.resume,
            split=args.split,
        )
    except (OSError, ValueError) as err:
        parser.error(str(err))
    completion: Completion = litellm_completion
    lock = nullcontext()
    if plan.config.provider == "openai_subscription":
        from infinidev.engine.subscription_safety import subscription_single_flight

        lock = subscription_single_flight()
    elif plan.config.provider in {"minimax", "glm", "qwen"}:
        provider_name = {
            "minimax": "MiniMax",
            "glm": "GLM",
            "qwen": "Qwen",
        }[plan.config.provider]
        api_key = getpass.getpass(f"{provider_name} API key: ")
        if not api_key:
            parser.error(f"{provider_name} API key is required")
        completion = external_api_completion(api_key)
    with lock:
        run_sequentially(
            plan.config,
            plan.cases,
            completion,
            lambda result: append_attempt(
                args.ledger,
                result,
                plan.config,
                manifest_sha256=plan.manifest_sha256,
                dataset_sha256=plan.dataset_sha256,
            ),
            dataset_sha256=plan.dataset_sha256,
            manifest_sha256=plan.manifest_sha256,
            terminal_tuple_ids=plan.terminal_tuple_ids,
            max_cases=args.max_cases,
        )


if __name__ == "__main__":
    main()
