#!/usr/bin/env python3
"""Collect isolated harness-feedback hypotheses sequentially through LiteLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

try:
    from bench.harness_feedback import (
        FeedbackCase,
        FeedbackObservation,
        HarnessFeedback,
        load_cases,
        load_observations,
    )
except ModuleNotFoundError:
    from harness_feedback import (  # type: ignore[no-redef]
        FeedbackCase,
        FeedbackObservation,
        HarnessFeedback,
        load_cases,
        load_observations,
    )


logger = logging.getLogger(__name__)
MIN_REQUEST_INTERVAL_SECONDS = 2.0


def is_rate_limit_error(error: str | None) -> bool:
    normalized = (error or "").lower()
    return any(
        marker in normalized
        for marker in ("ratelimit", "rate limit", "status code: 429", "status_code=429")
    )


@contextmanager
def exclusive_behavior_run():
    """Share the global subscription single-flight lock with behavior probes."""
    from infinidev.engine.subscription_safety import subscription_single_flight

    with subscription_single_flight():
        yield


@dataclass(frozen=True)
class FeedbackRunConfig:
    """Immutable raw-feedback route with no system-prompt treatment."""

    model: str
    model_identity: str
    provider: str = ""
    api_base: str | None = None
    max_tokens: int = 700
    temperature: float = 0.0
    reasoning_effort: str | None = None
    min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS
    seed: int = 0

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FeedbackRunConfig:
        config = cls(
            model=str(value.get("model", "")).strip(),
            model_identity=str(value.get("model_identity", "")).strip(),
            provider=str(value.get("provider", "")).strip(),
            api_base=str(value["api_base"]) if value.get("api_base") else None,
            max_tokens=int(value.get("max_tokens", 700)),
            temperature=float(value.get("temperature", 0.0)),
            reasoning_effort=(
                str(value["reasoning_effort"]).strip()
                if value.get("reasoning_effort")
                else None
            ),
            min_request_interval_seconds=float(
                value.get("min_request_interval_seconds", MIN_REQUEST_INTERVAL_SECONDS)
            ),
            seed=int(value.get("seed", 0)),
        )
        if not config.model or not config.model_identity:
            raise ValueError("feedback config needs model and immutable model_identity")
        if config.max_tokens < 1:
            raise ValueError("feedback max_tokens must be positive")
        if config.min_request_interval_seconds < MIN_REQUEST_INTERVAL_SECONDS:
            raise ValueError("feedback request interval must be at least 2.0 seconds")
        if config.reasoning_effort not in {
            None, "low", "medium", "high", "xhigh", "max", "ultra"
        }:
            raise ValueError("unsupported feedback reasoning_effort")
        return config


@dataclass(frozen=True)
class FeedbackReply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


Completion = Callable[[FeedbackRunConfig, FeedbackCase], FeedbackReply]


def litellm_completion(config: FeedbackRunConfig, case: FeedbackCase) -> FeedbackReply:
    """Send exactly one user message with no system prompt and no retries."""
    import litellm

    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": case.rendered_prompt()}],
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
    response = litellm.completion(**kwargs)
    usage = getattr(response, "usage", None)
    return FeedbackReply(
        text=str(response.choices[0].message.content or ""),
        input_tokens=_usage(usage, "prompt_tokens"),
        output_tokens=_usage(usage, "completion_tokens"),
    )


def _usage(usage: object, key: str) -> int | None:
    value = getattr(usage, key, None)
    if value is None and isinstance(usage, dict):
        value = usage.get(key)
    return int(value) if value is not None else None


def run_one(
    case: FeedbackCase,
    config: FeedbackRunConfig,
    repetition: int,
    completion: Completion,
    *,
    dataset_sha256: str,
) -> FeedbackObservation:
    started = time.monotonic()
    response_text = ""
    try:
        reply = completion(config, case)
        response_text = reply.text
        feedback = HarnessFeedback.from_text(reply.text)
        return FeedbackObservation(
            case_id=case.id,
            case_sha256=case.sha256,
            model_identity=config.model_identity,
            repetition=repetition,
            response_text=reply.text,
            feedback=feedback,
            dataset_sha256=dataset_sha256,
            latency_seconds=time.monotonic() - started,
            input_tokens=reply.input_tokens,
            output_tokens=reply.output_tokens,
        )
    except Exception as exc:
        return FeedbackObservation(
            case_id=case.id,
            case_sha256=case.sha256,
            model_identity=config.model_identity,
            repetition=repetition,
            response_text=response_text,
            feedback=None,
            dataset_sha256=dataset_sha256,
            latency_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def pending_runs(
    cases: Iterable[FeedbackCase],
    repetitions: int,
    completed: Iterable[FeedbackObservation],
    *,
    model_identity: str,
) -> list[tuple[FeedbackCase, int]]:
    done = {
        (row.case_id, row.case_sha256, row.model_identity, row.repetition)
        for row in completed
    }
    return [
        (case, repetition)
        for case in cases
        for repetition in range(repetitions)
        if (case.id, case.sha256, model_identity, repetition) not in done
    ]


def append_observation(path: Path, row: FeedbackObservation) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run_sequentially(
    work: Iterable[tuple[FeedbackCase, int]],
    config: FeedbackRunConfig,
    completion: Completion,
    record: Callable[[FeedbackObservation], None],
    *,
    dataset_sha256: str,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> int:
    if config.min_request_interval_seconds < MIN_REQUEST_INTERVAL_SECONDS:
        raise ValueError("feedback request interval must be at least 2.0 seconds")
    count = 0
    last_started: float | None = None
    for case, repetition in work:
        now = monotonic()
        if last_started is not None:
            remaining = config.min_request_interval_seconds - (now - last_started)
            if remaining > 0:
                sleep(remaining)
        last_started = monotonic()
        row = run_one(
            case, config, repetition, completion, dataset_sha256=dataset_sha256
        )
        record(row)
        count += 1
        if row.error and (
            is_rate_limit_error(row.error)
            or any(marker in row.error.lower() for marker in ("connection", "timeout", "dns"))
        ):
            logger.error("Stopping feedback campaign after provider error: %s", row.error)
            break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", choices=("calibration", "validation"))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    config = FeedbackRunConfig.from_dict(json.loads(args.config.read_text(encoding="utf-8")))
    dataset_sha = hashlib.sha256(args.cases.read_bytes()).hexdigest()
    cases = [
        case for case in load_cases(args.cases)
        if (args.include_drafts or case.review_status == "approved")
        and (not args.split or case.split == args.split)
    ]
    if not cases:
        parser.error("no feedback cases matched; drafts require --include-drafts")
    completed = load_observations(args.output) if args.output.exists() else []
    work = pending_runs(
        cases, args.repetitions, completed, model_identity=config.model_identity
    )
    random.Random(config.seed).shuffle(work)
    with exclusive_behavior_run():
        count = run_sequentially(
            work,
            config,
            litellm_completion,
            lambda row: append_observation(args.output, row),
            dataset_sha256=dataset_sha,
        )
    logger.info("Completed %d/%d feedback calls", count, len(work))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
